use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State},
    http::header,
    response::Response,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::{AppState, auth::AuthUser, error::AppError, models::Algorithm};

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub document_id: String,
    #[serde(default)]
    pub mode: AlgorithmGenerationMode,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmGenerationMode {
    Structured,
    Source,
    Physician,
}

impl Default for AlgorithmGenerationMode {
    fn default() -> Self {
        Self::Physician
    }
}

impl AlgorithmGenerationMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Structured => "structured",
            Self::Source => "source",
            Self::Physician => "physician",
        }
    }
}

#[derive(Deserialize)]
pub struct ExportPdfRequest {
    pub markdown: String,
}

#[derive(Deserialize)]
pub struct StreamQuery {
    pub after: Option<i64>,
}

pub async fn generate(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<GenerateRequest>,
) -> Result<Response, AppError> {
    user.require_password_changed()?;
    let generation_mode = body.mode.as_str();
    // Fetch document with full_text from Rust DB
    let (full_text, diagnosis_name) = sqlx::query_as::<_, (String, String)>(
        "SELECT
            COALESCE(d.full_text, src.full_text, ''),
            COALESCE(d.diagnosis_name, src.diagnosis_name, 'Неизвестное заболевание')
         FROM documents d
         LEFT JOIN document_sources src ON src.content_hash = d.content_hash
         WHERE d.id = ? AND d.user_id = ? AND d.status = 'ready'",
    )
    .bind(&body.document_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::BadRequest("Document not found or not ready".into()))?;

    if full_text.is_empty() {
        return Err(AppError::BadRequest(
            "Document has no extracted text".into(),
        ));
    }

    let running = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM algorithms a JOIN documents d ON d.id = a.document_id WHERE a.document_id = ? AND d.user_id = ? AND a.status = 'running'",
    )
    .bind(&body.document_id)
    .bind(&user.id)
    .fetch_one(&state.db)
    .await
    .unwrap_or(0);
    if running > 0 {
        return Err(AppError::BadRequest(
            "Algorithm generation is already running".into(),
        ));
    }

    let algo_id = Uuid::new_v4().to_string();
    let mut txn = state.db.begin().await?;
    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'algorithm' AND target_id IN (
            SELECT a.id FROM algorithms a JOIN documents d ON d.id = a.document_id
            WHERE a.document_id = ? AND d.user_id = ?
        )",
    )
    .bind(&body.document_id)
    .bind(&user.id)
    .execute(&mut *txn)
    .await?;
    sqlx::query(
        "DELETE FROM algorithms WHERE document_id = ? AND document_id IN (SELECT id FROM documents WHERE user_id = ?)",
    )
        .bind(&body.document_id)
        .bind(&user.id)
        .execute(&mut *txn)
        .await?;
    sqlx::query(
        "INSERT INTO algorithms (
            id, document_id, content_markdown, status, generation_mode
         ) VALUES (?, ?, '', 'running', ?)",
    )
    .bind(&algo_id)
    .bind(&body.document_id)
    .bind(generation_mode)
    .execute(&mut *txn)
    .await?;
    txn.commit().await?;
    emit_algorithm_changed(
        &state,
        &user.id,
        &body.document_id,
        Some(&algo_id),
        "running",
    )
    .await?;
    crate::routes::monitoring::log_audit_event(
        &state.db,
        Some(&user),
        Some(&user.id),
        Some(&user.username),
        Some(&user.id),
        "algorithm_generation_started",
        "algorithm",
        Some(&algo_id),
        serde_json::json!({
            "document_id": &body.document_id,
            "algorithm_id": &algo_id,
            "mode": generation_mode,
        }),
    )
    .await?;

    // Send full_text + diagnosis_name to Python (stateless, SSE stream — no timeout)
    let resp = state
        .python
        .stream_client()
        .post(state.python.url("/algorithms/generate"))
        .json(&serde_json::json!({
            "full_text": full_text,
            "diagnosis_name": diagnosis_name,
            "mode": generation_mode,
        }))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let response_body = resp.text().await.unwrap_or_default();
        sqlx::query("UPDATE algorithms SET status = 'error', content_markdown = ? WHERE id = ?")
            .bind(format!("AI service error {status}: {response_body}"))
            .bind(&algo_id)
            .execute(&state.db)
            .await
            .ok();
        crate::routes::monitoring::log_audit_event(
            &state.db,
            Some(&user),
            Some(&user.id),
            Some(&user.username),
            Some(&user.id),
            "algorithm_generation_error",
            "algorithm",
            Some(&algo_id),
            serde_json::json!({
                "document_id": &body.document_id,
                "algorithm_id": &algo_id,
                "mode": generation_mode,
                "status": status.as_u16(),
            }),
        )
        .await
        .ok();
        return Err(AppError::AiServiceUnavailable(format!(
            "AI service error {status}: {response_body}"
        )));
    }

    let db = state.db.clone();
    let stream_user_id = user.id.clone();
    let stream_username = user.username.clone();
    let stream_document_id = body.document_id.clone();

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let update_db = db.clone();
        let update_algo_id = algo_id.clone();
        let event_db = db.clone();
        let event_algo_id = algo_id.clone();
        let relay = crate::sse::relay_and_collect_with_events(
            resp,
            &tx,
            move |seq, token| {
                let event_db = event_db.clone();
                let event_algo_id = event_algo_id.clone();
                async move {
                    if let Err(e) = crate::sse::insert_generation_event(
                        &event_db,
                        "algorithm",
                        &event_algo_id,
                        seq,
                        "token",
                        &token,
                    )
                    .await
                    {
                        tracing::error!("Failed to save algorithm stream event: {}", e);
                    }
                }
            },
            move |partial, seq| {
            let update_db = update_db.clone();
            let update_algo_id = update_algo_id.clone();
            async move {
                if let Err(e) = sqlx::query(
                    "UPDATE algorithms SET content_markdown = ?, stream_seq = ? WHERE id = ? AND status = 'running'",
                )
                .bind(partial)
                .bind(seq)
                .bind(update_algo_id)
                .execute(&update_db)
                .await
                {
                    tracing::error!("Failed to update algorithm progress: {}", e);
                }
            }
            },
        )
        .await;
        let collected = relay.collected;
        let token_seq = relay.token_seq;

        if let Some(error_message) = relay.error {
            let event = crate::sse::format_sse_event(token_seq + 1, "error", &error_message);
            let _ = tx.send(Ok(event)).await;
            if let Err(e) = sqlx::query(
                "UPDATE algorithms SET status = 'error' WHERE id = ? AND status = 'running'",
            )
            .bind(&algo_id)
            .execute(&db)
            .await
            {
                tracing::error!("Failed to mark algorithm as error: {}", e);
            }
            if let Err(e) = crate::sse::insert_generation_event(
                &db,
                "algorithm",
                &algo_id,
                token_seq + 1,
                "error",
                &error_message,
            )
            .await
            {
                tracing::error!("Failed to save algorithm upstream error event: {}", e);
            }
            if let Err(e) = crate::routes::events::emit_account_event(
                &db,
                &stream_user_id,
                "algorithm_changed",
                "algorithm",
                Some(&algo_id),
                serde_json::json!({
                    "document_id": &stream_document_id,
                    "algorithm_id": &algo_id,
                    "action": "error",
                }),
            )
            .await
            {
                tracing::error!("Failed to emit failed algorithm event: {:?}", e);
            }
            if let Err(e) = crate::routes::monitoring::log_system_event(
                &db,
                &stream_user_id,
                &stream_username,
                "algorithm_generation_error",
                "algorithm",
                Some(&algo_id),
                serde_json::json!({
                    "document_id": &stream_document_id,
                    "algorithm_id": &algo_id,
                    "mode": generation_mode,
                    "tokens": token_seq,
                    "error": error_message,
                }),
            )
            .await
            {
                tracing::error!("Failed to audit failed algorithm event: {:?}", e);
            }
            return;
        }

        if !collected.trim().is_empty() {
            let save_result = sqlx::query(
                "UPDATE algorithms SET content_markdown = ?, stream_seq = ?, status = 'completed' WHERE id = ?",
            )
            .bind(&collected)
            .bind(token_seq)
            .bind(&algo_id)
            .execute(&db)
            .await;

            if let Err(e) = save_result {
                tracing::error!("Failed to save algorithm to DB: {}", e);
                if let Err(e) = sqlx::query("UPDATE algorithms SET status = 'error' WHERE id = ?")
                    .bind(&algo_id)
                    .execute(&db)
                    .await
                {
                    tracing::error!("Failed to mark algorithm as error: {}", e);
                }
                if let Err(e) = crate::sse::insert_generation_event(
                    &db,
                    "algorithm",
                    &algo_id,
                    token_seq + 1,
                    "error",
                    "Не удалось сохранить алгоритм",
                )
                .await
                {
                    tracing::error!("Failed to save algorithm error event: {}", e);
                }
                let event = "event: save_error\ndata: \"Не удалось сохранить алгоритм\"\n\n";
                let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
            } else if let Err(e) = crate::sse::insert_generation_event(
                &db,
                "algorithm",
                &algo_id,
                token_seq + 1,
                "done",
                "",
            )
            .await
            {
                tracing::error!("Failed to save algorithm done event: {}", e);
            }
            if let Err(e) = crate::routes::events::emit_account_event(
                &db,
                &stream_user_id,
                "algorithm_changed",
                "algorithm",
                Some(&algo_id),
                serde_json::json!({
                    "document_id": &stream_document_id,
                    "algorithm_id": &algo_id,
                    "action": "completed",
                }),
            )
            .await
            {
                tracing::error!("Failed to emit completed algorithm event: {:?}", e);
            }
            if let Err(e) = crate::routes::monitoring::log_system_event(
                &db,
                &stream_user_id,
                &stream_username,
                "algorithm_generation_completed",
                "algorithm",
                Some(&algo_id),
                serde_json::json!({
                    "document_id": &stream_document_id,
                    "algorithm_id": &algo_id,
                    "mode": generation_mode,
                    "tokens": token_seq,
                }),
            )
            .await
            {
                tracing::error!("Failed to audit completed algorithm event: {:?}", e);
            }
        } else {
            if let Err(e) = sqlx::query(
                "UPDATE algorithms SET status = 'error' WHERE id = ? AND status = 'running'",
            )
            .bind(&algo_id)
            .execute(&db)
            .await
            {
                tracing::error!("Failed to mark algorithm as error: {}", e);
            }
            if let Err(e) = crate::sse::insert_generation_event(
                &db,
                "algorithm",
                &algo_id,
                token_seq + 1,
                "error",
                "AI не вернул текст алгоритма",
            )
            .await
            {
                tracing::error!("Failed to save empty algorithm error event: {}", e);
            }
            if let Err(e) = crate::routes::events::emit_account_event(
                &db,
                &stream_user_id,
                "algorithm_changed",
                "algorithm",
                Some(&algo_id),
                serde_json::json!({
                    "document_id": &stream_document_id,
                    "algorithm_id": &algo_id,
                    "action": "error",
                }),
            )
            .await
            {
                tracing::error!("Failed to emit failed algorithm event: {:?}", e);
            }
            if let Err(e) = crate::routes::monitoring::log_system_event(
                &db,
                &stream_user_id,
                &stream_username,
                "algorithm_generation_error",
                "algorithm",
                Some(&algo_id),
                serde_json::json!({
                    "document_id": &stream_document_id,
                    "algorithm_id": &algo_id,
                    "mode": generation_mode,
                    "tokens": token_seq,
                }),
            )
            .await
            {
                tracing::error!("Failed to audit failed algorithm event: {:?}", e);
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    };

    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .map_err(|e| AppError::Internal(format!("Failed to build SSE response: {e}")))
}

pub async fn get(
    State(state): State<AppState>,
    user: AuthUser,
    Path(algo_id): Path<String>,
) -> Result<Json<Algorithm>, AppError> {
    user.require_password_changed()?;
    let algo = sqlx::query_as::<_, Algorithm>(
        "SELECT a.id, a.document_id, a.content_markdown, a.status, a.generation_mode, \
                a.stream_seq, a.created_at \
         FROM algorithms a JOIN documents d ON d.id = a.document_id \
         WHERE a.id = ? AND d.user_id = ?",
    )
    .bind(&algo_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Algorithm not found".into()))?;
    Ok(Json(algo))
}

pub async fn get_by_document(
    State(state): State<AppState>,
    user: AuthUser,
    Path(doc_id): Path<String>,
) -> Result<Json<Option<Algorithm>>, AppError> {
    user.require_password_changed()?;
    let algo = sqlx::query_as::<_, Algorithm>(
        "SELECT a.id, a.document_id, a.content_markdown, a.status, a.generation_mode, \
                a.stream_seq, a.created_at \
         FROM algorithms a JOIN documents d ON d.id = a.document_id \
         WHERE a.document_id = ? AND d.user_id = ? ORDER BY a.created_at DESC LIMIT 1",
    )
    .bind(&doc_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?;
    Ok(Json(algo))
}

pub async fn stream(
    State(state): State<AppState>,
    user: AuthUser,
    Path(algo_id): Path<String>,
    Query(query): Query<StreamQuery>,
) -> Result<Response, AppError> {
    user.require_password_changed()?;
    sqlx::query(
        "SELECT a.id FROM algorithms a JOIN documents d ON d.id = a.document_id WHERE a.id = ? AND d.user_id = ?",
    )
        .bind(&algo_id)
        .bind(&user.id)
        .fetch_optional(&state.db)
        .await?
        .ok_or_else(|| AppError::NotFound("Algorithm not found".into()))?;

    crate::sse::resume_generation_events(
        state.db.clone(),
        "algorithm",
        algo_id,
        query.after.unwrap_or(0),
        "SELECT status FROM algorithms WHERE id = ?",
    )
    .map_err(|e| AppError::Internal(format!("Failed to build SSE response: {e}")))
}

pub async fn export_pdf(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<ExportPdfRequest>,
) -> Result<Response, AppError> {
    user.require_password_changed()?;
    let resp = state
        .python
        .client()
        .post(state.python.url("/algorithms/export-pdf"))
        .json(&serde_json::json!({ "markdown": body.markdown }))
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(AppError::AiServiceUnavailable("PDF export failed".into()));
    }

    let content_disposition = resp
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("attachment; filename=algorithm.pdf")
        .to_string();

    let pdf_bytes = resp
        .bytes()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to read PDF: {e}")))?;

    Response::builder()
        .header(header::CONTENT_TYPE, "application/pdf")
        .header(header::CONTENT_DISPOSITION, content_disposition)
        .body(Body::from(pdf_bytes))
        .map_err(|e| AppError::Internal(format!("Failed to build PDF response: {e}")))
}

async fn emit_algorithm_changed(
    state: &AppState,
    user_id: &str,
    document_id: &str,
    algorithm_id: Option<&str>,
    action: &str,
) -> Result<(), AppError> {
    crate::routes::events::emit_account_event(
        &state.db,
        user_id,
        "algorithm_changed",
        "algorithm",
        algorithm_id,
        serde_json::json!({
            "document_id": document_id,
            "algorithm_id": algorithm_id,
            "action": action,
        }),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::{AlgorithmGenerationMode, GenerateRequest};

    #[test]
    fn generate_request_defaults_to_physician_mode() {
        let request: GenerateRequest = serde_json::from_str(r#"{"document_id":"doc-1"}"#).unwrap();

        assert_eq!(request.document_id, "doc-1");
        assert!(matches!(request.mode, AlgorithmGenerationMode::Physician));
    }

    #[test]
    fn generate_request_accepts_source_mode() {
        let request: GenerateRequest =
            serde_json::from_str(r#"{"document_id":"doc-1","mode":"source"}"#).unwrap();

        assert!(matches!(request.mode, AlgorithmGenerationMode::Source));
    }

    #[test]
    fn generate_request_accepts_physician_mode() {
        let request: GenerateRequest =
            serde_json::from_str(r#"{"document_id":"doc-1","mode":"physician"}"#).unwrap();

        assert!(matches!(request.mode, AlgorithmGenerationMode::Physician));
    }

    #[test]
    fn generate_request_rejects_unknown_mode() {
        let result =
            serde_json::from_str::<GenerateRequest>(r#"{"document_id":"doc-1","mode":"unknown"}"#);

        assert!(result.is_err());
    }
}
