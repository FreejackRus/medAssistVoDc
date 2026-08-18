use axum::{
    Json,
    extract::{Multipart, Path, State},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{AppState, auth::AuthUser, error::AppError, models::DocumentSummary};

const CR_PDF_API: &str = "https://apicr.minzdrav.gov.ru/api.ashx";

#[derive(Deserialize)]
struct IngestResponse {
    #[allow(dead_code)]
    document_id: String,
    diagnosis_name: Option<String>,
    mkb_code: Option<String>,
    chunk_count: i64,
    full_text: String,
    #[allow(dead_code)]
    sections_json: Option<String>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct DocumentSource {
    content_hash: String,
    vector_document_id: String,
    file_path: String,
    diagnosis_name: Option<String>,
    mkb_code: Option<String>,
    chunk_count: i64,
}

#[derive(Debug, sqlx::FromRow)]
struct DocumentProcessingRow {
    id: String,
    filename: String,
    file_path: String,
    content_hash: Option<String>,
    status: String,
}

#[derive(Debug, sqlx::FromRow)]
struct DocumentDeleteRow {
    id: String,
    file_path: String,
    content_hash: Option<String>,
}

pub async fn upload(
    State(state): State<AppState>,
    user: AuthUser,
    mut multipart: Multipart,
) -> Result<Json<DocumentSummary>, AppError> {
    user.require_password_changed()?;
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut original_name = String::from("document.pdf");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?
    {
        if field.name() == Some("pdf") {
            original_name = field.file_name().unwrap_or("document.pdf").to_string();
            file_bytes = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read file: {e}")))?
                    .to_vec(),
            );
        }
    }

    let file_bytes =
        file_bytes.ok_or_else(|| AppError::BadRequest("No PDF file provided".into()))?;

    if !original_name.to_lowercase().ends_with(".pdf") {
        return Err(AppError::BadRequest("Only PDF files are accepted".into()));
    }

    Ok(Json(
        create_document_from_pdf(&state, &user, original_name, file_bytes).await?,
    ))
}

#[derive(Deserialize)]
pub struct ImportRecommendationRequest {
    pub code_version: String,
    pub title: String,
}

pub async fn import_recommendation(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<ImportRecommendationRequest>,
) -> Result<Json<DocumentSummary>, AppError> {
    user.require_password_changed()?;
    if body.code_version.is_empty() || body.code_version.len() > 50 {
        return Err(AppError::BadRequest("Invalid code_version".into()));
    }
    if !body
        .code_version
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
    {
        return Err(AppError::BadRequest("Invalid code_version format".into()));
    }

    let filename = format!("{}.pdf", body.title);

    // Download PDF from Minzdrav API
    let pdf_url = format!("{}?id={}&op=GetClinrecPdf", CR_PDF_API, body.code_version);
    let pdf_resp = state
        .python
        .client()
        .get(&pdf_url)
        .header("User-Agent", "Mozilla/5.0 (compatible; MedAssistant/2.0)")
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to download from Minzdrav: {e}")))?;

    if !pdf_resp.status().is_success() {
        return Err(AppError::BadRequest(format!(
            "Minzdrav returned status {}",
            pdf_resp.status()
        )));
    }

    // Enforce download size limit
    let max_download = state.config.upload_max_body_size_bytes();
    if let Some(len) = pdf_resp.content_length() {
        if len as usize > max_download {
            return Err(AppError::BadRequest(format!(
                "PDF too large: {} MB (max {} MB)",
                len / 1024 / 1024,
                state.config.upload_max_body_size_mb
            )));
        }
    }

    let file_bytes = pdf_resp
        .bytes()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to read PDF bytes: {e}")))?
        .to_vec();

    if file_bytes.len() > max_download {
        return Err(AppError::BadRequest(
            "Downloaded PDF exceeds size limit".into(),
        ));
    }

    if file_bytes.len() < 1000 || &file_bytes[..5] != b"%PDF-" {
        return Err(AppError::BadRequest(
            "Downloaded file is not a valid PDF".into(),
        ));
    }

    Ok(Json(
        create_document_from_pdf(&state, &user, filename, file_bytes).await?,
    ))
}

async fn create_document_from_pdf(
    state: &AppState,
    user: &AuthUser,
    filename: String,
    file_bytes: Vec<u8>,
) -> Result<DocumentSummary, AppError> {
    let content_hash = format!("{:x}", Sha256::digest(&file_bytes));
    ensure_not_uploaded_by_user(state, &user.id, &content_hash).await?;

    if let Some(source) = fetch_ready_source(state, &content_hash).await? {
        let doc_id = create_document_from_source(state, user, &filename, &source).await?;
        tracing::info!(
            "Document reused from source: doc_id={}, source_hash={}, vector_document_id={}",
            doc_id,
            source.content_hash,
            source.vector_document_id
        );
        return fetch_summary(state, &doc_id, &user.id).await;
    }

    let doc_id = Uuid::new_v4().to_string();
    let safe_name = format!("{}.pdf", Uuid::new_v4());
    let upload_dir = std::path::Path::new(&state.config.upload_dir);
    tokio::fs::create_dir_all(upload_dir)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to create upload dir: {e}")))?;
    let dest = upload_dir.join(&safe_name);
    tokio::fs::write(&dest, &file_bytes)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to write file: {e}")))?;
    let file_path = dest.to_string_lossy().to_string();

    let source_insert = sqlx::query(
        "INSERT OR IGNORE INTO document_sources (
            content_hash,
            vector_document_id,
            file_path,
            status
        ) VALUES (?, ?, ?, 'processing')",
    )
    .bind(&content_hash)
    .bind(&doc_id)
    .bind(&file_path)
    .execute(&state.db)
    .await?;

    if source_insert.rows_affected() == 0 {
        tokio::fs::remove_file(&file_path).await.ok();
        if let Some(source) = fetch_ready_source(state, &content_hash).await? {
            let doc_id = create_document_from_source(state, user, &filename, &source).await?;
            return fetch_summary(state, &doc_id, &user.id).await;
        }
        return Err(AppError::BadRequest(
            "Такой документ уже обрабатывается. Повторите загрузку позже.".into(),
        ));
    }

    sqlx::query(
        "INSERT INTO documents (
            id,
            user_id,
            filename,
            file_path,
            content_hash,
            status
        ) VALUES (?, ?, ?, ?, ?, 'processing')",
    )
    .bind(&doc_id)
    .bind(&user.id)
    .bind(&filename)
    .bind(&file_path)
    .bind(&content_hash)
    .execute(&state.db)
    .await?;
    emit_documents_changed(state, &user.id, &doc_id, "created").await?;

    ingest_document_source(
        state,
        user,
        &doc_id,
        &content_hash,
        &doc_id,
        &filename,
        file_bytes,
        false,
    )
    .await?;

    fetch_summary(state, &doc_id, &user.id).await
}

async fn ensure_not_uploaded_by_user(
    state: &AppState,
    user_id: &str,
    content_hash: &str,
) -> Result<(), AppError> {
    let existing = sqlx::query_scalar::<_, String>(
        "SELECT filename FROM documents WHERE user_id = ? AND content_hash = ? AND status != 'error' LIMIT 1",
    )
    .bind(user_id)
    .bind(content_hash)
    .fetch_optional(&state.db)
    .await?;

    if let Some(existing_name) = existing {
        return Err(AppError::BadRequest(format!(
            "Этот документ уже загружен как '{}'",
            existing_name
        )));
    }

    Ok(())
}

async fn fetch_ready_source(
    state: &AppState,
    content_hash: &str,
) -> Result<Option<DocumentSource>, AppError> {
    sqlx::query_as::<_, DocumentSource>(
        "SELECT
            content_hash,
            vector_document_id,
            file_path,
            diagnosis_name,
            mkb_code,
            chunk_count
         FROM document_sources
         WHERE content_hash = ?
           AND status = 'ready'
           AND full_text IS NOT NULL
         LIMIT 1",
    )
    .bind(content_hash)
    .fetch_optional(&state.db)
    .await
    .map_err(Into::into)
}

async fn fetch_source(
    state: &AppState,
    content_hash: &str,
) -> Result<Option<DocumentSource>, AppError> {
    sqlx::query_as::<_, DocumentSource>(
        "SELECT
            content_hash,
            vector_document_id,
            file_path,
            diagnosis_name,
            mkb_code,
            chunk_count
         FROM document_sources
         WHERE content_hash = ?
         LIMIT 1",
    )
    .bind(content_hash)
    .fetch_optional(&state.db)
    .await
    .map_err(Into::into)
}

async fn create_document_from_source(
    state: &AppState,
    user: &AuthUser,
    filename: &str,
    source: &DocumentSource,
) -> Result<String, AppError> {
    let doc_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO documents (
            id,
            user_id,
            filename,
            file_path,
            content_hash,
            diagnosis_name,
            mkb_code,
            full_text,
            chunk_count,
            status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, 'ready')",
    )
    .bind(&doc_id)
    .bind(&user.id)
    .bind(filename)
    .bind(&source.file_path)
    .bind(&source.content_hash)
    .bind(&source.diagnosis_name)
    .bind(&source.mkb_code)
    .bind(source.chunk_count)
    .execute(&state.db)
    .await?;
    emit_documents_changed(state, &user.id, &doc_id, "created").await?;
    Ok(doc_id)
}

async fn ingest_document_source(
    state: &AppState,
    user: &AuthUser,
    doc_id: &str,
    content_hash: &str,
    vector_document_id: &str,
    filename: &str,
    file_bytes: Vec<u8>,
    update_all_documents: bool,
) -> Result<(), AppError> {
    let form = reqwest::multipart::Form::new()
        .part(
            "pdf",
            reqwest::multipart::Part::bytes(file_bytes)
                .file_name(filename.to_string())
                .mime_str("application/pdf")
                .unwrap(),
        )
        .text("document_id", vector_document_id.to_string());

    let python_resp = state
        .python
        .client()
        .post(state.python.url("/documents/ingest"))
        .multipart(form)
        .send()
        .await;

    match python_resp {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(result) = resp.json::<IngestResponse>().await {
                sqlx::query(
                    "UPDATE document_sources
                     SET diagnosis_name = ?,
                         mkb_code = ?,
                         full_text = ?,
                         chunk_count = ?,
                         status = 'ready',
                         updated_at = datetime('now')
                     WHERE content_hash = ?",
                )
                .bind(&result.diagnosis_name)
                .bind(&result.mkb_code)
                .bind(&result.full_text)
                .bind(result.chunk_count)
                .bind(content_hash)
                .execute(&state.db)
                .await?;

                if update_all_documents {
                    sqlx::query(
                        "UPDATE documents
                         SET diagnosis_name = ?,
                             mkb_code = ?,
                             full_text = NULL,
                             chunk_count = ?,
                             status = 'ready',
                             updated_at = datetime('now')
                         WHERE content_hash = ?",
                    )
                    .bind(&result.diagnosis_name)
                    .bind(&result.mkb_code)
                    .bind(result.chunk_count)
                    .bind(content_hash)
                    .execute(&state.db)
                    .await?;
                } else {
                    sqlx::query(
                        "UPDATE documents
                         SET diagnosis_name = ?,
                             mkb_code = ?,
                             full_text = NULL,
                             chunk_count = ?,
                             status = 'ready',
                             updated_at = datetime('now')
                         WHERE id = ?",
                    )
                    .bind(&result.diagnosis_name)
                    .bind(&result.mkb_code)
                    .bind(result.chunk_count)
                    .bind(doc_id)
                    .execute(&state.db)
                    .await?;
                }

                tracing::info!(
                    "Document source ingested: doc_id={}, vector_document_id={}, diagnosis={:?}, chunks={}",
                    doc_id,
                    vector_document_id,
                    result.diagnosis_name,
                    result.chunk_count
                );
                emit_documents_changed(state, &user.id, doc_id, "updated").await?;
            } else {
                mark_ingestion_failed(state, user, doc_id, content_hash, update_all_documents)
                    .await?;
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::error!("Python ingestion failed: {} - {}", status, body);
            mark_ingestion_failed(state, user, doc_id, content_hash, update_all_documents).await?;
        }
        Err(e) => {
            tracing::error!("Python ingestion error: {}", e);
            mark_ingestion_failed(state, user, doc_id, content_hash, update_all_documents).await?;
        }
    }

    Ok(())
}

async fn mark_ingestion_failed(
    state: &AppState,
    user: &AuthUser,
    doc_id: &str,
    content_hash: &str,
    update_all_documents: bool,
) -> Result<(), AppError> {
    sqlx::query(
        "UPDATE document_sources
         SET status = 'error', updated_at = datetime('now')
         WHERE content_hash = ?",
    )
    .bind(content_hash)
    .execute(&state.db)
    .await?;
    if update_all_documents {
        sqlx::query(
            "UPDATE documents SET status = 'error', updated_at = datetime('now') WHERE content_hash = ?",
        )
        .bind(content_hash)
        .execute(&state.db)
        .await?;
    } else {
        sqlx::query(
            "UPDATE documents SET status = 'error', updated_at = datetime('now') WHERE id = ?",
        )
        .bind(doc_id)
        .execute(&state.db)
        .await?;
    }
    emit_documents_changed(state, &user.id, doc_id, "updated").await?;
    Ok(())
}

async fn fetch_summary(
    state: &AppState,
    doc_id: &str,
    user_id: &str,
) -> Result<DocumentSummary, AppError> {
    sqlx::query_as::<_, DocumentSummary>(
        "SELECT id, user_id, filename, diagnosis_name, mkb_code, status, chunk_count, created_at, updated_at FROM documents WHERE id = ? AND user_id = ?",
    )
    .bind(doc_id)
    .bind(user_id)
    .fetch_one(&state.db)
    .await
    .map_err(Into::into)
}

pub async fn list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<Vec<DocumentSummary>>, AppError> {
    user.require_password_changed()?;
    let docs = sqlx::query_as::<_, DocumentSummary>(
        "SELECT id, user_id, filename, diagnosis_name, mkb_code, status, chunk_count, created_at, updated_at FROM documents WHERE user_id = ? ORDER BY created_at DESC",
    )
    .bind(&user.id)
    .fetch_all(&state.db)
    .await?;
    Ok(Json(docs))
}

pub async fn get(
    State(state): State<AppState>,
    user: AuthUser,
    Path(doc_id): Path<String>,
) -> Result<Json<DocumentSummary>, AppError> {
    user.require_password_changed()?;
    let doc = sqlx::query_as::<_, DocumentSummary>(
        "SELECT id, user_id, filename, diagnosis_name, mkb_code, status, chunk_count, created_at, updated_at FROM documents WHERE id = ? AND user_id = ?",
    )
    .bind(&doc_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Document not found".into()))?;
    Ok(Json(doc))
}

pub async fn retry(
    State(state): State<AppState>,
    user: AuthUser,
    Path(doc_id): Path<String>,
) -> Result<Json<DocumentSummary>, AppError> {
    user.require_password_changed()?;
    let doc = sqlx::query_as::<_, DocumentProcessingRow>(
        "SELECT id, filename, file_path, content_hash, status FROM documents WHERE id = ? AND user_id = ?",
    )
        .bind(&doc_id)
        .bind(&user.id)
        .fetch_optional(&state.db)
        .await?
        .ok_or_else(|| AppError::NotFound("Document not found".into()))?;

    if doc.status == "processing" {
        return Err(AppError::BadRequest(
            "Document is already being processed".into(),
        ));
    }

    let file_bytes = tokio::fs::read(&doc.file_path)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to read file: {e}")))?;
    let content_hash = doc
        .content_hash
        .clone()
        .unwrap_or_else(|| format!("{:x}", Sha256::digest(&file_bytes)));

    if doc.content_hash.is_none() {
        sqlx::query("UPDATE documents SET content_hash = ? WHERE id = ?")
            .bind(&content_hash)
            .bind(&doc_id)
            .execute(&state.db)
            .await?;
    }

    if doc.status != "ready" {
        if let Some(source) = fetch_ready_source(&state, &content_hash).await? {
            sqlx::query(
                "UPDATE documents
                 SET file_path = ?,
                     diagnosis_name = ?,
                     mkb_code = ?,
                     full_text = NULL,
                     chunk_count = ?,
                     status = 'ready',
                     updated_at = datetime('now')
                 WHERE id = ?",
            )
            .bind(&source.file_path)
            .bind(&source.diagnosis_name)
            .bind(&source.mkb_code)
            .bind(source.chunk_count)
            .bind(&doc_id)
            .execute(&state.db)
            .await?;
            emit_documents_changed(&state, &user.id, &doc_id, "updated").await?;
            return Ok(Json(fetch_summary(&state, &doc_id, &user.id).await?));
        }
    }

    let source = fetch_source(&state, &content_hash).await?;
    let vector_document_id = if let Some(source) = source {
        sqlx::query(
            "UPDATE document_sources
             SET file_path = ?,
                 status = 'processing',
                 updated_at = datetime('now')
             WHERE content_hash = ?",
        )
        .bind(&doc.file_path)
        .bind(&content_hash)
        .execute(&state.db)
        .await?;
        source.vector_document_id
    } else {
        sqlx::query(
            "INSERT INTO document_sources (
                content_hash,
                vector_document_id,
                file_path,
                status
            ) VALUES (?, ?, ?, 'processing')",
        )
        .bind(&content_hash)
        .bind(&doc_id)
        .bind(&doc.file_path)
        .execute(&state.db)
        .await?;
        doc_id.clone()
    };

    sqlx::query(
        "UPDATE documents SET status = 'processing', updated_at = datetime('now') WHERE content_hash = ?",
    )
    .bind(&content_hash)
    .execute(&state.db)
    .await?;
    emit_documents_changed(&state, &user.id, &doc_id, "updated").await?;

    let mut txn = state.db.begin().await?;
    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'algorithm' AND target_id IN (
            SELECT a.id FROM algorithms a
            JOIN documents d ON d.id = a.document_id
            WHERE d.content_hash = ?
        )",
    )
    .bind(&content_hash)
    .execute(&mut *txn)
    .await?;
    sqlx::query(
        "DELETE FROM algorithms WHERE document_id IN (
            SELECT id FROM documents WHERE content_hash = ?
        )",
    )
    .bind(&content_hash)
    .execute(&mut *txn)
    .await?;
    txn.commit().await?;

    ingest_document_source(
        &state,
        &user,
        &doc.id,
        &content_hash,
        &vector_document_id,
        &doc.filename,
        file_bytes,
        true,
    )
    .await?;

    Ok(Json(fetch_summary(&state, &doc_id, &user.id).await?))
}

pub async fn delete(
    State(state): State<AppState>,
    user: AuthUser,
    Path(doc_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    user.require_password_changed()?;
    let doc = sqlx::query_as::<_, DocumentDeleteRow>(
        "SELECT id, file_path, content_hash FROM documents WHERE id = ? AND user_id = ?",
    )
    .bind(&doc_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Document not found".into()))?;

    let mut txn = state.db.begin().await?;
    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'algorithm' AND target_id IN (SELECT id FROM algorithms WHERE document_id = ?)",
    )
    .bind(&doc.id)
    .execute(&mut *txn)
    .await?;
    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'chat' AND target_id IN (
            SELECT m.id FROM chat_messages m
            JOIN chat_sessions s ON s.id = m.session_id
            WHERE s.document_id = ?
        )",
    )
    .bind(&doc.id)
    .execute(&mut *txn)
    .await?;
    sqlx::query("DELETE FROM documents WHERE id = ?")
        .bind(&doc.id)
        .execute(&mut *txn)
        .await?;
    txn.commit().await?;
    emit_documents_changed(&state, &user.id, &doc_id, "deleted").await?;

    if let Some(content_hash) = doc.content_hash {
        if !cleanup_unused_document_source(&state, &content_hash).await? {
            tokio::fs::remove_file(&doc.file_path).await.ok();
            state
                .python
                .client()
                .delete(state.python.url(&format!("/documents/{}", doc.id)))
                .send()
                .await
                .ok();
        }
    } else {
        tokio::fs::remove_file(&doc.file_path).await.ok();
        state
            .python
            .client()
            .delete(state.python.url(&format!("/documents/{}", doc.id)))
            .send()
            .await
            .ok();
    }

    Ok(Json(serde_json::json!({ "ok": true })))
}

pub async fn cleanup_unused_document_source(
    state: &AppState,
    content_hash: &str,
) -> Result<bool, AppError> {
    let source = fetch_source(state, content_hash).await?;
    let Some(source) = source else {
        return Ok(false);
    };

    let references =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM documents WHERE content_hash = ?")
            .bind(content_hash)
            .fetch_one(&state.db)
            .await
            .unwrap_or(0);

    if references > 0 {
        return Ok(true);
    }

    state
        .python
        .client()
        .delete(
            state
                .python
                .url(&format!("/documents/{}", source.vector_document_id)),
        )
        .send()
        .await
        .ok();
    tokio::fs::remove_file(&source.file_path).await.ok();
    sqlx::query("DELETE FROM document_sources WHERE content_hash = ?")
        .bind(content_hash)
        .execute(&state.db)
        .await?;

    Ok(true)
}

async fn emit_documents_changed(
    state: &AppState,
    user_id: &str,
    doc_id: &str,
    action: &str,
) -> Result<(), AppError> {
    crate::routes::events::emit_account_event(
        &state.db,
        user_id,
        "documents_changed",
        "document",
        Some(doc_id),
        serde_json::json!({ "document_id": doc_id, "action": action }),
    )
    .await
}
