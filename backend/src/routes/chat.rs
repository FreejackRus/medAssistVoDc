use axum::{
    Json,
    body::Body,
    extract::{Multipart, Path, Query, State},
    http::header,
    response::Response,
};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use uuid::Uuid;

use crate::{
    AppState,
    auth::AuthUser,
    error::AppError,
    models::{ChatMessage, ChatSession},
};

const MAX_CHAT_ATTACHMENTS: usize = 5;
const ATTACHMENT_PROMPT_TEXT_LIMIT: usize = 50_000;

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub document_id: String,
}

#[derive(Deserialize)]
pub struct ListSessionsQuery {
    pub document_id: Option<String>,
}

#[derive(Deserialize)]
pub struct StreamQuery {
    pub after: Option<i64>,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct ChatAttachmentResponse {
    pub id: String,
    pub message_id: String,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: i64,
    pub status: String,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub status: String,
    pub stream_seq: i64,
    pub created_at: NaiveDateTime,
    pub attachments: Vec<ChatAttachmentResponse>,
}

#[derive(Debug, sqlx::FromRow)]
struct ChatAttachmentSource {
    vector_document_id: String,
    full_text: String,
    chunk_count: i64,
}

#[derive(Debug)]
struct PendingAttachment {
    filename: String,
    file_path: String,
    content_hash: String,
    vector_document_id: String,
    size_bytes: i64,
    chunk_count: i64,
    full_text: String,
}

#[derive(Debug)]
struct PromptAttachmentContext {
    filename: String,
    vector_document_id: String,
    full_text: String,
}

#[derive(Debug, Deserialize)]
struct ChatAttachmentIngestResponse {
    #[allow(dead_code)]
    document_id: String,
    #[allow(dead_code)]
    filename: String,
    full_text: String,
    chunk_count: i64,
}

pub async fn create_session(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<CreateSessionRequest>,
) -> Result<Json<ChatSession>, AppError> {
    user.require_password_changed()?;
    let title = sqlx::query_scalar::<_, String>(
        "SELECT COALESCE(diagnosis_name, filename) FROM documents WHERE id = ? AND user_id = ? AND status = 'ready'",
    )
    .bind(&body.document_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::BadRequest("Document not found or not ready".into()))?;

    let id = Uuid::new_v4().to_string();
    sqlx::query("INSERT INTO chat_sessions (id, user_id, document_id, title) VALUES (?, ?, ?, ?)")
        .bind(&id)
        .bind(&user.id)
        .bind(&body.document_id)
        .bind(&title)
        .execute(&state.db)
        .await?;

    let session = sqlx::query_as::<_, ChatSession>(
        "SELECT id, user_id, document_id, title, created_at FROM chat_sessions WHERE id = ? AND user_id = ?",
    )
    .bind(&id)
    .bind(&user.id)
    .fetch_one(&state.db)
    .await?;
    emit_sessions_changed(
        &state,
        &user.id,
        Some(&body.document_id),
        Some(&id),
        "created",
    )
    .await?;

    Ok(Json(session))
}

pub async fn list_sessions(
    State(state): State<AppState>,
    user: AuthUser,
    Query(query): Query<ListSessionsQuery>,
) -> Result<Json<Vec<ChatSession>>, AppError> {
    user.require_password_changed()?;
    let sessions = if let Some(doc_id) = query.document_id {
        sqlx::query_as::<_, ChatSession>(
            "SELECT id, user_id, document_id, title, created_at FROM chat_sessions WHERE user_id = ? AND document_id = ? ORDER BY created_at DESC",
        )
        .bind(&user.id)
        .bind(&doc_id)
        .fetch_all(&state.db)
        .await?
    } else {
        sqlx::query_as::<_, ChatSession>(
            "SELECT id, user_id, document_id, title, created_at FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC",
        )
        .bind(&user.id)
        .fetch_all(&state.db)
        .await?
    };

    Ok(Json(sessions))
}

pub async fn get_messages(
    State(state): State<AppState>,
    user: AuthUser,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<ChatMessageResponse>>, AppError> {
    user.require_password_changed()?;
    sqlx::query("SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?")
        .bind(&session_id)
        .bind(&user.id)
        .fetch_optional(&state.db)
        .await?
        .ok_or_else(|| AppError::NotFound("Session not found".into()))?;

    let rows = sqlx::query_as::<_, ChatMessage>(
        "SELECT id, session_id, role, content, status, stream_seq, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
    )
    .bind(&session_id)
    .fetch_all(&state.db)
    .await?;

    let mut messages = Vec::with_capacity(rows.len());
    for row in rows {
        let attachments = fetch_message_attachments(&state, &row.id).await?;
        messages.push(ChatMessageResponse {
            id: row.id,
            session_id: row.session_id,
            role: row.role,
            content: row.content,
            status: row.status,
            stream_seq: row.stream_seq,
            created_at: row.created_at,
            attachments,
        });
    }

    Ok(Json(messages))
}

pub async fn send_message(
    State(state): State<AppState>,
    user: AuthUser,
    Path(session_id): Path<String>,
    multipart: Multipart,
) -> Result<Response, AppError> {
    user.require_password_changed()?;
    let (message, attachments) = parse_chat_multipart(&state, multipart).await?;
    if message.is_empty() {
        return Err(AppError::BadRequest("Empty message".into()));
    }
    if message.len() > 10_000 {
        return Err(AppError::BadRequest(
            "Message too long (max 10 000 chars)".into(),
        ));
    }
    if attachments.len() > MAX_CHAT_ATTACHMENTS {
        return Err(AppError::BadRequest(format!(
            "Слишком много вложений: максимум {} PDF",
            MAX_CHAT_ATTACHMENTS
        )));
    }

    let mut prepared_attachments = Vec::with_capacity(attachments.len());
    for (filename, bytes) in attachments {
        prepared_attachments.push(prepare_chat_attachment(&state, filename, bytes).await?);
    }

    // Get session's document info from Rust DB
    let (document_id, diagnosis_name, vector_document_id) =
        sqlx::query_as::<_, (String, String, String)>(
            "SELECT
                s.document_id,
                COALESCE(d.diagnosis_name, src.diagnosis_name, d.filename),
                COALESCE(src.vector_document_id, d.id)
             FROM chat_sessions s
             JOIN documents d ON s.document_id = d.id
             LEFT JOIN document_sources src ON src.content_hash = d.content_hash
             WHERE s.id = ? AND s.user_id = ? AND d.user_id = ?",
        )
        .bind(&session_id)
        .bind(&user.id)
        .bind(&user.id)
        .fetch_optional(&state.db)
        .await?
        .ok_or_else(|| AppError::NotFound("Session not found".into()))?;

    // Check if this is the first message — update session title
    let msg_count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM chat_messages WHERE session_id = ?")
            .bind(&session_id)
            .fetch_one(&state.db)
            .await
            .unwrap_or(0);

    if msg_count == 0 {
        let title: String = if message.len() > 50 {
            format!(
                "{}...",
                &message[..message
                    .char_indices()
                    .take(50)
                    .last()
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(50)]
            )
        } else {
            message.clone()
        };
        sqlx::query("UPDATE chat_sessions SET title = ? WHERE id = ?")
            .bind(&title)
            .bind(&session_id)
            .execute(&state.db)
            .await
            .ok();
        emit_sessions_changed(
            &state,
            &user.id,
            Some(&document_id),
            Some(&session_id),
            "updated",
        )
        .await?;
    }

    // Save user message to Rust DB
    let user_msg_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO chat_messages (id, session_id, role, content) VALUES (?, ?, 'user', ?)",
    )
    .bind(&user_msg_id)
    .bind(&session_id)
    .bind(&message)
    .execute(&state.db)
    .await?;
    insert_message_attachments(&state, &user_msg_id, &prepared_attachments).await?;
    let attachment_contexts =
        fetch_session_attachment_contexts(&state, &session_id, &user_msg_id, &prepared_attachments)
            .await?;
    emit_messages_changed(&state, &user.id, &session_id, Some(&user_msg_id), "created").await?;

    // Fetch conversation history from Rust DB
    let history_rows = sqlx::query_as::<_, (String, String)>(
        "SELECT role, content FROM chat_messages WHERE session_id = ? AND status = 'completed' ORDER BY created_at ASC",
    )
    .bind(&session_id)
    .fetch_all(&state.db)
    .await?;

    let history: Vec<serde_json::Value> = history_rows
        .iter()
        .filter(|(role, _)| role == "user" || role == "assistant")
        .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
        .collect();

    // Drop the last user message from history (it's the current question),
    // and limit to the most recent 20 messages to avoid oversized prompts.
    let history_without_current: Vec<_> = if history.len() > 1 {
        let h = &history[..history.len() - 1];
        let start = h.len().saturating_sub(20);
        h[start..].to_vec()
    } else {
        vec![]
    };

    // Call Python stateless RAG endpoint (SSE stream — no timeout)
    let resp = state
        .python
        .stream_client()
        .post(state.python.url("/chat/completions"))
        .json(&serde_json::json!({
            "question": message,
            "document_id": vector_document_id,
            "diagnosis_name": diagnosis_name,
            "history": &history_without_current,
            "attachments": attachment_contexts.iter().map(|attachment| {
                serde_json::json!({
                    "filename": attachment.filename,
                    "document_id": attachment.vector_document_id,
                    "text": prompt_attachment_text(&attachment.full_text, &message),
                })
            }).collect::<Vec<_>>(),
        }))
        .send()
        .await?;

    if !resp.status().is_success() {
        crate::routes::monitoring::log_audit_event(
            &state.db,
            Some(&user),
            Some(&user.id),
            Some(&user.username),
            Some(&user.id),
            "chat_generation_error",
            "chat_session",
            Some(&session_id),
            serde_json::json!({
                "session_id": &session_id,
                "document_id": &document_id,
                "status": resp.status().as_u16(),
            }),
        )
        .await
        .ok();
        return Err(AppError::AiServiceUnavailable(
            "AI service chat error".into(),
        ));
    }

    let assistant_msg_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO chat_messages (id, session_id, role, content, status) VALUES (?, ?, 'assistant', '', 'running')",
    )
    .bind(&assistant_msg_id)
    .bind(&session_id)
    .execute(&state.db)
    .await?;
    emit_messages_changed(
        &state,
        &user.id,
        &session_id,
        Some(&assistant_msg_id),
        "created",
    )
    .await?;
    crate::routes::monitoring::log_audit_event(
        &state.db,
        Some(&user),
        Some(&user.id),
        Some(&user.username),
        Some(&user.id),
        "chat_generation_started",
        "chat_message",
        Some(&assistant_msg_id),
        serde_json::json!({
            "session_id": &session_id,
            "document_id": &document_id,
            "message_id": &assistant_msg_id,
        }),
    )
    .await?;

    let db = state.db.clone();
    let stream_user_id = user.id.clone();
    let stream_username = user.username.clone();
    let stream_session_id = session_id.clone();
    let stream_document_id = document_id.clone();

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let update_db = db.clone();
        let update_msg_id = assistant_msg_id.clone();
        let event_db = db.clone();
        let event_msg_id = assistant_msg_id.clone();
        let relay = crate::sse::relay_and_collect_with_events(
            resp,
            &tx,
            move |seq, token| {
                let event_db = event_db.clone();
                let event_msg_id = event_msg_id.clone();
                async move {
                    if let Err(e) = crate::sse::insert_generation_event(
                        &event_db,
                        "chat",
                        &event_msg_id,
                        seq,
                        "token",
                        &token,
                    )
                    .await
                    {
                        tracing::error!("Failed to save chat stream event: {}", e);
                    }
                }
            },
            move |partial, seq| {
            let update_db = update_db.clone();
            let update_msg_id = update_msg_id.clone();
            async move {
                if let Err(e) = sqlx::query(
                    "UPDATE chat_messages SET content = ?, stream_seq = ? WHERE id = ? AND status = 'running'",
                )
                .bind(partial)
                .bind(seq)
                .bind(update_msg_id)
                .execute(&update_db)
                .await
                {
                    tracing::error!("Failed to update assistant message progress: {}", e);
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
                "UPDATE chat_messages SET status = 'error' WHERE id = ? AND status = 'running'",
            )
            .bind(&assistant_msg_id)
            .execute(&db)
            .await
            {
                tracing::error!("Failed to mark assistant message as error: {}", e);
            }
            if let Err(e) = crate::sse::insert_generation_event(
                &db,
                "chat",
                &assistant_msg_id,
                token_seq + 1,
                "error",
                &error_message,
            )
            .await
            {
                tracing::error!("Failed to save chat upstream error event: {}", e);
            }
            if let Err(e) = crate::routes::events::emit_account_event(
                &db,
                &stream_user_id,
                "messages_changed",
                "chat_message",
                Some(&assistant_msg_id),
                serde_json::json!({
                    "session_id": &stream_session_id,
                    "document_id": &stream_document_id,
                    "message_id": &assistant_msg_id,
                    "action": "error",
                }),
            )
            .await
            {
                tracing::error!("Failed to emit failed chat event: {:?}", e);
            }
            if let Err(e) = crate::routes::monitoring::log_system_event(
                &db,
                &stream_user_id,
                &stream_username,
                "chat_generation_error",
                "chat_message",
                Some(&assistant_msg_id),
                serde_json::json!({
                    "session_id": &stream_session_id,
                    "document_id": &stream_document_id,
                    "message_id": &assistant_msg_id,
                    "tokens": token_seq,
                    "error": error_message,
                }),
            )
            .await
            {
                tracing::error!("Failed to audit failed chat event: {:?}", e);
            }
            return;
        }

        if !collected.trim().is_empty() {
            if let Err(e) = sqlx::query(
                "UPDATE chat_messages SET content = ?, stream_seq = ?, status = 'completed' WHERE id = ?",
            )
            .bind(&collected)
            .bind(token_seq)
            .bind(&assistant_msg_id)
            .execute(&db)
            .await
            {
                tracing::error!("Failed to save assistant message to DB: {}", e);
                if let Err(e) =
                    sqlx::query("UPDATE chat_messages SET status = 'error' WHERE id = ?")
                        .bind(&assistant_msg_id)
                        .execute(&db)
                        .await
                {
                    tracing::error!("Failed to mark assistant message as error: {}", e);
                }
                if let Err(e) = crate::sse::insert_generation_event(
                    &db,
                    "chat",
                    &assistant_msg_id,
                    token_seq + 1,
                    "error",
                    "Не удалось сохранить ответ",
                )
                .await
                {
                    tracing::error!("Failed to save chat error event: {}", e);
                }
                let event = "event: save_error\ndata: \"Не удалось сохранить ответ\"\n\n";
                let _ = tx.send(Ok(axum::body::Bytes::from(event))).await;
            } else if let Err(e) = crate::sse::insert_generation_event(
                &db,
                "chat",
                &assistant_msg_id,
                token_seq + 1,
                "done",
                "",
            )
            .await
            {
                tracing::error!("Failed to save chat done event: {}", e);
            }
            if let Err(e) = crate::routes::events::emit_account_event(
                &db,
                &stream_user_id,
                "messages_changed",
                "chat_message",
                Some(&assistant_msg_id),
                serde_json::json!({
                    "session_id": &stream_session_id,
                    "document_id": &stream_document_id,
                    "message_id": &assistant_msg_id,
                    "action": "completed",
                }),
            )
            .await
            {
                tracing::error!("Failed to emit completed chat event: {:?}", e);
            }
            if let Err(e) = crate::routes::monitoring::log_system_event(
                &db,
                &stream_user_id,
                &stream_username,
                "chat_generation_completed",
                "chat_message",
                Some(&assistant_msg_id),
                serde_json::json!({
                    "session_id": &stream_session_id,
                    "document_id": &stream_document_id,
                    "message_id": &assistant_msg_id,
                    "tokens": token_seq,
                }),
            )
            .await
            {
                tracing::error!("Failed to audit completed chat event: {:?}", e);
            }
        } else {
            if let Err(e) = sqlx::query(
                "UPDATE chat_messages SET status = 'error' WHERE id = ? AND status = 'running'",
            )
            .bind(&assistant_msg_id)
            .execute(&db)
            .await
            {
                tracing::error!("Failed to mark assistant message as error: {}", e);
            }
            if let Err(e) = crate::sse::insert_generation_event(
                &db,
                "chat",
                &assistant_msg_id,
                token_seq + 1,
                "error",
                "AI не вернул текст ответа",
            )
            .await
            {
                tracing::error!("Failed to save empty chat error event: {}", e);
            }
            if let Err(e) = crate::routes::events::emit_account_event(
                &db,
                &stream_user_id,
                "messages_changed",
                "chat_message",
                Some(&assistant_msg_id),
                serde_json::json!({
                    "session_id": &stream_session_id,
                    "document_id": &stream_document_id,
                    "message_id": &assistant_msg_id,
                    "action": "error",
                }),
            )
            .await
            {
                tracing::error!("Failed to emit failed chat event: {:?}", e);
            }
            if let Err(e) = crate::routes::monitoring::log_system_event(
                &db,
                &stream_user_id,
                &stream_username,
                "chat_generation_error",
                "chat_message",
                Some(&assistant_msg_id),
                serde_json::json!({
                    "session_id": &stream_session_id,
                    "document_id": &stream_document_id,
                    "message_id": &assistant_msg_id,
                    "tokens": token_seq,
                }),
            )
            .await
            {
                tracing::error!("Failed to audit failed chat event: {:?}", e);
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

pub async fn stream_message(
    State(state): State<AppState>,
    user: AuthUser,
    Path(message_id): Path<String>,
    Query(query): Query<StreamQuery>,
) -> Result<Response, AppError> {
    user.require_password_changed()?;
    sqlx::query(
        "SELECT m.id FROM chat_messages m \
         JOIN chat_sessions s ON s.id = m.session_id \
         WHERE m.id = ? AND m.role = 'assistant' AND s.user_id = ?",
    )
    .bind(&message_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Message not found".into()))?;

    crate::sse::resume_generation_events(
        state.db.clone(),
        "chat",
        message_id,
        query.after.unwrap_or(0),
        "SELECT status FROM chat_messages WHERE id = ?",
    )
    .map_err(|e| AppError::Internal(format!("Failed to build SSE response: {e}")))
}

pub async fn delete_session(
    State(state): State<AppState>,
    user: AuthUser,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    user.require_password_changed()?;
    let document_id = sqlx::query_scalar::<_, String>(
        "SELECT document_id FROM chat_sessions WHERE id = ? AND user_id = ?",
    )
    .bind(&session_id)
    .bind(&user.id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Session not found".into()))?;

    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'chat' AND target_id IN (SELECT id FROM chat_messages WHERE session_id = ?)",
    )
    .bind(&session_id)
    .execute(&state.db)
    .await?;

    let deleted = sqlx::query("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?")
        .bind(&session_id)
        .bind(&user.id)
        .execute(&state.db)
        .await?;

    if deleted.rows_affected() == 0 {
        return Err(AppError::NotFound("Session not found".into()));
    }
    emit_sessions_changed(
        &state,
        &user.id,
        Some(&document_id),
        Some(&session_id),
        "deleted",
    )
    .await?;

    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn parse_chat_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(String, Vec<(String, Vec<u8>)>), AppError> {
    let mut message = String::new();
    let mut attachments = Vec::new();
    let max_bytes = state.config.upload_max_body_size_bytes();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?
    {
        let field_name = field.name().unwrap_or("").to_string();
        match field_name.as_str() {
            "message" => {
                message = field
                    .text()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read message: {e}")))?
                    .trim()
                    .to_string();
            }
            "attachment" | "attachments" => {
                let filename = field.file_name().unwrap_or("attachment.pdf").to_string();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read attachment: {e}")))?
                    .to_vec();
                if bytes.len() > max_bytes {
                    return Err(AppError::BadRequest(format!(
                        "Файл '{}' слишком большой: максимум {} MB",
                        filename, state.config.upload_max_body_size_mb
                    )));
                }
                attachments.push((filename, bytes));
            }
            _ => {}
        }
    }

    Ok((message, attachments))
}

async fn prepare_chat_attachment(
    state: &AppState,
    filename: String,
    file_bytes: Vec<u8>,
) -> Result<PendingAttachment, AppError> {
    if !filename.to_lowercase().ends_with(".pdf") {
        return Err(AppError::BadRequest(format!(
            "Файл '{}' отклонён: принимаются только PDF",
            filename
        )));
    }
    if file_bytes.len() < 5 || &file_bytes[..5] != b"%PDF-" {
        return Err(AppError::BadRequest(format!(
            "Файл '{}' не похож на корректный PDF",
            filename
        )));
    }

    let content_hash = format!("{:x}", Sha256::digest(&file_bytes));
    let size_bytes = i64::try_from(file_bytes.len()).unwrap_or(i64::MAX);
    let upload_dir = std::path::Path::new(&state.config.upload_dir).join("chat-attachments");
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to create attachment dir: {e}")))?;
    let file_path = upload_dir
        .join(format!("{}.pdf", Uuid::new_v4()))
        .to_string_lossy()
        .to_string();
    tokio::fs::write(&file_path, &file_bytes)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to write attachment: {e}")))?;

    if let Some(source) = fetch_ready_attachment_source(state, &content_hash).await? {
        return Ok(PendingAttachment {
            filename,
            file_path,
            content_hash,
            vector_document_id: source.vector_document_id,
            size_bytes,
            chunk_count: source.chunk_count,
            full_text: source.full_text,
        });
    }

    let vector_document_id = format!("chat_attachment_{}", content_hash);
    let inserted = sqlx::query(
        "INSERT OR IGNORE INTO chat_attachment_sources (
            content_hash,
            vector_document_id,
            status
        ) VALUES (?, ?, 'processing')",
    )
    .bind(&content_hash)
    .bind(&vector_document_id)
    .execute(&state.db)
    .await?;

    if inserted.rows_affected() == 0 {
        let status = sqlx::query_scalar::<_, String>(
            "SELECT status FROM chat_attachment_sources WHERE content_hash = ?",
        )
        .bind(&content_hash)
        .fetch_optional(&state.db)
        .await?;
        if status.as_deref() == Some("error") {
            sqlx::query(
                "UPDATE chat_attachment_sources
                 SET vector_document_id = ?, status = 'processing', updated_at = datetime('now')
                 WHERE content_hash = ?",
            )
            .bind(&vector_document_id)
            .bind(&content_hash)
            .execute(&state.db)
            .await?;
        } else {
            tokio::fs::remove_file(&file_path).await.ok();
            return Err(AppError::BadRequest(format!(
                "Файл '{}' уже обрабатывается. Повторите отправку позже.",
                filename
            )));
        }
    }

    let form = reqwest::multipart::Form::new()
        .part(
            "pdf",
            reqwest::multipart::Part::bytes(file_bytes)
                .file_name(filename.clone())
                .mime_str("application/pdf")
                .unwrap(),
        )
        .text("document_id", vector_document_id.clone());

    let resp = state
        .python
        .client()
        .post(state.python.url("/chat/attachments/ingest"))
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        sqlx::query(
            "UPDATE chat_attachment_sources SET status = 'error', updated_at = datetime('now') WHERE content_hash = ?",
        )
        .bind(&content_hash)
        .execute(&state.db)
        .await
        .ok();
        tokio::fs::remove_file(&file_path).await.ok();
        return Err(AppError::BadRequest(format!(
            "Не удалось прочитать PDF-вложение '{}': {} {}",
            filename, status, body
        )));
    }

    let result = resp.json::<ChatAttachmentIngestResponse>().await?;
    sqlx::query(
        "UPDATE chat_attachment_sources
         SET full_text = ?, chunk_count = ?, status = 'ready', updated_at = datetime('now')
         WHERE content_hash = ?",
    )
    .bind(&result.full_text)
    .bind(result.chunk_count)
    .bind(&content_hash)
    .execute(&state.db)
    .await?;

    Ok(PendingAttachment {
        filename,
        file_path,
        content_hash,
        vector_document_id,
        size_bytes,
        chunk_count: result.chunk_count,
        full_text: result.full_text,
    })
}

async fn fetch_ready_attachment_source(
    state: &AppState,
    content_hash: &str,
) -> Result<Option<ChatAttachmentSource>, AppError> {
    sqlx::query_as::<_, ChatAttachmentSource>(
        "SELECT vector_document_id, full_text, chunk_count
         FROM chat_attachment_sources
         WHERE content_hash = ? AND status = 'ready' AND full_text IS NOT NULL
         LIMIT 1",
    )
    .bind(content_hash)
    .fetch_optional(&state.db)
    .await
    .map_err(Into::into)
}

async fn insert_message_attachments(
    state: &AppState,
    message_id: &str,
    attachments: &[PendingAttachment],
) -> Result<(), AppError> {
    for attachment in attachments {
        sqlx::query(
            "INSERT INTO chat_attachments (
                id,
                message_id,
                filename,
                file_path,
                content_hash,
                vector_document_id,
                mime_type,
                size_bytes,
                chunk_count,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, 'application/pdf', ?, ?, 'ready')",
        )
        .bind(Uuid::new_v4().to_string())
        .bind(message_id)
        .bind(&attachment.filename)
        .bind(&attachment.file_path)
        .bind(&attachment.content_hash)
        .bind(&attachment.vector_document_id)
        .bind(attachment.size_bytes)
        .bind(attachment.chunk_count)
        .execute(&state.db)
        .await?;
    }
    Ok(())
}

async fn fetch_session_attachment_contexts(
    state: &AppState,
    session_id: &str,
    current_message_id: &str,
    current_attachments: &[PendingAttachment],
) -> Result<Vec<PromptAttachmentContext>, AppError> {
    let mut seen = HashSet::new();
    let mut contexts = Vec::new();

    for attachment in current_attachments {
        if seen.insert(attachment.vector_document_id.clone()) {
            contexts.push(PromptAttachmentContext {
                filename: attachment.filename.clone(),
                vector_document_id: attachment.vector_document_id.clone(),
                full_text: attachment.full_text.clone(),
            });
        }
    }

    if contexts.len() >= MAX_CHAT_ATTACHMENTS {
        return Ok(contexts);
    }

    let previous_rows = sqlx::query_as::<_, (String, String, String)>(
        "SELECT
            a.filename,
            a.vector_document_id,
            src.full_text
         FROM chat_attachments a
         JOIN chat_messages m ON m.id = a.message_id
         JOIN chat_attachment_sources src ON src.content_hash = a.content_hash
         WHERE m.session_id = ?
           AND a.message_id <> ?
           AND a.status = 'ready'
           AND src.status = 'ready'
           AND src.full_text IS NOT NULL
         ORDER BY a.created_at DESC",
    )
    .bind(session_id)
    .bind(current_message_id)
    .fetch_all(&state.db)
    .await?;

    for (filename, vector_document_id, full_text) in previous_rows {
        if !seen.insert(vector_document_id.clone()) {
            continue;
        }
        contexts.push(PromptAttachmentContext {
            filename,
            vector_document_id,
            full_text,
        });
        if contexts.len() >= MAX_CHAT_ATTACHMENTS {
            break;
        }
    }

    Ok(contexts)
}

async fn fetch_message_attachments(
    state: &AppState,
    message_id: &str,
) -> Result<Vec<ChatAttachmentResponse>, AppError> {
    sqlx::query_as::<_, ChatAttachmentResponse>(
        "SELECT
            id,
            message_id,
            filename,
            mime_type,
            size_bytes,
            status,
            created_at
         FROM chat_attachments
         WHERE message_id = ?
         ORDER BY created_at ASC",
    )
    .bind(message_id)
    .fetch_all(&state.db)
    .await
    .map_err(Into::into)
}

fn prompt_attachment_text(text: &str, question: &str) -> String {
    if text.len() <= ATTACHMENT_PROMPT_TEXT_LIMIT {
        return text.to_string();
    }

    let end = text
        .char_indices()
        .take_while(|(idx, _)| *idx <= ATTACHMENT_PROMPT_TEXT_LIMIT)
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(ATTACHMENT_PROMPT_TEXT_LIMIT.min(text.len()));
    let raw_head = &text[..end];
    let head = raw_head
        .rsplit_once('\n')
        .map(|(h, _)| h)
        .unwrap_or(raw_head)
        .trim_end();
    let exact_matches = exact_question_snippets(text, question);
    if !exact_matches.is_empty() {
        return format!(
            "{head}\n\n[... полный текст PDF-вложения сокращен ...]\n\nТочные совпадения из полного PDF по текущему вопросу:\n{exact_matches}\n\n[... дополнительные релевантные фрагменты также извлекаются через поиск по вложению ...]"
        );
    }

    format!(
        "{head}\n\n[... полный текст PDF-вложения сокращен; дополнительные релевантные фрагменты извлекаются через поиск по вложению ...]"
    )
}

fn exact_question_snippets(text: &str, question: &str) -> String {
    const MAX_TERMS: usize = 16;
    const MAX_SNIPPETS: usize = 4;
    const SNIPPET_RADIUS: usize = 360;

    let lower_text = text.to_lowercase();
    let mut seen_terms = HashSet::new();
    let mut seen_positions = Vec::new();
    let mut snippets = Vec::new();

    for term in question
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '-'))
        .filter(|term| term.chars().count() >= 4)
        .take(MAX_TERMS)
    {
        let lower_term = term.to_lowercase();
        if !seen_terms.insert(lower_term.clone()) {
            continue;
        }
        let Some(index) = lower_text.find(&lower_term) else {
            continue;
        };
        if seen_positions
            .iter()
            .any(|position: &usize| position.abs_diff(index) < SNIPPET_RADIUS)
        {
            continue;
        }
        seen_positions.push(index);
        snippets.push(format!(
            "<match term=\"{}\">\n{}\n</match>",
            term,
            snippet_around(text, index, SNIPPET_RADIUS)
        ));
        if snippets.len() >= MAX_SNIPPETS {
            break;
        }
    }

    snippets.join("\n\n")
}

fn snippet_around(text: &str, index: usize, radius: usize) -> String {
    let mut start = index.saturating_sub(radius);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }

    let mut end = (index + radius).min(text.len());
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }

    let prefix = if start > 0 { "... " } else { "" };
    let suffix = if end < text.len() { " ..." } else { "" };
    format!("{prefix}{}{suffix}", text[start..end].trim())
}

async fn emit_sessions_changed(
    state: &AppState,
    user_id: &str,
    document_id: Option<&str>,
    session_id: Option<&str>,
    action: &str,
) -> Result<(), AppError> {
    crate::routes::events::emit_account_event(
        &state.db,
        user_id,
        "sessions_changed",
        "chat_session",
        session_id,
        serde_json::json!({
            "document_id": document_id,
            "session_id": session_id,
            "action": action,
        }),
    )
    .await
}

async fn emit_messages_changed(
    state: &AppState,
    user_id: &str,
    session_id: &str,
    message_id: Option<&str>,
    action: &str,
) -> Result<(), AppError> {
    crate::routes::events::emit_account_event(
        &state.db,
        user_id,
        "messages_changed",
        "chat_message",
        message_id,
        serde_json::json!({
            "session_id": session_id,
            "message_id": message_id,
            "action": action,
        }),
    )
    .await
}
