use std::time::Duration;

use axum::{
    body::{Body, Bytes},
    extract::{Query, State},
    http::header,
    response::Response,
};
use serde::Deserialize;
use serde_json::Value;
use sqlx::{FromRow, SqlitePool};

use crate::{AppState, auth::AuthUser, error::AppError};

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    pub after: Option<i64>,
}

#[derive(Debug, FromRow)]
struct AccountEventRow {
    id: i64,
    event_type: String,
    payload: Option<String>,
}

pub async fn stream(
    State(state): State<AppState>,
    user: AuthUser,
    Query(query): Query<EventsQuery>,
) -> Result<Response, AppError> {
    user.require_password_changed()?;
    let pool = state.db.clone();
    let user_id = user.id.clone();
    let session_id = user.session_id.clone();
    let stream = async_stream::stream! {
        let mut last_id = query.after.unwrap_or(0).max(0);
        let mut ticks = 0usize;

        loop {
            if ticks % 30 == 0 {
                if let Err(e) = sqlx::query("UPDATE auth_sessions SET last_seen_at = datetime('now') WHERE id = ?")
                    .bind(&session_id)
                    .execute(&pool)
                    .await
                {
                    tracing::error!("Failed to touch event stream session: {}", e);
                }
            }
            ticks = ticks.wrapping_add(1);

            let rows = match sqlx::query_as::<_, AccountEventRow>(
                "SELECT id, event_type, payload FROM account_events WHERE user_id = ? AND id > ? ORDER BY id ASC LIMIT 100",
            )
            .bind(&user_id)
            .bind(last_id)
            .fetch_all(&pool)
            .await
            {
                Ok(rows) => rows,
                Err(e) => {
                    tracing::error!("Failed to read account events: {}", e);
                    yield Ok::<_, std::io::Error>(format_event(last_id + 1, "error", serde_json::json!({ "message": "Не удалось прочитать события" })));
                    return;
                }
            };

            if rows.is_empty() {
                tokio::time::sleep(Duration::from_millis(1000)).await;
                continue;
            }

            for row in rows {
                last_id = row.id;
                let payload = row
                    .payload
                    .as_deref()
                    .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                    .unwrap_or_else(|| serde_json::json!({}));
                yield Ok::<_, std::io::Error>(format_event(row.id, &row.event_type, payload));
            }
        }
    };

    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .map_err(|e| AppError::Internal(format!("Failed to build event stream: {e}")))
}

pub async fn emit_account_event(
    pool: &SqlitePool,
    user_id: &str,
    event_type: &str,
    entity_type: &str,
    entity_id: Option<&str>,
    payload: Value,
) -> Result<(), AppError> {
    sqlx::query(
        "INSERT INTO account_events (user_id, event_type, entity_type, entity_id, payload) \
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(user_id)
    .bind(event_type)
    .bind(entity_type)
    .bind(entity_id)
    .bind(payload.to_string())
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn emit_admin_users_changed(
    pool: &SqlitePool,
    target_user_id: &str,
    action: &str,
) -> Result<(), AppError> {
    let recipients = sqlx::query_scalar::<_, String>(
        "SELECT DISTINCT u.id FROM users u \
         WHERE u.role = 'admin' \
         OR u.id = (SELECT created_by FROM users WHERE id = ?) \
         OR u.id = ?",
    )
    .bind(target_user_id)
    .bind(target_user_id)
    .fetch_all(pool)
    .await?;

    for user_id in recipients {
        emit_account_event(
            pool,
            &user_id,
            "admin_users_changed",
            "user",
            Some(target_user_id),
            serde_json::json!({ "user_id": target_user_id, "action": action }),
        )
        .await?;
    }

    Ok(())
}

fn format_event(id: i64, event_type: &str, payload: Value) -> Bytes {
    Bytes::from(format!(
        "id: {id}\nevent: {event_type}\ndata: {}\n\n",
        payload
    ))
}
