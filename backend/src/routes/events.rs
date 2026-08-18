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
                let session_active = session_is_active(&pool, &session_id).await.unwrap_or(false);
                if !session_active {
                    return;
                }
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

async fn session_is_active(pool: &SqlitePool, session_id: &str) -> Result<bool, sqlx::Error> {
    sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM auth_sessions WHERE id = ? AND expires_at > datetime('now'))",
    )
    .bind(session_id)
    .fetch_one(pool)
    .await
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

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    #[tokio::test]
    async fn revoked_and_expired_sessions_stop_event_streams() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query("CREATE TABLE auth_sessions (id TEXT PRIMARY KEY, expires_at TEXT NOT NULL)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "INSERT INTO auth_sessions (id, expires_at) VALUES ('active', datetime('now', '+1 hour')), ('expired', datetime('now', '-1 hour'))",
        )
        .execute(&pool)
        .await
        .unwrap();

        assert!(session_is_active(&pool, "active").await.unwrap());
        assert!(!session_is_active(&pool, "expired").await.unwrap());
        sqlx::query("DELETE FROM auth_sessions WHERE id = 'active'")
            .execute(&pool)
            .await
            .unwrap();
        assert!(!session_is_active(&pool, "active").await.unwrap());
    }
}
