use axum::body::Bytes;
use futures::StreamExt;
use sqlx::{FromRow, SqlitePool};
use std::future::Future;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, FromRow)]
struct GenerationEventRow {
    seq: i64,
    event_type: String,
    content: String,
}

pub struct RelayResult {
    pub collected: String,
    pub token_seq: i64,
    pub error: Option<String>,
}

pub async fn insert_generation_event(
    pool: &SqlitePool,
    stream_type: &str,
    target_id: &str,
    seq: i64,
    event_type: &str,
    content: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO generation_events (stream_type, target_id, seq, event_type, content) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(stream_type)
    .bind(target_id)
    .bind(seq)
    .bind(event_type)
    .bind(content)
    .execute(pool)
    .await?;

    Ok(())
}

pub fn format_sse_event(seq: i64, event_type: &str, content: &str) -> Bytes {
    let data = serde_json::to_string(content).unwrap_or_else(|_| "\"\"".to_string());
    Bytes::from(format!("id: {seq}\nevent: {event_type}\ndata: {data}\n\n"))
}

fn parse_sse_string(data: &str) -> Option<String> {
    serde_json::from_str::<String>(data).ok()
}

fn parse_sse_error_message(data: &str) -> String {
    parse_sse_string(data)
        .unwrap_or_else(|| data.to_string())
        .trim()
        .to_string()
}

fn is_error_event(event_type: &str) -> bool {
    event_type == "error" || event_type == "save_error"
}

/// Read an SSE byte stream from an upstream response, relay raw bytes to the
/// client, collect decoded text content, and periodically persist progress.
///
/// The channel sender is best-effort: if the receiver drops after a page
/// refresh, the function still finishes reading upstream and persists text.
pub async fn relay_and_collect_with_events<FToken, TokenFut, FUpdate, UpdateFut>(
    resp: reqwest::Response,
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    mut on_token: FToken,
    mut on_update: FUpdate,
) -> RelayResult
where
    FToken: FnMut(i64, String) -> TokenFut,
    TokenFut: Future<Output = ()>,
    FUpdate: FnMut(String, i64) -> UpdateFut,
    UpdateFut: Future<Output = ()>,
{
    let mut collected = String::new();
    let mut current_event = "message".to_string();
    let mut line_buffer = String::new();
    let mut byte_stream = resp.bytes_stream();
    let mut last_update_len = 0usize;
    let mut last_update_seq = 0i64;
    let mut seq = 0i64;
    let mut error = None;

    while let Some(chunk) = byte_stream.next().await {
        match chunk {
            Ok(bytes) => {
                let _ = tx.send(Ok(bytes.clone())).await;

                let text = String::from_utf8_lossy(&bytes);
                line_buffer.push_str(&text);

                while let Some(pos) = line_buffer.find('\n') {
                    let line = line_buffer[..pos].trim_end_matches('\r').to_string();
                    line_buffer = line_buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        current_event = "message".to_string();
                        continue;
                    }
                    if let Some(event_type) = line.strip_prefix("event: ") {
                        current_event = event_type.trim().to_string();
                        continue;
                    }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if is_error_event(&current_event) {
                            let message = parse_sse_error_message(data);
                            error = Some(if message.is_empty() {
                                "AI-сервис вернул ошибку без описания".to_string()
                            } else {
                                message
                            });
                            break;
                        }
                        if let Some(token) = parse_sse_string(data) {
                            if token.is_empty() {
                                continue;
                            }
                            seq += 1;
                            on_token(seq, token.clone()).await;
                            collected.push_str(&token);
                        }
                    }
                }

                if seq > last_update_seq && collected.len().saturating_sub(last_update_len) >= 256 {
                    last_update_len = collected.len();
                    last_update_seq = seq;
                    on_update(collected.clone(), seq).await;
                }

                if error.is_some() {
                    break;
                }
            }
            Err(e) => {
                tracing::error!("SSE relay error: {e}");
                error = Some("AI-сервис прервал поток ответа".to_string());
                break;
            }
        }
    }

    let remaining = line_buffer.trim();
    if error.is_none() {
        if let Some(data) = remaining.strip_prefix("data: ") {
            if is_error_event(&current_event) {
                let message = parse_sse_error_message(data);
                error = Some(if message.is_empty() {
                    "AI-сервис вернул ошибку без описания".to_string()
                } else {
                    message
                });
            } else if let Some(token) = parse_sse_string(data) {
                if !token.is_empty() {
                    seq += 1;
                    on_token(seq, token.clone()).await;
                    collected.push_str(&token);
                }
            }
        }
    }

    if !collected.is_empty() && seq > last_update_seq {
        on_update(collected.clone(), seq).await;
    }

    RelayResult {
        collected,
        token_seq: seq,
        error,
    }
}

pub fn resume_generation_events(
    pool: SqlitePool,
    stream_type: &'static str,
    target_id: String,
    after_seq: i64,
    status_sql: &'static str,
) -> Result<axum::response::Response, axum::http::Error> {
    let stream = async_stream::stream! {
        let mut last_seq = after_seq.max(0);

        loop {
            let rows = match sqlx::query_as::<_, GenerationEventRow>(
                "SELECT seq, event_type, content FROM generation_events WHERE stream_type = ? AND target_id = ? AND seq > ? ORDER BY seq ASC LIMIT 100",
            )
            .bind(stream_type)
            .bind(&target_id)
            .bind(last_seq)
            .fetch_all(&pool)
            .await
            {
                Ok(rows) => rows,
                Err(e) => {
                    tracing::error!("Failed to read generation events: {}", e);
                    yield Ok::<_, std::io::Error>(format_sse_event(
                        last_seq + 1,
                        "error",
                        "Не удалось восстановить поток генерации",
                    ));
                    return;
                }
            };

            let fetched = rows.len();
            for row in rows {
                last_seq = row.seq;
                let is_terminal = row.event_type == "done" || row.event_type == "error";
                yield Ok::<_, std::io::Error>(format_sse_event(row.seq, &row.event_type, &row.content));
                if is_terminal {
                    return;
                }
            }

            if fetched == 100 {
                continue;
            }

            let status = match sqlx::query_scalar::<_, String>(status_sql)
                .bind(&target_id)
                .fetch_optional(&pool)
                .await
            {
                Ok(status) => status,
                Err(e) => {
                    tracing::error!("Failed to read generation status: {}", e);
                    yield Ok::<_, std::io::Error>(format_sse_event(
                        last_seq + 1,
                        "error",
                        "Не удалось прочитать статус генерации",
                    ));
                    return;
                }
            };

            match status.as_deref() {
                Some("running") => {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                Some("completed") => {
                    yield Ok::<_, std::io::Error>(format_sse_event(last_seq + 1, "done", ""));
                    return;
                }
                Some("error") => {
                    yield Ok::<_, std::io::Error>(format_sse_event(
                        last_seq + 1,
                        "error",
                        "Генерация завершилась ошибкой",
                    ));
                    return;
                }
                _ => {
                    yield Ok::<_, std::io::Error>(format_sse_event(
                        last_seq + 1,
                        "error",
                        "Поток генерации не найден",
                    ));
                    return;
                }
            }
        }
    };

    axum::response::Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(axum::body::Body::from_stream(stream))
}
