use axum::{Json, extract::State};
use serde_json::{Value, json};

use crate::AppState;

pub async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let db_ok = sqlx::query("SELECT 1").execute(&state.db).await.is_ok();

    let ai_ok = state.python.health().await.is_ok();

    Json(json!({
        "status": if db_ok && ai_ok { "ok" } else { "degraded" },
        "db": if db_ok { "ok" } else { "error" },
        "ai_service": if ai_ok { "ok" } else { "unavailable" },
    }))
}
