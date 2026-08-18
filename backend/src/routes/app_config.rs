use axum::{Json, extract::State};
use serde_json::{Value, json};

use crate::AppState;

pub async fn get_config(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "upload_max_mb": state.config.upload_max_body_size_mb,
    }))
}
