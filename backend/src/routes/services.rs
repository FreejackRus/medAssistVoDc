use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

use crate::{AppState, auth::AuthUser, error::AppError};

#[derive(Deserialize)]
pub struct MatchRequest {
    step_text: String,
    #[serde(default)]
    step_title: String,
}

#[derive(Serialize, Deserialize)]
pub struct ServiceItem {
    id: String,
    name: String,
}

#[derive(Serialize, Deserialize)]
pub struct MatchResponse {
    services: Vec<ServiceItem>,
    step_title: String,
}

pub async fn match_services(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<MatchRequest>,
) -> Result<Json<MatchResponse>, AppError> {
    user.require_password_changed()?;
    if body.step_text.is_empty() || body.step_text.len() > 10_000 {
        return Err(AppError::BadRequest(
            "step_text must be 1-10000 chars".into(),
        ));
    }
    if body.step_title.len() > 1_000 {
        return Err(AppError::BadRequest(
            "step_title must be under 1000 chars".into(),
        ));
    }

    let resp = state
        .python
        .client()
        .post(state.python.url("/services/match"))
        .json(&serde_json::json!({
            "step_text": body.step_text,
            "step_title": body.step_title,
        }))
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(AppError::AiServiceUnavailable(
            "Service matching failed".into(),
        ));
    }

    let result: MatchResponse = resp
        .json()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to parse service match response: {e}")))?;

    Ok(Json(result))
}
