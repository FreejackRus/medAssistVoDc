use axum::{Json, extract::State};
use serde_json::Value;

use crate::{AppState, auth::AuthUser, error::AppError};

pub async fn get_recommendations(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<Value>, AppError> {
    user.require_password_changed()?;
    let resp = state
        .python
        .client()
        .get(state.python.url("/clinical-recommendations/"))
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(AppError::AiServiceUnavailable(
            "Failed to fetch clinical recommendations".into(),
        ));
    }

    let data: Value = resp
        .json()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to parse recommendations: {e}")))?;

    Ok(Json(data))
}
