use axum::{
    Json,
    extract::{Query, State},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AppState, auth::AuthUser, error::AppError};

#[derive(Debug, Deserialize, Serialize)]
pub struct RecommendationsQuery {
    #[serde(default)]
    q: String,
    #[serde(default = "default_page")]
    page: u32,
    #[serde(default = "default_page_size")]
    page_size: u32,
}

fn default_page() -> u32 {
    1
}

fn default_page_size() -> u32 {
    20
}

fn validate_query(query: &RecommendationsQuery) -> Result<(), AppError> {
    if query.page == 0 {
        return Err(AppError::BadRequest("page must be at least 1".into()));
    }
    if !(1..=100).contains(&query.page_size) {
        return Err(AppError::BadRequest(
            "page_size must be between 1 and 100".into(),
        ));
    }
    if query.q.chars().count() > 200 {
        return Err(AppError::BadRequest(
            "search query must not exceed 200 characters".into(),
        ));
    }
    Ok(())
}

pub async fn get_recommendations(
    State(state): State<AppState>,
    user: AuthUser,
    Query(query): Query<RecommendationsQuery>,
) -> Result<Json<Value>, AppError> {
    user.require_password_changed()?;
    validate_query(&query)?;
    let resp = state
        .python
        .client()
        .get(state.python.url("/clinical-recommendations/"))
        .query(&query)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_pagination_bounds() {
        let valid = RecommendationsQuery {
            q: "терапия".into(),
            page: 1,
            page_size: 20,
        };
        assert!(validate_query(&valid).is_ok());

        assert!(matches!(
            validate_query(&RecommendationsQuery { page: 0, ..valid }),
            Err(AppError::BadRequest(_))
        ));
    }

    #[test]
    fn rejects_oversized_page_and_query() {
        let oversized_page = RecommendationsQuery {
            q: String::new(),
            page: 1,
            page_size: 101,
        };
        assert!(matches!(
            validate_query(&oversized_page),
            Err(AppError::BadRequest(_))
        ));

        let oversized_query = RecommendationsQuery {
            q: "я".repeat(201),
            page: 1,
            page_size: 20,
        };
        assert!(matches!(
            validate_query(&oversized_query),
            Err(AppError::BadRequest(_))
        ));
    }
}
