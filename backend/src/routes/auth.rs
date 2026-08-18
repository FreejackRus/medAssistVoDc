use axum::{
    Json,
    extract::State,
    http::{HeaderMap, HeaderValue, header},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{AppState, auth, auth::AuthUser, error::AppError};

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct ChangePasswordRequest {
    #[serde(default)]
    pub current_password: Option<String>,
    pub new_password: String,
}

#[derive(Debug, Deserialize)]
pub struct CompleteOnboardingRequest {
    pub token: String,
    pub new_password: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProfileRequest {
    pub display_name: Option<String>,
    pub organization: Option<String>,
    pub position: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AuthUserResponse {
    pub id: String,
    pub username: String,
    pub role: String,
    pub must_change_password: bool,
    pub display_name: Option<String>,
    pub organization: Option<String>,
    pub position: Option<String>,
    pub notes: Option<String>,
    pub profile_fields: serde_json::Value,
    pub allowed_profile_fields: Vec<String>,
    pub active_sessions: i64,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub user: AuthUserResponse,
}

pub async fn login(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoginRequest>,
) -> Result<(HeaderMap, Json<LoginResponse>), AppError> {
    let username = body.username.trim();
    if username.is_empty() || body.password.is_empty() {
        return Err(AppError::BadRequest("Введите логин и пароль".into()));
    }

    let row = sqlx::query_as::<_, (String, String, String, bool)>(
        "SELECT id, password_hash, role, must_change_password FROM users WHERE username = ?",
    )
    .bind(username)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::Unauthorized("Неверный логин или пароль".into()))?;

    if !auth::verify_password(&body.password, &row.1) {
        return Err(AppError::Unauthorized("Неверный логин или пароль".into()));
    }

    let token = auth::generate_token();
    let token_hash = auth::hash_token(&token);
    let session_id = Uuid::new_v4().to_string();
    let user_agent = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.chars().take(500).collect::<String>());

    sqlx::query(
        "INSERT INTO auth_sessions (id, user_id, token_hash, user_agent, expires_at) \
         VALUES (?, ?, ?, ?, datetime('now', '+30 days'))",
    )
    .bind(&session_id)
    .bind(&row.0)
    .bind(&token_hash)
    .bind(user_agent)
    .execute(&state.db)
    .await?;
    sqlx::query("UPDATE users SET last_login_at = datetime('now') WHERE id = ?")
        .bind(&row.0)
        .execute(&state.db)
        .await?;
    crate::routes::events::emit_admin_users_changed(&state.db, &row.0, "login").await?;

    crate::routes::events::emit_account_event(
        &state.db,
        &row.0,
        "presence_changed",
        "auth_session",
        Some(&session_id),
        serde_json::json!({ "active_sessions": active_sessions(&state, &row.0).await? }),
    )
    .await?;

    let response_headers = session_cookie_headers(&state, &token)?;
    Ok((
        response_headers,
        Json(LoginResponse {
            user: user_response(&state, &row.0, username, &row.2, row.3).await?,
        }),
    ))
}

pub async fn me(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<AuthUserResponse>, AppError> {
    Ok(Json(
        user_response(
            &state,
            &user.id,
            &user.username,
            &user.role,
            user.must_change_password,
        )
        .await?,
    ))
}

pub async fn logout(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<(HeaderMap, Json<serde_json::Value>), AppError> {
    sqlx::query("DELETE FROM auth_sessions WHERE id = ?")
        .bind(&user.session_id)
        .execute(&state.db)
        .await?;
    crate::routes::events::emit_account_event(
        &state.db,
        &user.id,
        "presence_changed",
        "auth_session",
        Some(&user.session_id),
        serde_json::json!({ "active_sessions": active_sessions(&state, &user.id).await? }),
    )
    .await?;
    Ok((
        clear_session_cookie_headers(&state)?,
        Json(serde_json::json!({ "ok": true })),
    ))
}

pub async fn change_password(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<ChangePasswordRequest>,
) -> Result<Json<AuthUserResponse>, AppError> {
    validate_new_password(&body.new_password)?;

    let password_hash =
        sqlx::query_scalar::<_, String>("SELECT password_hash FROM users WHERE id = ?")
            .bind(&user.id)
            .fetch_one(&state.db)
            .await?;
    if !user.must_change_password {
        let current_password = body
            .current_password
            .as_deref()
            .ok_or_else(|| AppError::BadRequest("Введите текущий пароль".into()))?;
        if !auth::verify_password(current_password, &password_hash) {
            return Err(AppError::BadRequest("Текущий пароль указан неверно".into()));
        }
        if current_password == body.new_password {
            return Err(AppError::BadRequest(
                "Новый пароль должен отличаться".into(),
            ));
        }
    }

    let new_hash = auth::hash_password(&body.new_password)?;
    sqlx::query(
        "UPDATE users \
         SET password_hash = ?, must_change_password = 0, onboarding_token_hash = NULL, onboarding_expires_at = NULL, updated_at = datetime('now') \
         WHERE id = ?",
    )
    .bind(new_hash)
    .bind(&user.id)
    .execute(&state.db)
    .await?;
    crate::routes::events::emit_admin_users_changed(&state.db, &user.id, "password_changed")
        .await?;

    Ok(Json(
        user_response(&state, &user.id, &user.username, &user.role, false).await?,
    ))
}

pub async fn complete_onboarding(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CompleteOnboardingRequest>,
) -> Result<(HeaderMap, Json<LoginResponse>), AppError> {
    validate_new_password(&body.new_password)?;
    let token_hash = auth::hash_token(body.token.trim());
    let row = sqlx::query_as::<_, (String, String, String)>(
        "SELECT id, username, role FROM users \
         WHERE onboarding_token_hash = ? \
         AND onboarding_expires_at > datetime('now') \
         AND must_change_password = 1",
    )
    .bind(&token_hash)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::BadRequest("Ссылка недействительна или устарела".into()))?;

    let password_hash = auth::hash_password(&body.new_password)?;
    sqlx::query(
        "UPDATE users \
         SET password_hash = ?, must_change_password = 0, onboarding_token_hash = NULL, onboarding_expires_at = NULL, updated_at = datetime('now'), last_login_at = datetime('now') \
         WHERE id = ?",
    )
    .bind(password_hash)
    .bind(&row.0)
    .execute(&state.db)
    .await?;

    let token = auth::generate_token();
    let session_id = Uuid::new_v4().to_string();
    let token_hash = auth::hash_token(&token);
    let user_agent = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.chars().take(500).collect::<String>());

    sqlx::query(
        "INSERT INTO auth_sessions (id, user_id, token_hash, user_agent, expires_at) \
         VALUES (?, ?, ?, ?, datetime('now', '+30 days'))",
    )
    .bind(&session_id)
    .bind(&row.0)
    .bind(&token_hash)
    .bind(user_agent)
    .execute(&state.db)
    .await?;

    crate::routes::events::emit_admin_users_changed(&state.db, &row.0, "onboarding_completed")
        .await?;
    crate::routes::events::emit_account_event(
        &state.db,
        &row.0,
        "presence_changed",
        "auth_session",
        Some(&session_id),
        serde_json::json!({ "active_sessions": active_sessions(&state, &row.0).await? }),
    )
    .await?;

    let response_headers = session_cookie_headers(&state, &token)?;
    Ok((
        response_headers,
        Json(LoginResponse {
            user: user_response(&state, &row.0, &row.1, &row.2, false).await?,
        }),
    ))
}

pub async fn update_profile(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<UpdateProfileRequest>,
) -> Result<Json<AuthUserResponse>, AppError> {
    user.require_password_changed()?;
    let current = fetch_auth_user(&state, &user.id, &user.username, &user.role, false).await?;
    let allowed = current.allowed_profile_fields;
    let can_edit = |field: &str| user.is_admin() || allowed.iter().any(|v| v == field);

    let mut display_name = current.display_name;
    let mut organization = current.organization;
    let mut position = current.position;
    let mut notes = current.notes;

    if body.display_name.is_some() && can_edit("display_name") {
        display_name = clean_optional(body.display_name, 200);
    }
    if body.organization.is_some() && can_edit("organization") {
        organization = clean_optional(body.organization, 200);
    }
    if body.position.is_some() && can_edit("position") {
        position = clean_optional(body.position, 200);
    }
    if body.notes.is_some() && can_edit("notes") {
        notes = clean_optional(body.notes, 2000);
    }

    sqlx::query(
        "UPDATE users SET display_name = ?, organization = ?, position = ?, notes = ?, updated_at = datetime('now') WHERE id = ?",
    )
    .bind(&display_name)
    .bind(&organization)
    .bind(&position)
    .bind(&notes)
    .bind(&user.id)
    .execute(&state.db)
    .await?;
    crate::routes::events::emit_admin_users_changed(&state.db, &user.id, "profile_changed").await?;

    Ok(Json(
        user_response(&state, &user.id, &user.username, &user.role, false).await?,
    ))
}

pub async fn active_sessions(state: &AppState, user_id: &str) -> Result<i64, AppError> {
    Ok(sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM auth_sessions \
         WHERE user_id = ? AND expires_at > datetime('now') \
         AND last_seen_at > datetime('now', '-2 minutes')",
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await
    .unwrap_or(0))
}

async fn user_response(
    state: &AppState,
    user_id: &str,
    username: &str,
    role: &str,
    must_change_password: bool,
) -> Result<AuthUserResponse, AppError> {
    fetch_auth_user(state, user_id, username, role, must_change_password).await
}

fn validate_new_password(password: &str) -> Result<(), AppError> {
    if password.len() < 8 {
        return Err(AppError::BadRequest(
            "Пароль должен содержать минимум 8 символов".into(),
        ));
    }
    if password.len() > 200 {
        return Err(AppError::BadRequest("Пароль слишком длинный".into()));
    }
    Ok(())
}

async fn fetch_auth_user(
    state: &AppState,
    user_id: &str,
    username: &str,
    role: &str,
    must_change_password: bool,
) -> Result<AuthUserResponse, AppError> {
    let row = sqlx::query_as::<_, (Option<String>, Option<String>, Option<String>, Option<String>, String, String)>(
        "SELECT display_name, organization, position, notes, profile_fields, allowed_profile_fields FROM users WHERE id = ?",
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;
    Ok(AuthUserResponse {
        id: user_id.to_string(),
        username: username.to_string(),
        role: role.to_string(),
        must_change_password,
        display_name: row.0,
        organization: row.1,
        position: row.2,
        notes: row.3,
        profile_fields: serde_json::from_str(&row.4).unwrap_or_else(|_| serde_json::json!({})),
        allowed_profile_fields: serde_json::from_str(&row.5).unwrap_or_default(),
        active_sessions: active_sessions(state, user_id).await?,
    })
}

fn clean_optional(value: Option<String>, max_len: usize) -> Option<String> {
    value
        .map(|v| v.trim().chars().take(max_len).collect::<String>())
        .filter(|v| !v.is_empty())
}

fn session_cookie_headers(state: &AppState, token: &str) -> Result<HeaderMap, AppError> {
    let cookie = session_cookie_value(
        &state.config.session_cookie_name,
        token,
        2_592_000,
        state.config.session_cookie_secure,
    );
    cookie_header(cookie)
}

fn clear_session_cookie_headers(state: &AppState) -> Result<HeaderMap, AppError> {
    let cookie = session_cookie_value(
        &state.config.session_cookie_name,
        "",
        0,
        state.config.session_cookie_secure,
    );
    cookie_header(cookie)
}

fn session_cookie_value(name: &str, token: &str, max_age: u32, secure: bool) -> String {
    let secure_attribute = if secure { "; Secure" } else { "" };
    format!(
        "{name}={token}; HttpOnly; SameSite=Strict; Path=/api; Max-Age={max_age}{secure_attribute}"
    )
}

fn cookie_header(cookie: String) -> Result<HeaderMap, AppError> {
    let value = HeaderValue::from_str(&cookie)
        .map_err(|error| AppError::Internal(format!("Failed to create session cookie: {error}")))?;
    let mut headers = HeaderMap::new();
    headers.insert(header::SET_COOKIE, value);
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_cookie_has_security_attributes() {
        let value = session_cookie_value("session", "secret", 2_592_000, true);
        assert!(value.contains("HttpOnly"));
        assert!(value.contains("SameSite=Strict"));
        assert!(value.contains("Path=/api"));
        assert!(value.contains("Max-Age=2592000"));
        assert!(value.contains("Secure"));
    }

    #[test]
    fn clear_cookie_expires_same_scope() {
        let value = session_cookie_value("session", "", 0, false);
        assert!(value.starts_with("session=;"));
        assert!(value.contains("Path=/api"));
        assert!(value.contains("Max-Age=0"));
        assert!(!value.contains("Secure"));
    }
}
