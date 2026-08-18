use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
};
use axum::{
    extract::FromRequestParts,
    http::{HeaderMap, Method, header, request::Parts},
};
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::{AppState, error::AppError};

pub const BOOTSTRAP_ADMIN_ID: &str = "00000000-0000-0000-0000-000000000001";

#[derive(Debug, Clone)]
pub struct AuthUser {
    pub id: String,
    pub username: String,
    pub role: String,
    pub must_change_password: bool,
    pub session_id: String,
}

impl AuthUser {
    pub fn is_admin(&self) -> bool {
        self.role == "admin"
    }

    pub fn can_manage_users(&self) -> bool {
        self.role == "admin" || self.role == "manager"
    }

    pub fn require_password_changed(&self) -> Result<(), AppError> {
        if self.must_change_password {
            return Err(AppError::Forbidden(
                "Необходимо сменить одноразовый пароль".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AdminUser(pub AuthUser);

impl FromRequestParts<AppState> for AuthUser {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let (token, from_cookie) =
            session_token(&parts.headers, &state.config.session_cookie_name)?;
        if from_cookie {
            validate_cookie_origin(&parts.method, &parts.headers, &state.config.cors_origins)?;
        }
        let token_hash = hash_token(&token);

        let row = sqlx::query_as::<_, (String, String, String, bool, String)>(
            "SELECT u.id, u.username, u.role, u.must_change_password, s.id \
             FROM auth_sessions s \
             JOIN users u ON u.id = s.user_id \
             WHERE s.token_hash = ? AND s.expires_at > datetime('now')",
        )
        .bind(&token_hash)
        .fetch_optional(&state.db)
        .await?
        .ok_or_else(|| AppError::Unauthorized("Требуется вход в систему".into()))?;

        sqlx::query("UPDATE auth_sessions SET last_seen_at = datetime('now') WHERE id = ?")
            .bind(&row.4)
            .execute(&state.db)
            .await?;

        Ok(Self {
            id: row.0,
            username: row.1,
            role: row.2,
            must_change_password: row.3,
            session_id: row.4,
        })
    }
}

impl FromRequestParts<AppState> for AdminUser {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let user = AuthUser::from_request_parts(parts, state).await?;
        user.require_password_changed()?;
        if !user.can_manage_users() {
            return Err(AppError::Forbidden("Недостаточно прав".into()));
        }
        Ok(Self(user))
    }
}

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}

pub fn generate_token() -> String {
    format!("{}.{}", Uuid::new_v4(), Uuid::new_v4())
}

pub fn generate_temporary_password() -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("{}-{}-{}", &raw[0..6], &raw[6..12], &raw[12..18])
}

pub fn generate_onboarding_token() -> String {
    format!("onb_{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

pub fn hash_password(password: &str) -> Result<String, AppError> {
    let salt = SaltString::encode_b64(Uuid::new_v4().as_bytes())
        .map_err(|e| AppError::Internal(format!("Failed to create password salt: {e}")))?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|e| AppError::Internal(format!("Failed to hash password: {e}")))
}

pub fn verify_password(password: &str, password_hash: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(password_hash) else {
        return false;
    };
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

pub async fn ensure_bootstrap_admin(pool: &SqlitePool) {
    sqlx::query(
        "INSERT OR IGNORE INTO users (
            id,
            username,
            password_hash,
            role,
            must_change_password,
            allowed_profile_fields
        ) VALUES (?, 'admin', '', 'admin', 1, '[\"display_name\",\"organization\",\"position\",\"notes\"]')",
    )
    .bind(BOOTSTRAP_ADMIN_ID)
    .execute(pool)
    .await
    .expect("Failed to ensure bootstrap admin row");

    let password_hash = sqlx::query_scalar::<_, String>(
        "SELECT password_hash FROM users WHERE id = ? AND role = 'admin'",
    )
    .bind(BOOTSTRAP_ADMIN_ID)
    .fetch_optional(pool)
    .await
    .expect("Failed to read bootstrap admin")
    .unwrap_or_default();

    if !password_hash.is_empty() {
        return;
    }

    let configured_password = std::env::var("ADMIN_PASSWORD")
        .ok()
        .filter(|p| !p.trim().is_empty());
    let password = configured_password.as_deref().unwrap_or("admin");
    let username = std::env::var("ADMIN_USERNAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "admin".to_string());
    let must_change = if configured_password.is_some() { 0 } else { 1 };
    let hash = hash_password(password).expect("Failed to hash bootstrap admin password");

    sqlx::query(
        "UPDATE users \
         SET username = ?, password_hash = ?, must_change_password = ?, updated_at = datetime('now') \
         WHERE id = ?",
    )
    .bind(username)
    .bind(hash)
    .bind(must_change)
    .bind(BOOTSTRAP_ADMIN_ID)
    .execute(pool)
    .await
    .expect("Failed to configure bootstrap admin");

    if configured_password.is_none() {
        tracing::warn!(
            "Created bootstrap admin with username 'admin' and one-time password 'admin'"
        );
    }
}

fn session_token(headers: &HeaderMap, cookie_name: &str) -> Result<(String, bool), AppError> {
    if let Some(token) = cookie_token(headers, cookie_name) {
        return Ok((token, true));
    }
    bearer_token(headers).map(|token| (token, false))
}

fn cookie_token(headers: &HeaderMap, cookie_name: &str) -> Option<String> {
    headers
        .get(header::COOKIE)
        .and_then(|value| value.to_str().ok())
        .and_then(|cookies| {
            cookies.split(';').find_map(|cookie| {
                let (name, value) = cookie.trim().split_once('=')?;
                (name == cookie_name && !value.trim().is_empty()).then(|| value.trim().to_string())
            })
        })
}

fn validate_cookie_origin(
    method: &Method,
    headers: &HeaderMap,
    allowed_origins: &[String],
) -> Result<(), AppError> {
    if matches!(method, &Method::GET | &Method::HEAD | &Method::OPTIONS) {
        return Ok(());
    }
    let Some(origin) = headers
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
    else {
        // Non-browser clients may omit Origin. Browser requests include it for
        // mutating fetches and are checked against the explicit allowlist.
        return Ok(());
    };
    if allowed_origins.iter().any(|allowed| allowed == origin) {
        return Ok(());
    }
    Err(AppError::Forbidden("Недопустимый источник запроса".into()))
}

fn bearer_token(headers: &HeaderMap) -> Result<String, AppError> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AppError::Unauthorized("Требуется вход в систему".into()))?;
    let Some(token) = value.strip_prefix("Bearer ") else {
        return Err(AppError::Unauthorized(
            "Некорректный токен авторизации".into(),
        ));
    };
    if token.trim().is_empty() {
        return Err(AppError::Unauthorized(
            "Некорректный токен авторизации".into(),
        ));
    }
    Ok(token.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookie_token_takes_precedence_over_legacy_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            "other=x; medassist_session=cookie-token".parse().unwrap(),
        );
        headers.insert(
            header::AUTHORIZATION,
            "Bearer legacy-token".parse().unwrap(),
        );

        assert_eq!(
            session_token(&headers, "medassist_session").unwrap(),
            ("cookie-token".into(), true)
        );
    }

    #[test]
    fn bearer_remains_available_during_migration() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            "Bearer legacy-token".parse().unwrap(),
        );

        assert_eq!(
            session_token(&headers, "medassist_session").unwrap(),
            ("legacy-token".into(), false)
        );
    }

    #[test]
    fn rejects_cross_origin_cookie_mutation() {
        let mut headers = HeaderMap::new();
        headers.insert(header::ORIGIN, "https://attacker.example".parse().unwrap());
        let allowed = vec!["https://medical.example".to_string()];

        assert!(matches!(
            validate_cookie_origin(&Method::POST, &headers, &allowed),
            Err(AppError::Forbidden(_))
        ));
        assert!(validate_cookie_origin(&Method::GET, &headers, &allowed).is_ok());
    }
}
