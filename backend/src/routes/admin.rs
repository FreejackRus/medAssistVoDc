use axum::{
    Json,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::{
    AppState,
    auth::{self, AdminUser},
    error::AppError,
    models::AdminUserSummary,
};

#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub role: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub display_name: Option<String>,
    pub organization: Option<String>,
    pub position: Option<String>,
    pub notes: Option<String>,
    pub allowed_profile_fields: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct UserWithTemporaryPassword {
    pub user: AdminUserSummary,
    pub temporary_password: String,
    pub onboarding_url: String,
}

#[derive(Debug, Serialize)]
pub struct ResetPasswordResponse {
    pub temporary_password: String,
    pub onboarding_url: String,
}

pub async fn list_users(
    State(state): State<AppState>,
    admin: AdminUser,
) -> Result<Json<Vec<AdminUserSummary>>, AppError> {
    let users = if admin.0.is_admin() {
        sqlx::query_as::<_, AdminUserSummary>(&admin_user_select(
            "GROUP BY u.id ORDER BY u.created_at ASC",
            true,
        ))
        .fetch_all(&state.db)
        .await?
    } else {
        sqlx::query_as::<_, AdminUserSummary>(&admin_user_select(
            "WHERE u.created_by = ? GROUP BY u.id ORDER BY u.created_at ASC",
            false,
        ))
        .bind(&admin.0.id)
        .fetch_all(&state.db)
        .await?
    };
    Ok(Json(users))
}

pub async fn create_user(
    State(state): State<AppState>,
    admin: AdminUser,
    Json(body): Json<CreateUserRequest>,
) -> Result<Json<UserWithTemporaryPassword>, AppError> {
    let username = validate_username(&body.username)?;
    let role = body.role.unwrap_or_else(|| "user".to_string());
    validate_role_for_actor(&admin, &role)?;

    let temporary_password = auth::generate_temporary_password();
    let password_hash = auth::hash_password(&temporary_password)?;
    let onboarding_token = auth::generate_onboarding_token();
    let onboarding_token_hash = auth::hash_token(&onboarding_token);
    let id = Uuid::new_v4().to_string();
    let allowed_profile_fields = if role == "user" {
        r#"["display_name","organization","position"]"#
    } else {
        r#"["display_name","organization","position","notes"]"#
    };

    let insert = sqlx::query(
        "INSERT INTO users (
            id,
            username,
            password_hash,
            role,
            must_change_password,
            created_by,
            allowed_profile_fields,
            onboarding_token_hash,
            onboarding_expires_at
        ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, datetime('now', '+7 days'))",
    )
    .bind(&id)
    .bind(&username)
    .bind(password_hash)
    .bind(&role)
    .bind(&admin.0.id)
    .bind(allowed_profile_fields)
    .bind(onboarding_token_hash)
    .execute(&state.db)
    .await;

    if let Err(e) = insert {
        if let sqlx::Error::Database(db_err) = &e {
            if db_err.is_unique_violation() {
                return Err(AppError::BadRequest("Пользователь уже существует".into()));
            }
        }
        return Err(e.into());
    }

    crate::routes::events::emit_admin_users_changed(&state.db, &id, "created").await?;
    let user = fetch_user_for_actor(&state, &admin, &id).await?;
    crate::routes::monitoring::log_audit_event(
        &state.db,
        Some(&admin.0),
        Some(&id),
        Some(&username),
        Some(&admin.0.id),
        "user_created",
        "user",
        Some(&id),
        serde_json::json!({
            "username": username,
            "role": role,
            "created_by": admin.0.username,
        }),
    )
    .await?;
    Ok(Json(UserWithTemporaryPassword {
        user,
        temporary_password,
        onboarding_url: onboarding_url(&onboarding_token),
    }))
}

pub async fn update_user(
    State(state): State<AppState>,
    admin: AdminUser,
    Path(user_id): Path<String>,
    Json(body): Json<UpdateUserRequest>,
) -> Result<Json<AdminUserSummary>, AppError> {
    let target = fetch_user_for_actor(&state, &admin, &user_id).await?;
    let allowed_profile_fields = normalize_allowed_fields(body.allowed_profile_fields);
    let display_name = clean_optional(body.display_name, 200);
    let organization = clean_optional(body.organization, 200);
    let position = clean_optional(body.position, 200);
    let notes = clean_optional(body.notes, 2000);

    sqlx::query(
        "UPDATE users SET
            display_name = ?,
            organization = ?,
            position = ?,
            notes = ?,
            allowed_profile_fields = ?,
            updated_at = datetime('now')
         WHERE id = ?",
    )
    .bind(&display_name)
    .bind(&organization)
    .bind(&position)
    .bind(&notes)
    .bind(&allowed_profile_fields)
    .bind(&user_id)
    .execute(&state.db)
    .await?;

    crate::routes::events::emit_admin_users_changed(&state.db, &user_id, "updated").await?;
    crate::routes::monitoring::log_audit_event(
        &state.db,
        Some(&admin.0),
        Some(&user_id),
        Some(&target.username),
        target.created_by.as_deref().or(Some(&target.id)),
        "user_updated",
        "user",
        Some(&user_id),
        serde_json::json!({
            "username": target.username,
            "role": target.role,
            "changed_fields": changed_user_fields(
                &target,
                display_name.as_deref(),
                organization.as_deref(),
                position.as_deref(),
                notes.as_deref(),
                &allowed_profile_fields,
            ),
            "allowed_profile_fields_before": target.allowed_profile_fields,
            "allowed_profile_fields_after": allowed_profile_fields,
        }),
    )
    .await?;
    Ok(Json(fetch_user_for_actor(&state, &admin, &user_id).await?))
}

pub async fn reset_password(
    State(state): State<AppState>,
    admin: AdminUser,
    Path(user_id): Path<String>,
) -> Result<Json<ResetPasswordResponse>, AppError> {
    let target = fetch_user_for_actor(&state, &admin, &user_id).await?;

    let temporary_password = auth::generate_temporary_password();
    let password_hash = auth::hash_password(&temporary_password)?;
    let onboarding_token = auth::generate_onboarding_token();
    let onboarding_token_hash = auth::hash_token(&onboarding_token);
    sqlx::query(
        "UPDATE users \
         SET password_hash = ?, must_change_password = 1, onboarding_token_hash = ?, onboarding_expires_at = datetime('now', '+7 days'), updated_at = datetime('now') \
         WHERE id = ?",
    )
    .bind(password_hash)
    .bind(onboarding_token_hash)
    .bind(&user_id)
    .execute(&state.db)
    .await?;
    sqlx::query("DELETE FROM auth_sessions WHERE user_id = ? AND user_id != ?")
        .bind(&user_id)
        .bind(&admin.0.id)
        .execute(&state.db)
        .await?;

    crate::routes::events::emit_account_event(
        &state.db,
        &user_id,
        "presence_changed",
        "user",
        Some(&user_id),
        serde_json::json!({ "must_change_password": true }),
    )
    .await?;
    crate::routes::events::emit_admin_users_changed(&state.db, &user_id, "password_reset").await?;
    crate::routes::monitoring::log_audit_event(
        &state.db,
        Some(&admin.0),
        Some(&user_id),
        Some(&target.username),
        target.created_by.as_deref().or(Some(&target.id)),
        "user_password_reset",
        "user",
        Some(&user_id),
        serde_json::json!({
            "username": target.username,
            "role": target.role,
            "temporary_password_created": true,
            "sessions_revoked": true,
        }),
    )
    .await?;

    Ok(Json(ResetPasswordResponse {
        temporary_password,
        onboarding_url: onboarding_url(&onboarding_token),
    }))
}

pub async fn delete_user(
    State(state): State<AppState>,
    admin: AdminUser,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if user_id == admin.0.id {
        return Err(AppError::BadRequest(
            "Нельзя удалить текущую учетную запись".into(),
        ));
    }
    let target = fetch_user_for_actor(&state, &admin, &user_id).await?;
    crate::routes::monitoring::log_audit_event(
        &state.db,
        Some(&admin.0),
        Some(&user_id),
        Some(&target.username),
        target.created_by.as_deref().or(Some(&target.id)),
        "user_deleted",
        "user",
        Some(&user_id),
        serde_json::json!({
            "username": target.username,
            "role": target.role,
            "created_by": target.creator_username,
        }),
    )
    .await?;
    crate::routes::events::emit_admin_users_changed(&state.db, &user_id, "deleted").await?;

    let documents = sqlx::query_as::<_, (String, String, Option<String>)>(
        "SELECT id, file_path, content_hash FROM documents WHERE user_id = ?",
    )
    .bind(&user_id)
    .fetch_all(&state.db)
    .await?;

    let mut txn = state.db.begin().await?;
    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'algorithm' AND target_id IN (
            SELECT a.id FROM algorithms a
            JOIN documents d ON d.id = a.document_id
            WHERE d.user_id = ?
        )",
    )
    .bind(&user_id)
    .execute(&mut *txn)
    .await?;
    sqlx::query(
        "DELETE FROM generation_events WHERE stream_type = 'chat' AND target_id IN (
            SELECT m.id FROM chat_messages m
            JOIN chat_sessions s ON s.id = m.session_id
            WHERE s.user_id = ?
        )",
    )
    .bind(&user_id)
    .execute(&mut *txn)
    .await?;
    sqlx::query("DELETE FROM documents WHERE user_id = ?")
        .bind(&user_id)
        .execute(&mut *txn)
        .await?;
    sqlx::query("DELETE FROM chat_sessions WHERE user_id = ?")
        .bind(&user_id)
        .execute(&mut *txn)
        .await?;
    sqlx::query("DELETE FROM account_events WHERE user_id = ?")
        .bind(&user_id)
        .execute(&mut *txn)
        .await?;
    sqlx::query("DELETE FROM auth_sessions WHERE user_id = ?")
        .bind(&user_id)
        .execute(&mut *txn)
        .await?;
    sqlx::query("DELETE FROM users WHERE id = ?")
        .bind(&user_id)
        .execute(&mut *txn)
        .await?;
    txn.commit().await?;

    let mut hashes: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (doc_id, file_path, content_hash) in documents {
        if let Some(content_hash) = content_hash {
            hashes
                .entry(content_hash)
                .or_default()
                .push((doc_id, file_path));
        } else {
            tokio::fs::remove_file(file_path).await.ok();
            state
                .python
                .client()
                .delete(state.python.url(&format!("/documents/{}", doc_id)))
                .send()
                .await
                .ok();
        }
    }
    for (content_hash, fallbacks) in hashes {
        if !crate::routes::documents::cleanup_unused_document_source(&state, &content_hash).await? {
            for (doc_id, file_path) in fallbacks {
                tokio::fs::remove_file(file_path).await.ok();
                state
                    .python
                    .client()
                    .delete(state.python.url(&format!("/documents/{}", doc_id)))
                    .send()
                    .await
                    .ok();
            }
        }
    }

    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn fetch_user_for_actor(
    state: &AppState,
    admin: &AdminUser,
    user_id: &str,
) -> Result<AdminUserSummary, AppError> {
    let user = sqlx::query_as::<_, AdminUserSummary>(&admin_user_select(
        "WHERE u.id = ? GROUP BY u.id",
        admin.0.is_admin(),
    ))
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound("Пользователь не найден".into()))?;

    if admin.0.is_admin() {
        return Ok(user);
    }
    if user.created_by.as_deref() == Some(admin.0.id.as_str()) && user.role == "user" {
        return Ok(user);
    }
    Err(AppError::Forbidden(
        "Недостаточно прав для этой учетной записи".into(),
    ))
}

fn admin_user_select(suffix: &str, include_stats: bool) -> String {
    let stats_select = if include_stats {
        "
            (SELECT COUNT(*) FROM documents d WHERE d.user_id = u.id) AS documents_count,
            (
                SELECT COUNT(*)
                FROM account_events ae
                WHERE ae.user_id = u.id
                  AND ae.event_type = 'algorithm_changed'
                  AND ae.payload LIKE '%\"action\":\"running\"%'
            ) AS algorithm_generations_count,
            (
                SELECT COUNT(*)
                FROM chat_messages m
                JOIN chat_sessions cs ON cs.id = m.session_id
                WHERE cs.user_id = u.id
                  AND m.role = 'assistant'
            ) AS chat_generations_count,
            (
                SELECT COUNT(DISTINCT a.document_id)
                FROM algorithms a
                JOIN documents d ON d.id = a.document_id
                WHERE d.user_id = u.id
                  AND a.status = 'completed'
            ) AS documents_with_algorithm_count,
            (
                SELECT COUNT(DISTINCT cs.document_id)
                FROM chat_sessions cs
                JOIN chat_messages m ON m.session_id = cs.id
                WHERE cs.user_id = u.id
                  AND m.role = 'assistant'
            ) AS documents_with_chat_count"
    } else {
        "
            NULL AS documents_count,
            NULL AS algorithm_generations_count,
            NULL AS chat_generations_count,
            NULL AS documents_with_algorithm_count,
            NULL AS documents_with_chat_count"
    };

    format!(
        "SELECT
            u.id,
            u.username,
            u.role,
            u.must_change_password,
            u.created_by,
            c.username AS creator_username,
            u.display_name,
            u.organization,
            u.position,
            u.notes,
            u.profile_fields,
            u.allowed_profile_fields,
            u.onboarding_expires_at,
            u.created_at,
            u.updated_at,
            u.last_login_at,
            COUNT(s.id) AS active_sessions,
            {stats_select}
         FROM users u
         LEFT JOIN users c ON c.id = u.created_by
         LEFT JOIN auth_sessions s ON s.user_id = u.id
            AND s.expires_at > datetime('now')
            AND s.last_seen_at > datetime('now', '-2 minutes')
         {suffix}"
    )
}

fn validate_role_for_actor(admin: &AdminUser, role: &str) -> Result<(), AppError> {
    if role != "admin" && role != "manager" && role != "user" {
        return Err(AppError::BadRequest("Некорректная роль".into()));
    }
    if !admin.0.is_admin() && role != "user" {
        return Err(AppError::Forbidden(
            "Менеджер может создавать только пользователей".into(),
        ));
    }
    Ok(())
}

fn changed_user_fields(
    target: &AdminUserSummary,
    display_name: Option<&str>,
    organization: Option<&str>,
    position: Option<&str>,
    notes: Option<&str>,
    allowed_profile_fields: &str,
) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if target.display_name.as_deref() != display_name {
        fields.push("display_name");
    }
    if target.organization.as_deref() != organization {
        fields.push("organization");
    }
    if target.position.as_deref() != position {
        fields.push("position");
    }
    if target.notes.as_deref() != notes {
        fields.push("notes");
    }
    if target.allowed_profile_fields != allowed_profile_fields {
        fields.push("allowed_profile_fields");
    }
    fields
}

fn validate_username(username: &str) -> Result<String, AppError> {
    let username = username.trim().to_lowercase();
    if username.len() < 3 || username.len() > 64 {
        return Err(AppError::BadRequest(
            "Логин должен содержать от 3 до 64 символов".into(),
        ));
    }
    if !username
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err(AppError::BadRequest(
            "Логин может содержать латинские буквы, цифры, точку, дефис и подчеркивание".into(),
        ));
    }
    Ok(username)
}

fn onboarding_url(token: &str) -> String {
    format!("/welcome?token={token}")
}

fn clean_optional(value: Option<String>, max_len: usize) -> Option<String> {
    value
        .map(|v| v.trim().chars().take(max_len).collect::<String>())
        .filter(|v| !v.is_empty())
}

fn normalize_allowed_fields(fields: Option<Vec<String>>) -> String {
    let allowed = fields
        .unwrap_or_default()
        .into_iter()
        .filter(|field| {
            matches!(
                field.as_str(),
                "display_name" | "organization" | "position" | "notes"
            )
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&allowed).unwrap_or_else(|_| "[]".to_string())
}
