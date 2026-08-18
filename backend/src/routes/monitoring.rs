use std::{collections::HashMap, fs, process::Command};

use axum::{
    Json,
    extract::{Query, State},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{FromRow, SqlitePool};

use crate::{
    AppState,
    auth::{AdminUser, AuthUser},
    error::AppError,
};

#[derive(Debug, Deserialize)]
pub struct MonitoringQuery {
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MonitoringSummary {
    pub scope: String,
    pub from: String,
    pub to: String,
    pub current: CurrentGenerationStats,
    pub system: Option<SystemSnapshot>,
    pub history: Vec<ActionCount>,
    pub top_users: Vec<UserActivitySummary>,
    pub metrics: Vec<SystemMetricSample>,
    pub logs: Vec<AuditLogEntry>,
}

#[derive(Debug, Serialize)]
pub struct CurrentGenerationStats {
    pub running_algorithms: i64,
    pub running_chats: i64,
    pub active_sessions: i64,
    pub processing_documents: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuMetric {
    pub index: i64,
    pub name: String,
    pub utilization_gpu_percent: Option<f64>,
    pub memory_used_mb: Option<f64>,
    pub memory_total_mb: Option<f64>,
    pub temperature_c: Option<f64>,
    pub power_draw_w: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct SystemSnapshot {
    pub load_1m: Option<f64>,
    pub load_5m: Option<f64>,
    pub load_15m: Option<f64>,
    pub cpu_count: i64,
    pub memory_total_kb: Option<i64>,
    pub memory_available_kb: Option<i64>,
    pub memory_used_percent: Option<f64>,
    pub gpu_metrics: Vec<GpuMetric>,
}

#[derive(Debug, Serialize, FromRow)]
pub struct ActionCount {
    pub action: String,
    pub count: i64,
}

#[derive(Debug, Serialize, FromRow)]
pub struct UserActivitySummary {
    pub user_id: Option<String>,
    pub username: String,
    pub user_status: String,
    pub algorithm_generations: i64,
    pub chat_generations: i64,
    pub admin_actions: i64,
}

#[derive(Debug, Serialize)]
pub struct SystemMetricSample {
    pub id: i64,
    pub load_1m: Option<f64>,
    pub load_5m: Option<f64>,
    pub load_15m: Option<f64>,
    pub cpu_count: i64,
    pub memory_total_kb: Option<i64>,
    pub memory_available_kb: Option<i64>,
    pub memory_used_percent: Option<f64>,
    pub active_sessions: i64,
    pub running_algorithms: i64,
    pub running_chats: i64,
    pub processing_documents: i64,
    pub gpu_metrics: Vec<GpuMetric>,
    pub created_at: String,
}

#[derive(Debug, FromRow)]
struct SystemMetricSampleRow {
    pub id: i64,
    pub load_1m: Option<f64>,
    pub load_5m: Option<f64>,
    pub load_15m: Option<f64>,
    pub cpu_count: i64,
    pub memory_total_kb: Option<i64>,
    pub memory_available_kb: Option<i64>,
    pub memory_used_percent: Option<f64>,
    pub active_sessions: i64,
    pub running_algorithms: i64,
    pub running_chats: i64,
    pub processing_documents: i64,
    pub gpu_metrics: Option<String>,
    pub created_at: String,
}

impl From<SystemMetricSampleRow> for SystemMetricSample {
    fn from(row: SystemMetricSampleRow) -> Self {
        Self {
            id: row.id,
            load_1m: row.load_1m,
            load_5m: row.load_5m,
            load_15m: row.load_15m,
            cpu_count: row.cpu_count,
            memory_total_kb: row.memory_total_kb,
            memory_available_kb: row.memory_available_kb,
            memory_used_percent: row.memory_used_percent,
            active_sessions: row.active_sessions,
            running_algorithms: row.running_algorithms,
            running_chats: row.running_chats,
            processing_documents: row.processing_documents,
            gpu_metrics: row
                .gpu_metrics
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Vec<GpuMetric>>(raw).ok())
                .unwrap_or_default(),
            created_at: row.created_at,
        }
    }
}

#[derive(Debug, Serialize, FromRow)]
pub struct AuditLogEntry {
    pub id: i64,
    pub actor_user_id: Option<String>,
    pub actor_username: Option<String>,
    pub target_user_id: Option<String>,
    pub target_username: Option<String>,
    pub action: String,
    pub entity_type: String,
    pub entity_id: Option<String>,
    pub payload: String,
    pub created_at: String,
}

pub async fn summary(
    State(state): State<AppState>,
    admin: AdminUser,
    Query(query): Query<MonitoringQuery>,
) -> Result<Json<MonitoringSummary>, AppError> {
    let (from, to) = normalize_period(query)?;
    let current = scoped_current_stats(&state.db, &admin.0).await?;
    let system = if admin.0.is_admin() {
        let snapshot = collect_system_snapshot();
        store_system_metric_sample(&state.db, &snapshot).await?;
        Some(snapshot)
    } else {
        None
    };

    let history = action_counts(&state.db, &admin.0, &from, &to).await?;
    let top_users = user_activity(&state.db, &admin.0, &from, &to).await?;
    let metrics = if admin.0.is_admin() {
        system_metrics(&state.db, &from, &to).await?
    } else {
        Vec::new()
    };
    let logs = audit_logs(&state.db, &admin.0, &from, &to).await?;

    Ok(Json(MonitoringSummary {
        scope: if admin.0.is_admin() {
            "all".to_string()
        } else {
            "managed".to_string()
        },
        from,
        to,
        current,
        system,
        history,
        top_users,
        metrics,
        logs,
    }))
}

pub async fn log_audit_event(
    pool: &SqlitePool,
    actor: Option<&AuthUser>,
    target_user_id: Option<&str>,
    target_username: Option<&str>,
    scope_owner_id: Option<&str>,
    action: &str,
    entity_type: &str,
    entity_id: Option<&str>,
    payload: Value,
) -> Result<(), AppError> {
    sqlx::query(
        "INSERT INTO audit_logs (
            actor_user_id,
            actor_username,
            target_user_id,
            target_username,
            scope_user_id,
            scope_owner_id,
            action,
            entity_type,
            entity_id,
            payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(actor.map(|user| user.id.as_str()))
    .bind(actor.map(|user| user.username.as_str()))
    .bind(target_user_id)
    .bind(target_username)
    .bind(target_user_id.or_else(|| actor.map(|user| user.id.as_str())))
    .bind(scope_owner_id)
    .bind(action)
    .bind(entity_type)
    .bind(entity_id)
    .bind(payload.to_string())
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn log_system_event(
    pool: &SqlitePool,
    user_id: &str,
    username: &str,
    action: &str,
    entity_type: &str,
    entity_id: Option<&str>,
    payload: Value,
) -> Result<(), AppError> {
    let scope_owner_id =
        sqlx::query_scalar::<_, String>("SELECT COALESCE(created_by, id) FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_optional(pool)
            .await?
            .unwrap_or_else(|| user_id.to_string());

    sqlx::query(
        "INSERT INTO audit_logs (
            actor_user_id,
            actor_username,
            target_user_id,
            target_username,
            scope_user_id,
            scope_owner_id,
            action,
            entity_type,
            entity_id,
            payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(user_id)
    .bind(username)
    .bind(user_id)
    .bind(username)
    .bind(user_id)
    .bind(scope_owner_id)
    .bind(action)
    .bind(entity_type)
    .bind(entity_id)
    .bind(payload.to_string())
    .execute(pool)
    .await?;
    Ok(())
}

async fn scoped_current_stats(
    pool: &SqlitePool,
    actor: &AuthUser,
) -> Result<CurrentGenerationStats, AppError> {
    Ok(CurrentGenerationStats {
        running_algorithms: scoped_count(
            pool,
            actor,
            "SELECT COUNT(*) FROM algorithms a JOIN documents d ON d.id = a.document_id WHERE a.status = 'running'",
            "SELECT COUNT(*) FROM algorithms a JOIN documents d ON d.id = a.document_id WHERE a.status = 'running' AND (d.user_id = ? OR d.user_id IN (SELECT id FROM users WHERE created_by = ?))",
        )
        .await?,
        running_chats: scoped_count(
            pool,
            actor,
            "SELECT COUNT(*) FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id WHERE m.role = 'assistant' AND m.status = 'running'",
            "SELECT COUNT(*) FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id WHERE m.role = 'assistant' AND m.status = 'running' AND (s.user_id = ? OR s.user_id IN (SELECT id FROM users WHERE created_by = ?))",
        )
        .await?,
        active_sessions: scoped_count(
            pool,
            actor,
            "SELECT COUNT(*) FROM auth_sessions WHERE expires_at > datetime('now') AND last_seen_at > datetime('now', '-2 minutes')",
            "SELECT COUNT(*) FROM auth_sessions WHERE expires_at > datetime('now') AND last_seen_at > datetime('now', '-2 minutes') AND (user_id = ? OR user_id IN (SELECT id FROM users WHERE created_by = ?))",
        )
        .await?,
        processing_documents: scoped_count(
            pool,
            actor,
            "SELECT COUNT(*) FROM documents WHERE status = 'processing'",
            "SELECT COUNT(*) FROM documents WHERE status = 'processing' AND (user_id = ? OR user_id IN (SELECT id FROM users WHERE created_by = ?))",
        )
        .await?,
    })
}

async fn scoped_count(
    pool: &SqlitePool,
    actor: &AuthUser,
    admin_sql: &str,
    manager_sql: &str,
) -> Result<i64, AppError> {
    let count = if actor.is_admin() {
        sqlx::query_scalar::<_, i64>(admin_sql)
            .fetch_one(pool)
            .await?
    } else {
        sqlx::query_scalar::<_, i64>(manager_sql)
            .bind(&actor.id)
            .bind(&actor.id)
            .fetch_one(pool)
            .await?
    };
    Ok(count)
}

async fn action_counts(
    pool: &SqlitePool,
    actor: &AuthUser,
    from: &str,
    to: &str,
) -> Result<Vec<ActionCount>, AppError> {
    let rows = if actor.is_admin() {
        sqlx::query_as::<_, ActionCount>(
            "SELECT action, COUNT(*) AS count
             FROM audit_logs
             WHERE created_at >= ? AND created_at <= ?
             GROUP BY action
             ORDER BY count DESC, action ASC",
        )
        .bind(from)
        .bind(to)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, ActionCount>(
            "SELECT action, COUNT(*) AS count
             FROM audit_logs
             WHERE created_at >= ? AND created_at <= ?
               AND (scope_user_id = ? OR scope_owner_id = ? OR actor_user_id = ?)
             GROUP BY action
             ORDER BY count DESC, action ASC",
        )
        .bind(from)
        .bind(to)
        .bind(&actor.id)
        .bind(&actor.id)
        .bind(&actor.id)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}

async fn user_activity(
    pool: &SqlitePool,
    actor: &AuthUser,
    from: &str,
    to: &str,
) -> Result<Vec<UserActivitySummary>, AppError> {
    let sql = "SELECT
            actor_user_id AS user_id,
            COALESCE(actor_username, 'system') AS username,
            CASE
                WHEN actor_user_id IS NULL THEN 'system'
                WHEN u.id IS NULL THEN 'deleted'
                ELSE 'active'
            END AS user_status,
            SUM(CASE WHEN action = 'algorithm_generation_started' THEN 1 ELSE 0 END) AS algorithm_generations,
            SUM(CASE WHEN action = 'chat_generation_started' THEN 1 ELSE 0 END) AS chat_generations,
            SUM(CASE WHEN action IN ('user_created', 'user_updated', 'user_deleted', 'user_password_reset') THEN 1 ELSE 0 END) AS admin_actions
         FROM audit_logs
         LEFT JOIN users u ON u.id = audit_logs.actor_user_id
         WHERE audit_logs.created_at >= ? AND audit_logs.created_at <= ?";

    let rows = if actor.is_admin() {
        sqlx::query_as::<_, UserActivitySummary>(&format!(
            "{sql}
             GROUP BY actor_user_id, username, user_status
             HAVING algorithm_generations > 0 OR chat_generations > 0 OR admin_actions > 0
             ORDER BY (algorithm_generations + chat_generations + admin_actions) DESC
             LIMIT 10"
        ))
        .bind(from)
        .bind(to)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, UserActivitySummary>(&format!(
            "{sql}
               AND (
                   actor_user_id = ?
                   OR actor_user_id IN (SELECT id FROM users WHERE created_by = ?)
                   OR (u.id IS NULL AND actor_user_id IS NOT NULL AND scope_owner_id = ?)
               )
             GROUP BY actor_user_id, username, user_status
             HAVING algorithm_generations > 0 OR chat_generations > 0 OR admin_actions > 0
             ORDER BY (algorithm_generations + chat_generations + admin_actions) DESC
             LIMIT 10"
        ))
        .bind(from)
        .bind(to)
        .bind(&actor.id)
        .bind(&actor.id)
        .bind(&actor.id)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}

async fn audit_logs(
    pool: &SqlitePool,
    actor: &AuthUser,
    from: &str,
    to: &str,
) -> Result<Vec<AuditLogEntry>, AppError> {
    let rows = if actor.is_admin() {
        sqlx::query_as::<_, AuditLogEntry>(
            "SELECT
                id,
                actor_user_id,
                actor_username,
                target_user_id,
                target_username,
                action,
                entity_type,
                entity_id,
                payload,
                created_at
             FROM audit_logs
             WHERE created_at >= ? AND created_at <= ?
             ORDER BY id DESC
             LIMIT 200",
        )
        .bind(from)
        .bind(to)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, AuditLogEntry>(
            "SELECT
                id,
                actor_user_id,
                actor_username,
                target_user_id,
                target_username,
                action,
                entity_type,
                entity_id,
                payload,
                created_at
             FROM audit_logs
             WHERE created_at >= ? AND created_at <= ?
               AND (scope_user_id = ? OR scope_owner_id = ? OR actor_user_id = ?)
             ORDER BY id DESC
             LIMIT 200",
        )
        .bind(from)
        .bind(to)
        .bind(&actor.id)
        .bind(&actor.id)
        .bind(&actor.id)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}

async fn system_metrics(
    pool: &SqlitePool,
    from: &str,
    to: &str,
) -> Result<Vec<SystemMetricSample>, AppError> {
    let rows = sqlx::query_as::<_, SystemMetricSampleRow>(
        "SELECT
            id,
            load_1m,
            load_5m,
            load_15m,
            cpu_count,
            memory_total_kb,
            memory_available_kb,
            memory_used_percent,
            active_sessions,
            running_algorithms,
            running_chats,
            processing_documents,
            gpu_metrics,
            created_at
         FROM system_metric_samples
         WHERE created_at >= ? AND created_at <= ?
         ORDER BY id DESC
         LIMIT 200",
    )
    .bind(from)
    .bind(to)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(Into::into).collect())
}

async fn store_system_metric_sample(
    pool: &SqlitePool,
    snapshot: &SystemSnapshot,
) -> Result<(), AppError> {
    let global = CurrentGenerationStats {
        running_algorithms: sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM algorithms WHERE status = 'running'",
        )
        .fetch_one(pool)
        .await
        .unwrap_or(0),
        running_chats: sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM chat_messages WHERE role = 'assistant' AND status = 'running'",
        )
        .fetch_one(pool)
        .await
        .unwrap_or(0),
        active_sessions: sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM auth_sessions WHERE expires_at > datetime('now') AND last_seen_at > datetime('now', '-2 minutes')",
        )
        .fetch_one(pool)
        .await
        .unwrap_or(0),
        processing_documents: sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM documents WHERE status = 'processing'",
        )
        .fetch_one(pool)
        .await
        .unwrap_or(0),
    };

    let gpu_metrics =
        serde_json::to_string(&snapshot.gpu_metrics).unwrap_or_else(|_| "[]".to_string());

    sqlx::query(
        "INSERT INTO system_metric_samples (
            load_1m,
            load_5m,
            load_15m,
            cpu_count,
            memory_total_kb,
            memory_available_kb,
            memory_used_percent,
            active_sessions,
            running_algorithms,
            running_chats,
            processing_documents,
            gpu_metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(snapshot.load_1m)
    .bind(snapshot.load_5m)
    .bind(snapshot.load_15m)
    .bind(snapshot.cpu_count)
    .bind(snapshot.memory_total_kb)
    .bind(snapshot.memory_available_kb)
    .bind(snapshot.memory_used_percent)
    .bind(global.active_sessions)
    .bind(global.running_algorithms)
    .bind(global.running_chats)
    .bind(global.processing_documents)
    .bind(gpu_metrics)
    .execute(pool)
    .await?;
    Ok(())
}

fn collect_system_snapshot() -> SystemSnapshot {
    let (load_1m, load_5m, load_15m) = read_loadavg();
    let (memory_total_kb, memory_available_kb) = read_meminfo();
    let memory_used_percent = match (memory_total_kb, memory_available_kb) {
        (Some(total), Some(available)) if total > 0 => {
            Some(((total - available) as f64 / total as f64 * 100.0).clamp(0.0, 100.0))
        }
        _ => None,
    };

    SystemSnapshot {
        load_1m,
        load_5m,
        load_15m,
        cpu_count: read_cpu_count(),
        memory_total_kb,
        memory_available_kb,
        memory_used_percent,
        gpu_metrics: read_gpu_metrics(),
    }
}

fn read_gpu_metrics() -> Vec<GpuMetric> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let Ok(raw) = String::from_utf8(output.stdout) else {
        return Vec::new();
    };

    raw.lines().filter_map(parse_gpu_metric_line).collect()
}

fn parse_gpu_metric_line(line: &str) -> Option<GpuMetric> {
    let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() < 7 {
        return None;
    }

    Some(GpuMetric {
        index: parts.first()?.parse::<i64>().ok()?,
        name: parts.get(1)?.to_string(),
        utilization_gpu_percent: parse_optional_gpu_number(parts.get(2)?),
        memory_used_mb: parse_optional_gpu_number(parts.get(3)?),
        memory_total_mb: parse_optional_gpu_number(parts.get(4)?),
        temperature_c: parse_optional_gpu_number(parts.get(5)?),
        power_draw_w: parse_optional_gpu_number(parts.get(6)?),
    })
}

fn parse_optional_gpu_number(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("N/A")
        || value.eq_ignore_ascii_case("[Not Supported]")
    {
        return None;
    }
    value.parse::<f64>().ok()
}

fn read_loadavg() -> (Option<f64>, Option<f64>, Option<f64>) {
    let Ok(raw) = fs::read_to_string("/proc/loadavg") else {
        return (None, None, None);
    };
    let values = raw
        .split_whitespace()
        .take(3)
        .map(|v| v.parse::<f64>().ok())
        .collect::<Vec<_>>();
    (
        values.first().copied().flatten(),
        values.get(1).copied().flatten(),
        values.get(2).copied().flatten(),
    )
}

fn read_cpu_count() -> i64 {
    let Ok(raw) = fs::read_to_string("/proc/stat") else {
        return 0;
    };
    raw.lines()
        .filter(|line| {
            line.strip_prefix("cpu")
                .and_then(|rest| rest.chars().next())
                .is_some_and(|c| c.is_ascii_digit())
        })
        .count() as i64
}

fn read_meminfo() -> (Option<i64>, Option<i64>) {
    let Ok(raw) = fs::read_to_string("/proc/meminfo") else {
        return (None, None);
    };
    let mut values = HashMap::new();
    for line in raw.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next().and_then(|v| v.parse::<i64>().ok()) else {
            continue;
        };
        values.insert(key.trim_end_matches(':').to_string(), value);
    }
    (
        values.get("MemTotal").copied(),
        values
            .get("MemAvailable")
            .copied()
            .or_else(|| values.get("MemFree").copied()),
    )
}

fn normalize_period(query: MonitoringQuery) -> Result<(String, String), AppError> {
    let now = chrono::Utc::now().naive_utc();
    let default_from = now - chrono::Duration::hours(24);
    let from = query
        .from
        .unwrap_or_else(|| default_from.format("%Y-%m-%d %H:%M:%S").to_string());
    let to = query
        .to
        .unwrap_or_else(|| now.format("%Y-%m-%d %H:%M:%S").to_string());
    Ok((normalize_datetime(&from)?, normalize_datetime(&to)?))
}

fn normalize_datetime(value: &str) -> Result<String, AppError> {
    let value = value.trim().replace('T', " ");
    if value.len() < 10 || value.len() > 19 {
        return Err(AppError::BadRequest(
            "Некорректный период мониторинга".into(),
        ));
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_digit() || c == '-' || c == ':' || c == ' ')
    {
        return Err(AppError::BadRequest(
            "Некорректный период мониторинга".into(),
        ));
    }
    Ok(if value.len() == 10 {
        format!("{value} 00:00:00")
    } else if value.len() == 16 {
        format!("{value}:00")
    } else {
        value
    })
}
