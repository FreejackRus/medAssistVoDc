use sqlx::{
    SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use std::{path::Path, str::FromStr};

pub async fn init_pool(database_url: &str) -> SqlitePool {
    // Ensure parent directory exists
    if let Some(path) = database_url.strip_prefix("sqlite:") {
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    let options = SqliteConnectOptions::from_str(database_url)
        .expect("Invalid DATABASE_URL")
        .create_if_missing(true)
        .pragma("journal_mode", "WAL")
        .pragma("foreign_keys", "ON");

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await
        .expect("Failed to connect to SQLite database");

    pool
}

pub async fn run_migrations(pool: &SqlitePool) {
    let mut conn = pool
        .acquire()
        .await
        .expect("Failed to acquire migration connection");

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS _migrations (
            name TEXT PRIMARY KEY NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(&mut *conn)
    .await
    .expect("Failed to create _migrations table");

    recover_delegated_user_migration(&mut conn).await;

    let migrations: &[(&str, &str)] = &[
        ("001_init", include_str!("../migrations/001_init.sql")),
        (
            "002_generation_progress",
            include_str!("../migrations/002_generation_progress.sql"),
        ),
        (
            "003_generation_events",
            include_str!("../migrations/003_generation_events.sql"),
        ),
        (
            "004_auth_and_account_events",
            include_str!("../migrations/004_auth_and_account_events.sql"),
        ),
        (
            "005_delegated_user_management",
            include_str!("../migrations/005_delegated_user_management.sql"),
        ),
        (
            "006_document_sources",
            include_str!("../migrations/006_document_sources.sql"),
        ),
        (
            "007_clear_document_full_text",
            include_str!("../migrations/007_clear_document_full_text.sql"),
        ),
        (
            "008_monitoring",
            include_str!("../migrations/008_monitoring.sql"),
        ),
        (
            "009_gpu_metrics",
            include_str!("../migrations/009_gpu_metrics.sql"),
        ),
        (
            "010_chat_attachments",
            include_str!("../migrations/010_chat_attachments.sql"),
        ),
        (
            "011_algorithm_generation_mode",
            include_str!("../migrations/011_algorithm_generation_mode.sql"),
        ),
    ];

    for &(name, sql) in migrations {
        let applied =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM _migrations WHERE name = ?")
                .bind(name)
                .fetch_one(&mut *conn)
                .await
                .unwrap_or(0);

        if applied > 0 {
            continue;
        }

        for statement in sql.split(';') {
            let trimmed = statement.trim();
            if !trimmed.is_empty() {
                sqlx::query(trimmed)
                    .execute(&mut *conn)
                    .await
                    .unwrap_or_else(|e| panic!("Migration {name} failed: {e}"));
            }
        }

        sqlx::query("INSERT INTO _migrations (name) VALUES (?)")
            .bind(name)
            .execute(&mut *conn)
            .await
            .unwrap_or_else(|e| panic!("Failed to record migration {name}: {e}"));

        tracing::info!("Applied migration: {name}");
    }

    sqlx::query("UPDATE algorithms SET status = 'error' WHERE status = 'running'")
        .execute(&mut *conn)
        .await
        .expect("Failed to reset running algorithms");
    sqlx::query("UPDATE chat_messages SET status = 'error' WHERE status = 'running'")
        .execute(&mut *conn)
        .await
        .expect("Failed to reset running chat messages");
}

async fn recover_delegated_user_migration(conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>) {
    let applied = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM _migrations WHERE name = '005_delegated_user_management'",
    )
    .fetch_one(&mut **conn)
    .await
    .unwrap_or(0);
    if applied > 0 {
        return;
    }

    let users_exists = table_exists(conn, "users").await;
    let users_next_exists = table_exists(conn, "users_next").await;
    if users_exists || !users_next_exists {
        return;
    }

    tracing::warn!("Recovering partial 005_delegated_user_management migration");
    sqlx::query("ALTER TABLE users_next RENAME TO users")
        .execute(&mut **conn)
        .await
        .expect("Failed to recover users table");
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_users_created_by ON users(created_by)")
        .execute(&mut **conn)
        .await
        .expect("Failed to create idx_users_created_by");
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
        .execute(&mut **conn)
        .await
        .expect("Failed to create idx_users_role");
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_users_onboarding_token ON users(onboarding_token_hash)",
    )
    .execute(&mut **conn)
    .await
    .expect("Failed to create idx_users_onboarding_token");
    sqlx::query("INSERT INTO _migrations (name) VALUES ('005_delegated_user_management')")
        .execute(&mut **conn)
        .await
        .expect("Failed to record recovered migration 005_delegated_user_management");
}

async fn table_exists(conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>, table: &str) -> bool {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
    )
    .bind(table)
    .fetch_one(&mut **conn)
    .await
    .unwrap_or(0)
        > 0
}
