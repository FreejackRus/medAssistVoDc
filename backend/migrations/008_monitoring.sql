CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_user_id TEXT,
    actor_username TEXT,
    target_user_id TEXT,
    target_username TEXT,
    scope_user_id TEXT,
    scope_owner_id TEXT,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_scope ON audit_logs(scope_user_id, scope_owner_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_user_id, created_at);

CREATE TABLE IF NOT EXISTS system_metric_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    load_1m REAL,
    load_5m REAL,
    load_15m REAL,
    cpu_count INTEGER NOT NULL DEFAULT 0,
    memory_total_kb INTEGER,
    memory_available_kb INTEGER,
    memory_used_percent REAL,
    active_sessions INTEGER NOT NULL DEFAULT 0,
    running_algorithms INTEGER NOT NULL DEFAULT 0,
    running_chats INTEGER NOT NULL DEFAULT 0,
    processing_documents INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_system_metric_samples_created_at ON system_metric_samples(created_at);
