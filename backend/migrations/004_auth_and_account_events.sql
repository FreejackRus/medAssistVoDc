CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY NOT NULL,
    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'user')),
    must_change_password INTEGER NOT NULL DEFAULT 1 CHECK(must_change_password IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
);

INSERT OR IGNORE INTO users (id, username, password_hash, role, must_change_password)
VALUES ('00000000-0000-0000-0000-000000000001', 'admin', '', 'admin', 1);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT UNIQUE NOT NULL,
    user_agent TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

ALTER TABLE documents ADD COLUMN user_id TEXT;
UPDATE documents SET user_id = '00000000-0000-0000-0000-000000000001' WHERE user_id IS NULL;

DROP INDEX IF EXISTS idx_documents_content_hash;
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_user_content_hash
ON documents(user_id, content_hash)
WHERE content_hash IS NOT NULL AND status != 'error';

ALTER TABLE chat_sessions ADD COLUMN user_id TEXT;
UPDATE chat_sessions
SET user_id = (
    SELECT d.user_id FROM documents d WHERE d.id = chat_sessions.document_id
)
WHERE user_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS account_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    payload TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_account_events_user_id ON account_events(user_id, id);
