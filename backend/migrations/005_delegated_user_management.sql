PRAGMA foreign_keys = OFF;
DROP TABLE IF EXISTS users_next;

CREATE TABLE users_next (
    id TEXT PRIMARY KEY NOT NULL,
    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'manager', 'user')),
    must_change_password INTEGER NOT NULL DEFAULT 1 CHECK(must_change_password IN (0, 1)),
    created_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    display_name TEXT,
    organization TEXT,
    position TEXT,
    notes TEXT,
    profile_fields TEXT NOT NULL DEFAULT '{}',
    allowed_profile_fields TEXT NOT NULL DEFAULT '[]',
    onboarding_token_hash TEXT,
    onboarding_expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
);

INSERT INTO users_next (
    id,
    username,
    password_hash,
    role,
    must_change_password,
    created_at,
    updated_at,
    last_login_at,
    allowed_profile_fields
)
SELECT
    id,
    username,
    password_hash,
    CASE WHEN role = 'admin' THEN 'admin' ELSE 'user' END,
    must_change_password,
    created_at,
    updated_at,
    last_login_at,
    CASE WHEN role = 'admin' THEN '["display_name","organization","position","notes"]' ELSE '[]' END
FROM users;

DROP TABLE users;
ALTER TABLE users_next RENAME TO users;

CREATE INDEX IF NOT EXISTS idx_users_created_by ON users(created_by);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_onboarding_token ON users(onboarding_token_hash);
PRAGMA foreign_keys = ON;
