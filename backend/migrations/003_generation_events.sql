ALTER TABLE algorithms
ADD COLUMN stream_seq INTEGER NOT NULL DEFAULT 0;

ALTER TABLE chat_messages
ADD COLUMN stream_seq INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS generation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_type TEXT NOT NULL CHECK(stream_type IN ('algorithm', 'chat')),
    target_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('token', 'done', 'error')),
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(stream_type, target_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_generation_events_target_seq
ON generation_events(stream_type, target_id, seq);
