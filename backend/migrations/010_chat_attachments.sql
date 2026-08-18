CREATE TABLE IF NOT EXISTS chat_attachment_sources (
    content_hash TEXT PRIMARY KEY NOT NULL,
    vector_document_id TEXT NOT NULL,
    full_text TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing' CHECK(status IN ('processing', 'ready', 'error')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_attachments (
    id TEXT PRIMARY KEY NOT NULL,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    vector_document_id TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'application/pdf',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready' CHECK(status IN ('ready', 'error')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_attachments_message
ON chat_attachments(message_id);

CREATE INDEX IF NOT EXISTS idx_chat_attachments_hash
ON chat_attachments(content_hash);
