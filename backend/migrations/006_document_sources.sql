CREATE TABLE IF NOT EXISTS document_sources (
    content_hash TEXT PRIMARY KEY NOT NULL,
    vector_document_id TEXT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    diagnosis_name TEXT,
    mkb_code TEXT,
    full_text TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing' CHECK(status IN ('processing', 'ready', 'error')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO document_sources (
    content_hash,
    vector_document_id,
    file_path,
    diagnosis_name,
    mkb_code,
    full_text,
    chunk_count,
    status,
    created_at,
    updated_at
)
SELECT
    d.content_hash,
    d.id,
    d.file_path,
    d.diagnosis_name,
    d.mkb_code,
    d.full_text,
    d.chunk_count,
    d.status,
    d.created_at,
    d.updated_at
FROM documents d
WHERE d.content_hash IS NOT NULL
  AND d.status = 'ready'
  AND d.full_text IS NOT NULL
ORDER BY d.created_at ASC;

CREATE INDEX IF NOT EXISTS idx_document_sources_status ON document_sources(status);
