ALTER TABLE algorithms
ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'
CHECK(status IN ('running', 'completed', 'error'));

ALTER TABLE chat_messages
ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'
CHECK(status IN ('running', 'completed', 'error'));

CREATE INDEX IF NOT EXISTS idx_algorithms_status ON algorithms(status);
CREATE INDEX IF NOT EXISTS idx_chat_messages_status ON chat_messages(status);
