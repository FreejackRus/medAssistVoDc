ALTER TABLE algorithms
ADD COLUMN generation_mode TEXT NOT NULL DEFAULT 'structured'
CHECK(generation_mode IN ('structured', 'source', 'physician'));
