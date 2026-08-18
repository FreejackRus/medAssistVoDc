UPDATE documents
SET full_text = NULL
WHERE full_text IS NOT NULL
  AND content_hash IN (
      SELECT content_hash
      FROM document_sources
      WHERE status = 'ready'
        AND full_text IS NOT NULL
  );
