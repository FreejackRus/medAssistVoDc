from __future__ import annotations

from dataclasses import dataclass
from src.config import settings


@dataclass
class Chunk:
    text: str
    metadata: dict  # section_title, page_number, document_id, chunk_index


def chunk_sections(
    sections: dict[str, str],
    document_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split parsed sections into overlapping chunks respecting section boundaries."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    chunks: list[Chunk] = []
    idx = 0

    for title, body in sections.items():
        # Prefix each chunk with the section title for context
        prefix = f"[{title}]\n"
        effective_size = chunk_size - len(prefix)
        if effective_size < 200:
            effective_size = chunk_size

        start = 0
        while start < len(body):
            end = start + effective_size

            # Try to break at paragraph/sentence boundary
            if end < len(body):
                newline_pos = body.rfind("\n", start, end)
                if newline_pos > start + effective_size // 2:
                    end = newline_pos + 1
                else:
                    period_pos = body.rfind(". ", start, end)
                    if period_pos > start + effective_size // 2:
                        end = period_pos + 2

            chunk_text = prefix + body[start:end].strip()
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "section_title": title,
                        "document_id": str(document_id),
                        "chunk_index": idx,
                    },
                ))
                idx += 1

            start = end - chunk_overlap if end < len(body) else len(body)

    return chunks
