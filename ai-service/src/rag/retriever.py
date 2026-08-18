from __future__ import annotations

from src.rag.vector_store import get_document_chunks, query_chunks

BROAD_OVERVIEW_MARKERS = (
    "про что",
    "о чем",
    "о чём",
    "что за документ",
    "кратко",
    "в целом",
    "обзор",
)
BROAD_TREATMENT_MARKERS = (
    "как леч",
    "лечение",
    "лечить",
    "терап",
    "алгоритм",
    "тактика",
)


def _needs_anchor_context(query: str) -> bool:
    q = query.lower()
    return any(marker in q for marker in BROAD_OVERVIEW_MARKERS + BROAD_TREATMENT_MARKERS)


def _section_priority(title: str, wants_treatment: bool) -> int | None:
    lower = title.lower()
    stripped = lower.strip()

    if stripped.startswith("1.1") or "определение" in lower:
        return 0
    if stripped.startswith("1.6") or "клиническая картина" in lower:
        return 1

    if not wants_treatment:
        return None

    if "инсулинотерап" in lower:
        return 2
    if stripped.startswith("3.") and ("лечение" in lower or "терап" in lower):
        return 3
    if stripped.startswith("3."):
        return 4

    return None


def _anchor_chunks(query: str, document_id: str, max_chunks: int = 4) -> list[dict]:
    if not _needs_anchor_context(query):
        return []

    q = query.lower()
    wants_treatment = any(marker in q for marker in BROAD_TREATMENT_MARKERS)
    ranked: list[tuple[int, int, dict]] = []
    seen_sections: set[str] = set()

    for chunk in get_document_chunks(document_id):
        metadata = chunk.get("metadata") or {}
        section_title = str(metadata.get("section_title") or "")
        section_key = section_title.strip().lower()
        if not section_key or section_key in seen_sections:
            continue

        priority = _section_priority(section_title, wants_treatment)
        if priority is None:
            continue

        raw_index = metadata.get("chunk_index", 0)
        try:
            chunk_index = int(raw_index)
        except (TypeError, ValueError):
            chunk_index = 0

        ranked.append((priority, chunk_index, chunk))
        seen_sections.add(section_key)

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [chunk for _, _, chunk in ranked[:max_chunks]]


def _merge_chunks(anchor_results: list[dict], semantic_results: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen_indices: set[str] = set()

    for chunk in anchor_results + semantic_results:
        metadata = chunk.get("metadata") or {}
        chunk_index = metadata.get("chunk_index")
        dedupe_key = str(chunk_index) if chunk_index is not None else chunk.get("text", "")
        if dedupe_key in seen_indices:
            continue
        seen_indices.add(dedupe_key)
        merged.append(chunk)

    return merged


def retrieve(query: str, document_id: str, top_k: int = 8) -> str:
    """Retrieve relevant chunks and format them as context string."""
    semantic_results = query_chunks(query, document_id=document_id, top_k=top_k)
    results = _merge_chunks(_anchor_chunks(query, document_id), semantic_results)
    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, 1):
        section = r["metadata"].get("section_title", "")
        header = f"[Фрагмент {i}: {section}]" if section else f"[Фрагмент {i}]"
        parts.append(f"{header}\n{r['text']}")

    return "\n\n---\n\n".join(parts)
