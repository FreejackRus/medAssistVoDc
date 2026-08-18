from __future__ import annotations

import json
import logging
import re

from src.llm.client import OllamaClient
from src.llm.prompts import SERVICE_MATCH_PROMPT
from src.services.catalog import get_top_services

log = logging.getLogger(__name__)

_llm: OllamaClient | None = None


def _get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        _llm = OllamaClient()
    return _llm


def match_services(step_text: str, step_title: str = "") -> list[dict]:
    """TF-IDF pre-filter + LLM selection of relevant services."""
    relevant = get_top_services(step_text)
    if relevant.empty:
        return []

    services_text = "\n".join(f"{row['ID']} - {row['Название']}" for _, row in relevant.iterrows())

    prompt = SERVICE_MATCH_PROMPT.format(
        services_text=services_text,
        step_title=step_title,
        step_text=step_text[:5000],
    )

    raw = _get_llm().chat([{"role": "system", "content": prompt}], temperature=0.0, num_predict=200)

    return _parse_response(raw)


def _parse_response(raw: str) -> list[dict]:
    """Extract JSON array of services from LLM response."""
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        match = re.search(r"\[.*", raw, re.DOTALL)
        if match:
            json_str = match.group(0)
            last = json_str.rfind("}")
            json_str = json_str[: last + 1] + "]" if last > 0 else "[]"
        else:
            return []
    else:
        json_str = match.group(0)

    try:
        items = json.loads(json_str)
        return [{"id": str(s.get("id", "")), "name": s.get("name", "")} for s in items]
    except json.JSONDecodeError:
        log.warning("Failed to parse LLM response as JSON")
        return []
