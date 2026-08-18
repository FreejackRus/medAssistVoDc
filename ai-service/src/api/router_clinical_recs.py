from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Query

from src.external.minzdrav import fetch_recommendations

log = logging.getLogger(__name__)
router = APIRouter(prefix="/clinical-recommendations", tags=["clinical-recommendations"])

_cache: dict = {"data": None, "fetched_at": 0.0}
_cache_lock = asyncio.Lock()
CACHE_TTL = 600  # 10 minutes


def _fresh(now: float) -> bool:
    return _cache["data"] is not None and (now - _cache["fetched_at"]) < CACHE_TTL


def _filter_recommendations(recs: list[dict], query: str) -> list[dict]:
    normalized = query.strip().casefold()
    if not normalized:
        return recs
    searchable_fields = ("title", "mkb_code", "keywords")
    return [
        rec
        for rec in recs
        if any(normalized in str(rec.get(field, "")).casefold() for field in searchable_fields)
    ]


def _paginate_recommendations(
    recs: list[dict],
    *,
    query: str,
    page: int,
    page_size: int,
) -> dict:
    filtered = _filter_recommendations(recs, query)
    total = len(filtered)
    start = (page - 1) * page_size
    return {
        "success": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "recommendations": filtered[start : start + page_size],
    }


@router.get("/")
async def get_clinical_recommendations(
    q: str = Query(default="", max_length=200),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    now = time.time()

    if _fresh(now):
        recs = _cache["data"]
        return _paginate_recommendations(recs, query=q, page=page, page_size=page_size)

    async with _cache_lock:
        now = time.time()
        if _fresh(now):
            recs = _cache["data"]
            return _paginate_recommendations(recs, query=q, page=page, page_size=page_size)

        try:
            recs = await fetch_recommendations()
            _cache["data"] = recs
            _cache["fetched_at"] = now
            return _paginate_recommendations(recs, query=q, page=page, page_size=page_size)
        except Exception as e:
            if _cache["data"] is not None:
                log.warning("Minzdrav API error, serving stale cache: %s", e)
                recs = _cache["data"]
                return _paginate_recommendations(recs, query=q, page=page, page_size=page_size)
            log.error("Minzdrav API error: %s", e)
            raise HTTPException(502, f"Error fetching recommendations: {e}")
