from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException

from src.external.minzdrav import fetch_recommendations

log = logging.getLogger(__name__)
router = APIRouter(prefix="/clinical-recommendations", tags=["clinical-recommendations"])

_cache: dict = {"data": None, "fetched_at": 0.0}
_cache_lock = asyncio.Lock()
CACHE_TTL = 600  # 10 minutes


def _fresh(now: float) -> bool:
    return _cache["data"] is not None and (now - _cache["fetched_at"]) < CACHE_TTL


@router.get("/")
async def get_clinical_recommendations():
    now = time.time()

    if _fresh(now):
        recs = _cache["data"]
        return {"success": True, "total": len(recs), "recommendations": recs}

    async with _cache_lock:
        now = time.time()
        if _fresh(now):
            recs = _cache["data"]
            return {"success": True, "total": len(recs), "recommendations": recs}

        try:
            recs = await fetch_recommendations()
            _cache["data"] = recs
            _cache["fetched_at"] = now
            return {"success": True, "total": len(recs), "recommendations": recs}
        except Exception as e:
            if _cache["data"] is not None:
                log.warning("Minzdrav API error, serving stale cache: %s", e)
                recs = _cache["data"]
                return {"success": True, "total": len(recs), "recommendations": recs}
            log.error("Minzdrav API error: %s", e)
            raise HTTPException(502, f"Error fetching recommendations: {e}")
