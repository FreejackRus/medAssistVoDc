from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas import ServiceMatchRequest, ServiceMatchResponse, ServiceItem
from src.services.matcher import match_services

log = logging.getLogger(__name__)
router = APIRouter(prefix="/services", tags=["services"])


@router.post("/match", response_model=ServiceMatchResponse)
async def match(body: ServiceMatchRequest):
    loop = asyncio.get_running_loop()
    services = await loop.run_in_executor(None, match_services, body.step_text, body.step_title)
    return ServiceMatchResponse(
        services=[ServiceItem(**s) for s in services],
        step_title=body.step_title,
    )
