from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from src.api.schemas import AlgorithmGenerate, ExportPdfRequest
from src.rag.pipeline import stream_algorithm

log = logging.getLogger(__name__)
router = APIRouter(prefix="/algorithms", tags=["algorithms"])


async def _stream_from_sync(gen_func, *args) -> AsyncIterator[str]:
    """Run a synchronous generator in a thread, yielding tokens as they arrive."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()

    def _produce():
        try:
            for token in gen_func(*args):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    fut = loop.run_in_executor(None, _produce)

    while True:
        token = await queue.get()
        if token is None:
            break
        if isinstance(token, Exception):
            raise token
        yield token

    await fut


def _stream_error_event(message: str) -> str:
    return f"event: error\ndata: {json.dumps(message)}\n\n"


@router.post("/generate")
async def generate_algorithm(body: AlgorithmGenerate):
    """Stream algorithm generation from full document text. Stateless — no DB access."""

    async def event_stream():
        try:
            async for token in _stream_from_sync(
                stream_algorithm,
                body.full_text,
                body.diagnosis_name,
                body.mode,
            ):
                yield f"data: {json.dumps(token)}\n\n"
        except Exception:
            log.exception("Algorithm streaming failed")
            yield _stream_error_event("AI-сервис не смог получить ответ от модели")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/export-pdf")
async def export_pdf(body: ExportPdfRequest):
    import os
    from src.pdf.exporter import algorithm_pdf_filename, markdown_to_pdf
    from starlette.background import BackgroundTask
    from urllib.parse import quote

    if not body.markdown or len(body.markdown.strip()) < 100:
        raise HTTPException(400, "Not enough content for PDF generation")

    pdf_path = markdown_to_pdf(body.markdown)
    filename = algorithm_pdf_filename(body.markdown)
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
        background=BackgroundTask(os.unlink, pdf_path),
    )
