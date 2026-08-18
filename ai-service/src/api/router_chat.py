from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatAttachmentIngestResponse, ChatRequest
from src.api.router_algorithms import _stream_error_event, _stream_from_sync
from src.config import settings
from src.pdf.extractor import extract_text
from src.rag.chunker import chunk_sections
from src.rag.pipeline import stream_rag_answer
from src.rag.vector_store import add_chunks, delete_document_chunks

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/attachments/ingest", response_model=ChatAttachmentIngestResponse)
def ingest_attachment(
    pdf: UploadFile = File(...),
    document_id: str = Form(...),
):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    if not document_id:
        raise HTTPException(400, "document_id is required")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}.pdf"

    with dest.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    try:
        full_text = extract_text(str(dest))
        if not full_text or len(full_text.strip()) < 30:
            raise HTTPException(422, "Could not extract sufficient text from PDF")
        delete_document_chunks(document_id)
        chunks = chunk_sections({"Материал пользователя": full_text}, document_id=document_id)
        chunk_count = add_chunks(chunks, document_id=document_id)
        return ChatAttachmentIngestResponse(
            document_id=document_id,
            filename=pdf.filename,
            full_text=full_text,
            chunk_count=chunk_count,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Chat attachment extraction failed")
        raise HTTPException(500, f"Attachment extraction failed: {exc}") from exc
    finally:
        dest.unlink(missing_ok=True)


@router.post("/completions")
async def chat_completion(body: ChatRequest):
    """Stateless RAG chat: accept question + document_id + context, stream SSE answer."""
    history = [msg.model_dump() for msg in body.history]

    async def event_stream():
        try:
            async for token in _stream_from_sync(
                stream_rag_answer,
                body.question,
                body.document_id,
                body.diagnosis_name,
                history,
                [attachment.model_dump() for attachment in body.attachments],
            ):
                yield f"data: {json.dumps(token)}\n\n"
        except Exception:
            log.exception("Chat streaming failed")
            yield _stream_error_event("AI-сервис не смог получить ответ от модели")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
