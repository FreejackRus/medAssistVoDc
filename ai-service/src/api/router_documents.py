from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import IngestResponse
from src.config import settings
from src.pdf.extractor import extract_text, extract_first_page
from src.pdf.parser import parse_sections, extract_diagnosis
from src.rag.chunker import chunk_sections
from src.rag.vector_store import add_chunks, delete_document_chunks

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    pdf: UploadFile = File(...),
    document_id: str = Form(...),
):
    """Process a PDF: extract text, parse, chunk, embed. No local DB — returns results to caller."""
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}.pdf"
    dest = upload_dir / safe_name

    with dest.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    try:
        full_text = extract_text(str(dest))
        if not full_text or len(full_text.strip()) < 100:
            raise HTTPException(422, "Could not extract sufficient text from PDF")

        first_page = extract_first_page(str(dest))
        diagnosis_name, mkb_code = extract_diagnosis(first_page, full_text, filename=pdf.filename)
        sections = parse_sections(full_text)
        delete_document_chunks(document_id)
        chunks = chunk_sections(sections, document_id=document_id)
        chunk_count = add_chunks(chunks, document_id=document_id)

        sections_json = json.dumps(sections, ensure_ascii=False)

        log.info("Document %s ingested: %d chunks", document_id, chunk_count)

        return IngestResponse(
            document_id=document_id,
            diagnosis_name=diagnosis_name,
            mkb_code=mkb_code,
            chunk_count=chunk_count,
            full_text=full_text,
            sections_json=sections_json,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Ingestion error for document %s: %s", document_id, e)
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        # Clean up temp PDF — Rust gateway keeps its own copy
        dest.unlink(missing_ok=True)


@router.delete("/{document_id}")
def delete_document(document_id: str):
    """Delete document chunks from ChromaDB vector store."""
    delete_document_chunks(document_id)
    return {"ok": True}
