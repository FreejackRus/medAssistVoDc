from __future__ import annotations

import logging
import fitz  # PyMuPDF

log = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        parts: list[str] = []
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")
                if len(text.strip()) < 100:
                    log.warning("Page %d: very little text (%d chars)", page_num + 1, len(text.strip()))
                parts.append(text)
            except Exception as e:
                log.error("Error processing page %d: %s", page_num + 1, e)
        return "\n".join(parts)


def extract_first_page(pdf_path: str) -> str:
    """Extract text from the first page only."""
    with fitz.open(pdf_path) as doc:
        if len(doc) > 0:
            return doc[0].get_text("text")
        return ""


def extract_text_by_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from each page, returning (page_number, text) pairs."""
    with fitz.open(pdf_path) as doc:
        pages = []
        for page_num, page in enumerate(doc):
            try:
                pages.append((page_num + 1, page.get_text("text")))
            except Exception as e:
                log.error("Error processing page %d: %s", page_num + 1, e)
        return pages
