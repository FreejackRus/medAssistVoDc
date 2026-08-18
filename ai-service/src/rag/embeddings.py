from __future__ import annotations

import logging
import time

import ollama as _ollama
from src.config import settings

_EMBED_BATCH_SIZE = 64
_EMBED_RETRIES = 2
_EMBED_RETRY_DELAY_SECONDS = 0.7

log = logging.getLogger(__name__)


class OllamaEmbeddings:
    """Thin wrapper around Ollama embedding models."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.embed_model
        self._client = _ollama.Client(host=base_url or settings.ollama_base_url)

    def _document_input(self, text: str) -> str:
        if self.model.startswith("nomic-embed-text"):
            return f"search_document: {text}"
        return text

    def _query_input(self, text: str) -> str:
        if self.model.startswith("nomic-embed-text"):
            return f"search_query: {text}"
        return text

    def _embed_with_retry(self, payload: str | list[str]):
        last_error: Exception | None = None
        for attempt in range(_EMBED_RETRIES + 1):
            try:
                return self._client.embed(model=self.model, input=payload)
            except _ollama.ResponseError as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and status_code < 500:
                    raise
                if attempt >= _EMBED_RETRIES:
                    break
                log.warning(
                    "Ollama embedding request failed; retrying",
                    extra={"attempt": attempt + 1, "model": self.model, "status_code": status_code},
                )
                time.sleep(_EMBED_RETRY_DELAY_SECONDS * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama embedding request failed")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (batched to avoid overloading the API)."""
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = [
                self._document_input(text)
                for text in texts[i : i + _EMBED_BATCH_SIZE]
            ]
            resp = self._embed_with_retry(batch)
            results.extend(resp["embeddings"])
        return results

    def embed_query(self, text: str) -> list[float]:
        resp = self._embed_with_retry(self._query_input(text))
        return resp["embeddings"][0]
