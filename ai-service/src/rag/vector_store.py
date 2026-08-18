from __future__ import annotations

import logging
import threading

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.rag.embeddings import OllamaEmbeddings

log = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_embedder: OllamaEmbeddings | None = None
_lock = threading.Lock()
COLLECTION_NAME = "medical_docs"


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = chromadb.PersistentClient(
                    path=settings.chroma_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
    return _client


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = OllamaEmbeddings()
    return _embedder


def get_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunks: list,  # list[Chunk]
    document_id: str,
) -> int:
    """Embed and upsert chunks into ChromaDB. Returns count added."""
    if not chunks:
        return 0

    embedder = _get_embedder()
    collection = get_collection()

    ids = [f"doc{document_id}_chunk{c.metadata['chunk_index']}" for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Batch embed
    embeddings = embedder.embed(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    log.info("Added %d chunks for document %s", len(chunks), document_id)
    return len(chunks)


def query_chunks(
    query: str,
    document_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Query ChromaDB and return list of {text, metadata, distance}."""
    top_k = top_k or settings.top_k
    embedder = _get_embedder()
    collection = get_collection()

    query_embedding = embedder.embed_query(query)

    where = {"document_id": str(document_id)} if document_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    chunks = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "text": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })
    return chunks


def get_document_chunks(document_id: str) -> list[dict]:
    """Return all chunks for a document ordered by ingestion chunk index."""
    collection = get_collection()
    results = collection.get(
        where={"document_id": str(document_id)},
        include=["documents", "metadatas"],
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    chunks = []
    for doc, metadata in zip(documents, metadatas):
        chunks.append({
            "text": doc,
            "metadata": metadata or {},
            "distance": 0,
        })

    def sort_key(chunk: dict) -> int:
        raw_index = chunk["metadata"].get("chunk_index", 0)
        try:
            return int(raw_index)
        except (TypeError, ValueError):
            return 0

    return sorted(chunks, key=sort_key)


def delete_document_chunks(document_id: str) -> None:
    """Delete all chunks for a document from ChromaDB."""
    collection = get_collection()
    try:
        collection.delete(where={"document_id": document_id})
        log.info("Deleted chunks for document %s", document_id)
    except Exception:
        log.exception("Failed to delete chunks for document %s", document_id)
        raise
