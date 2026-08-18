from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title="Medical Assistant AI Service", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for d in (settings.upload_dir, settings.export_dir, settings.chroma_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    from src.api.router_documents import router as documents_router
    from src.api.router_algorithms import router as algorithms_router
    from src.api.router_chat import router as chat_router
    from src.api.router_services import router as services_router
    from src.api.router_clinical_recs import router as clinical_recs_router

    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(algorithms_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(services_router, prefix="/api/v1")
    app.include_router(clinical_recs_router, prefix="/api/v1")

    @app.get("/api/v1/health")
    async def health():
        import asyncio
        from src.llm.client import OllamaClient
        client = OllamaClient()
        ollama_ok = await asyncio.to_thread(client.ping)
        return {
            "status": "ok",
            "ollama": "ok" if ollama_ok else "unavailable",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)
