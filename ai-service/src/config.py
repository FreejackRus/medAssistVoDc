from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MED_", env_file=".env", extra="ignore")

    # Ollama
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"

    # Database
    database_url: str = f"sqlite:///{BASE_DIR / 'storage' / 'medassistant.db'}"

    # Storage paths
    chroma_dir: str = str(BASE_DIR / "storage" / "chroma")
    upload_dir: str = str(BASE_DIR / "storage" / "uploads")
    export_dir: str = str(BASE_DIR / "storage" / "exports")

    # Data files
    services_file: str = str(BASE_DIR / "data" / "services.xlsx")

    # LLM parameters
    max_context_tokens: int = 32768
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 8
    algorithm_context_char_cap: int = 50000

    # Font paths to try for PDF export (cross-platform)
    font_paths: list[str] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/run/current-system/sw/share/fonts/truetype/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]


settings = Settings()
