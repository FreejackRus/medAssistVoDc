from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings

log = logging.getLogger(__name__)

_df: pd.DataFrame | None = None
_vectorizer: TfidfVectorizer | None = None
_embeddings = None
_load_lock = threading.Lock()


def _load():
    global _df, _vectorizer, _embeddings

    path = Path(settings.services_file)
    if not path.exists():
        log.warning("Services file not found: %s", path)
        _df = pd.DataFrame(columns=["Название", "ID"])
        return

    df = pd.read_excel(path, sheet_name="iblock_element_admin (1)")
    df.columns = ["Название", "ID"]
    df = df.dropna(subset=["Название"])

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    embeddings = vectorizer.fit_transform(df["Название"].fillna("").tolist())

    _df = df
    _vectorizer = vectorizer
    _embeddings = embeddings
    log.info("Loaded %d services", len(df))


def get_services_df() -> pd.DataFrame:
    if _df is None:
        with _load_lock:
            if _df is None:
                _load()
    return _df  # type: ignore


def get_top_services(step_text: str, top_k: int = 50) -> pd.DataFrame:
    """Return top_k most relevant services by TF-IDF similarity."""
    df = get_services_df()
    if df.empty or _vectorizer is None or _embeddings is None:
        return df.head(top_k)

    query_vec = _vectorizer.transform([step_text])
    sims = cosine_similarity(query_vec, _embeddings).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return df.iloc[top_idx]
