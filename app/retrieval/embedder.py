"""Singleton embedder (sentence-transformers) to avoid loading model multiple times."""
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
