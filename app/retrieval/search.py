"""Retrieve top-k chunks for a query. Filters by relevance to avoid hallucinations."""
from typing import Any, Dict, List, Optional

from app.config import TOP_K_RETRIEVAL, RELEVANCE_MAX_L2
from app.retrieval.store import VectorStore


def retrieve(store: VectorStore, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return chunks whose best L2 distance is within RELEVANCE_MAX_L2; otherwise empty (graceful refusal)."""
    k = top_k or TOP_K_RETRIEVAL
    chunks, distances = store.search(query, top_k=k, return_distances=True)
    if not chunks or not distances:
        return []
    # If even the nearest chunk is too far, treat as no relevant context (no hallucination)
    if min(distances) > RELEVANCE_MAX_L2:
        return []
    return chunks
