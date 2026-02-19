"""Retrieve top-k chunks for a query."""
from typing import Any, Dict, List, Optional

from app.config import TOP_K_RETRIEVAL
from app.retrieval.store import VectorStore


def retrieve(store: VectorStore, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    k = top_k or TOP_K_RETRIEVAL
    return store.search(query, top_k=k)
