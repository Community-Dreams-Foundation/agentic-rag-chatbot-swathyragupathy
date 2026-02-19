"""FAISS vector store: embed chunks, persist index and metadata to disk."""
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.config import FAISS_INDEX_DIR
from app.retrieval.embedder import get_embedder

INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"


class VectorStore:
    def __init__(self, reset: bool = False):
        self.embedder = get_embedder()
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._dim = None
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        idx_path = FAISS_INDEX_DIR / INDEX_FILE
        meta_path = FAISS_INDEX_DIR / META_FILE
        if reset and idx_path.exists():
            idx_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        if idx_path.exists() and meta_path.exists() and not reset:
            self._load(idx_path, meta_path)
        else:
            self._metadata = []

    def _load(self, idx_path: Path, meta_path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(idx_path))
        self._dim = self._index.d
        self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self._index is None:
            return
        import faiss
        idx_path = FAISS_INDEX_DIR / INDEX_FILE
        meta_path = FAISS_INDEX_DIR / META_FILE
        faiss.write_index(self._index, str(idx_path))
        meta_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=0), encoding="utf-8")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        import faiss
        texts = [c["content"] for c in chunks]
        emb = self.embedder.encode(texts)
        emb = np.asarray(emb, dtype=np.float32)
        if self._index is None:
            self._dim = emb.shape[1]
            self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(emb)
        for c in chunks:
            self._metadata.append({
                "source": c["source"],
                "locator": c["locator"],
                "chunk_id": c["chunk_id"],
                "content": c["content"],
            })
        self._save()

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._index is None or len(self._metadata) == 0:
            return []
        k = min(top_k, len(self._metadata))
        q_emb = self.embedder.encode([query])
        q_emb = np.asarray(q_emb, dtype=np.float32)
        _, indices = self._index.search(q_emb, k)
        out = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._metadata):
                continue
            m = self._metadata[idx]
            out.append({
                "source": m["source"],
                "locator": m["locator"],
                "chunk_id": m["chunk_id"],
                "content": m["content"],
            })
        return out

    @property
    def collection(self):
        """Fake collection for compatibility with main.py startup count check."""
        class _Fake:
            def count(self):
                return len(self._parent._metadata)
            _parent = None
        f = _Fake()
        f._parent = self
        return f