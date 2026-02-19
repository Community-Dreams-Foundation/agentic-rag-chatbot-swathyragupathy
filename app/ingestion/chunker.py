"""Chunk text with overlap and attach metadata (source, locator, chunk_id)."""
import re
from pathlib import Path
from typing import Any, Dict, List

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def _rough_token_count(text: str) -> int:
    """Approximate token count (words + punctuation)."""
    return len(re.findall(r"\S+", text))


def chunk_document(
    path: Path,
    segments: List[tuple],
) -> List[Dict[str, Any]]:
    """
    segments: list of (text, locator) from parser.
    Returns list of dicts: content, source (filename), locator, chunk_id.
    """
    source_name = Path(path).name
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for text, locator in segments:
        if not text.strip():
            continue
        tokens = _rough_token_count(text)
        if tokens <= CHUNK_SIZE:
            chunk_id += 1
            chunks.append({
                "content": text.strip(),
                "source": source_name,
                "locator": locator,
                "chunk_id": str(chunk_id),
            })
            continue

        start = 0
        words = text.split()
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            window = " ".join(words[start:end])
            overlap_start = max(0, end - CHUNK_OVERLAP)
            chunk_id += 1
            chunks.append({
                "content": window,
                "source": source_name,
                "locator": locator,
                "chunk_id": str(chunk_id),
            })
            start = overlap_start + CHUNK_SIZE if overlap_start + CHUNK_SIZE < end else end
            if start >= len(words):
                break

    return chunks
