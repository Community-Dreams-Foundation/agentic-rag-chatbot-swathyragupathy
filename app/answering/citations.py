"""Build citation list [{source, locator, snippet}] from retrieved chunks and answer text."""
import re
from typing import Any, Dict, List


def build_citations_from_chunks(
    chunks: List[Dict[str, Any]],
    answer_text: str,
) -> List[Dict[str, str]]:
    """
    Produce citations in order of first appearance in the answer. Each citation has source, locator, snippet.
    Snippet: up to 120 chars from the chunk content (or from answer if we detect a quoted part).
    """
    cited_indices = _extract_cited_indices(answer_text)
    if not cited_indices:
        # Cite all chunks used as context (in order)
        cited_indices = list(range(1, len(chunks) + 1))
    seen = set()
    out = []
    for idx in cited_indices:
        if idx in seen or idx < 1 or idx > len(chunks):
            continue
        seen.add(idx)
        c = chunks[idx - 1]
        snippet = _snippet_from_chunk(c["content"])
        out.append({
            "source": c["source"],
            "locator": c["locator"],
            "snippet": snippet,
        })
    if not out and chunks:
        # Fallback: one citation from first chunk
        c = chunks[0]
        out.append({
            "source": c["source"],
            "locator": c["locator"],
            "snippet": _snippet_from_chunk(c["content"]),
        })
    return out


def _extract_cited_indices(text: str) -> List[int]:
    order = []
    seen = set()
    for m in re.finditer(r"\[(\d+)\]", text):
        idx = int(m.group(1))
        if 1 <= idx <= 20 and idx not in seen:
            order.append(idx)
            seen.add(idx)
    return order


def _snippet_from_chunk(content: str, max_len: int = 120) -> str:
    s = (content or "").strip().replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."
