"""Parse supported file types to plain text with optional page/section metadata."""
from pathlib import Path
from typing import List, Tuple


def parse_file(path: Path) -> List[Tuple[str, str]]:
    """
    Parse a file and return a list of (text, locator) where locator is e.g. "page 1" or "section X".
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _parse_txt(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    # Fallback: read as text
    return _parse_txt(path)


def _parse_txt(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [(text, "document")]


def _parse_pdf(path: Path) -> List[Tuple[str, str]]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _parse_pdf_fallback(path)

    doc = fitz.open(path)
    out: List[Tuple[str, str]] = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text().strip()
        if text:
            out.append((text, f"page {i + 1}"))
    doc.close()
    return out


def _parse_pdf_fallback(path: Path) -> List[Tuple[str, str]]:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        out = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                out.append((text.strip(), f"page {i + 1}"))
        return out
    except Exception:
        return []
