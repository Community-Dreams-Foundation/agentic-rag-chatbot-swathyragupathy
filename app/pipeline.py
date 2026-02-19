"""End-to-end pipeline: ingest dir -> index -> query -> answer with citations; optional memory."""
from pathlib import Path
from typing import Any, Dict, List

from app.ingestion import parse_file, chunk_document
from app.retrieval import VectorStore, retrieve
from app.answering import answer_with_citations, answer_with_citations_and_tools
from app.memory import maybe_write_memory, append_memory


def ingest_directory(store: VectorStore, directory: Path) -> int:
    """Parse and chunk all .txt and .pdf in directory, add to store. Returns number of chunks added."""
    total = 0
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".txt", ".pdf"):
            continue
        segments = parse_file(path)
        chunks = chunk_document(path, segments)
        if chunks:
            store.add_chunks(chunks)
            total += len(chunks)
    return total


def query(store: VectorStore, question: str, use_weather_tool: bool = True) -> tuple[str, List[Dict[str, str]]]:
    """Retrieve chunks, generate answer with citations. Can use weather tool (Feature C) for weather questions. Returns (answer, citations)."""
    chunks = retrieve(store, question)
    return answer_with_citations_and_tools(question, chunks, use_tools=use_weather_tool)


def process_user_message_for_memory(user_message: str) -> List[Dict[str, str]]:
    """Decide what to store and return list of {target, summary}. Caller appends to files."""
    writes = maybe_write_memory(user_message)
    for w in writes:
        append_memory(w["target"], w["summary"])
    return writes
