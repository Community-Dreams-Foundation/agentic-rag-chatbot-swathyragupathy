"""FastAPI app: upload, query, memory, weather (Feature C)."""
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import OPENAI_API_KEY, REPO_ROOT, SAMPLE_DOCS_DIR
from app.pipeline import ingest_directory, query, process_user_message_for_memory
from app.retrieval import VectorStore

app = FastAPI(title="Agentic RAG Chatbot")

STATIC_DIR = REPO_ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)


def get_store() -> VectorStore:
    return VectorStore(reset=False)


@app.get("/", response_class=HTMLResponse)
def index():
    path = STATIC_DIR / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<h1>Agentic RAG Chatbot</h1><p>Add static/index.html for the UI.</p>"


@app.post("/upload")
def upload(files: list[UploadFile] = File(default=[])):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    if not files:
        return {"uploaded": []}
    store = get_store()
    saved = []
    for f in files:
        if not f.filename or f.filename.startswith("."):
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in (".txt", ".pdf"):
            continue
        path = REPO_ROOT / "uploads" / f.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f.file.read()
        path.write_bytes(content)
        from app.ingestion import parse_file, chunk_document
        segments = parse_file(path)
        chunks = chunk_document(path, segments)
        if chunks:
            store.add_chunks(chunks)
            saved.append(f.filename)
    return {"uploaded": saved}


@app.post("/query")
def query_endpoint(question: str = Form(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    store = get_store()
    answer, citations = query(store, question)
    return {"answer": answer, "citations": citations}


@app.post("/memory")
def memory_endpoint(user_message: str = Form(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    writes = process_user_message_for_memory(user_message)
    return {"memory_writes": writes}


@app.get("/weather")
def weather_endpoint(location: str = "", start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Feature C: Get weather time-series analysis for a location (Open-Meteo + sandboxed analytics)."""
    from app.weather.tool import get_weather_analysis
    if not location.strip():
        raise HTTPException(status_code=400, detail="Provide a location (e.g. ?location=London)")
    summary = get_weather_analysis(location=location.strip(), start_date=start_date or None, end_date=end_date or None)
    return {"location": location, "summary": summary}


# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    """Ingest sample_docs only if index is empty so the app works out of the box."""
    if not SAMPLE_DOCS_DIR.exists():
        return
    store = get_store()
    try:
        if store.collection.count() == 0:
            n = ingest_directory(store, SAMPLE_DOCS_DIR)
            if n:
                print(f"Ingested {n} chunks from sample_docs")
    except Exception as e:
        print(f"Sample docs ingest warning: {e}")
