"""FastAPI app: upload, query, memory, weather (Feature C), streaming, conversations, files."""
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import OPENAI_API_KEY, REPO_ROOT, SAMPLE_DOCS_DIR
from app.pipeline import ingest_directory, query, query_stream, process_user_message_for_memory
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


# In-memory conversation store: session_id -> list of {id, title, messages}
_conversations: dict[str, list[dict]] = {}


def _get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    return (x_session_id or "").strip() or str(uuid.uuid4())


@app.post("/query")
def query_endpoint(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    x_session_id: Optional[str] = Header(None),
):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    sid = session_id or _get_session_id(x_session_id)
    store = get_store()
    answer, citations = query(store, question)
    # Persist to conversation history
    convs = _conversations.setdefault(sid, [])
    if conversation_id:
        conv = next((c for c in convs if c["id"] == conversation_id), None)
    else:
        conv = {"id": str(uuid.uuid4()), "title": (question[:50] + "..." if len(question) > 50 else question), "messages": []}
        convs.append(conv)
    conv["messages"].append({"role": "user", "content": question})
    conv["messages"].append({"role": "assistant", "content": answer, "citations": citations})
    return {"answer": answer, "citations": citations, "conversation_id": conv["id"], "session_id": sid}


def _is_weather_question(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(
        w in q for w in ("weather", "temperature", "forecast", "rain", "sunny", "humidity", "how hot", "how cold")
    )


def _stream_query_events(question: str, session_id: str, conversation_id: Optional[str]):
    store = get_store()
    final_answer = None
    final_citations = None

    # Weather-like questions use full query (with tool) and return a single event; no token streaming
    if _is_weather_question(question):
        final_answer, final_citations = query(store, question)
        final_citations = final_citations or []
        convs = _conversations.setdefault(session_id, [])
        if conversation_id:
            conv = next((c for c in convs if c["id"] == conversation_id), None)
        else:
            conv = {"id": str(uuid.uuid4()), "title": (question[:50] + "..." if len(question) > 50 else question), "messages": []}
            convs.append(conv)
        conv["messages"].append({"role": "user", "content": question})
        conv["messages"].append({"role": "assistant", "content": final_answer, "citations": final_citations})
        yield f"data: {json.dumps({'answer': final_answer, 'citations': final_citations, 'conversation_id': conv['id'], 'session_id': session_id})}\n\n"
        return

    for kind, val, cites in query_stream(store, question):
        if kind == "delta":
            yield f"data: {json.dumps({'t': val})}\n\n"
        else:
            final_answer = val
            final_citations = cites or []
            convs = _conversations.setdefault(session_id, [])
            if conversation_id:
                conv = next((c for c in convs if c["id"] == conversation_id), None)
            else:
                conv = {"id": str(uuid.uuid4()), "title": (question[:50] + "..." if len(question) > 50 else question), "messages": []}
                convs.append(conv)
            conv["messages"].append({"role": "user", "content": question})
            conv["messages"].append({"role": "assistant", "content": final_answer, "citations": final_citations or []})
            yield f"data: {json.dumps({'answer': final_answer, 'citations': final_citations, 'conversation_id': conv['id'], 'session_id': session_id})}\n\n"


@app.post("/query/stream")
def query_stream_endpoint(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    x_session_id: Optional[str] = Header(None),
):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    sid = session_id or _get_session_id(x_session_id)
    return StreamingResponse(
        _stream_query_events(question, sid, conversation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


# ---- Conversations (multi-user by session_id) ----
@app.get("/conversations")
def list_conversations(
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None),
):
    sid = session_id or _get_session_id(x_session_id)
    convs = _conversations.get(sid, [])
    return {"conversations": [{"id": c["id"], "title": c["title"], "message_count": len(c["messages"])} for c in convs]}


@app.get("/conversations/{conv_id}")
def get_conversation(
    conv_id: str,
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None),
):
    sid = session_id or _get_session_id(x_session_id)
    convs = _conversations.get(sid, [])
    conv = next((c for c in convs if c["id"] == conv_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conv["id"], "title": conv["title"], "messages": conv["messages"]}


# ---- File management: list, delete, reindex, inspect chunks ----
def _sanitize_filename(name: str) -> str:
    return Path(name).name if name else ""


@app.get("/files")
def list_files():
    """List indexed sources (filenames) with chunk counts."""
    store = get_store()
    sources = store.list_sources()
    out = [{"filename": s, "chunk_count": len(store.get_chunks_by_source(s))} for s in sources]
    return {"files": out}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Remove file from index (and optionally from uploads)."""
    name = _sanitize_filename(filename)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    store = get_store()
    removed = store.delete_by_source(name)
    upload_path = REPO_ROOT / "uploads" / name
    if upload_path.exists():
        upload_path.unlink(missing_ok=True)
    return {"removed": removed, "filename": name}


@app.get("/files/{filename}/chunks")
def get_file_chunks(filename: str):
    """Inspect chunks for a source."""
    name = _sanitize_filename(filename)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    store = get_store()
    chunks = store.get_chunks_by_source(name)
    return {"filename": name, "chunks": [{"chunk_id": c["chunk_id"], "locator": c["locator"], "content_preview": (c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"])} for c in chunks]}


@app.post("/files/{filename}/reindex")
def reindex_file(filename: str):
    """Re-parse and re-index a file (replace existing chunks for that source)."""
    name = _sanitize_filename(filename)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = REPO_ROOT / "uploads" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found in uploads")
    store = get_store()
    store.delete_by_source(name)
    from app.ingestion import parse_file, chunk_document
    segments = parse_file(path)
    chunks = chunk_document(path, segments)
    if chunks:
        store.add_chunks(chunks)
    return {"filename": name, "chunks_indexed": len(chunks) if chunks else 0}


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
