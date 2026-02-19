# Architecture Overview

## Goal
This chatbot implements file-grounded Q&A (RAG) with citations, selective persistent memory, and optional weather time-series analysis (Feature C). It uses OpenAI for generation, local sentence-transformers + FAISS for embeddings and retrieval, and Open-Meteo for weather data.

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)
- **Supported inputs:** `.txt`, `.pdf`.
- **Parsing:** TXT read as plain text; PDF via PyMuPDF (or PyPDF2 fallback) with per-page text and page numbers.
- **Chunking:** Fixed-size chunks (~600 tokens) with overlap (~100 tokens). Metadata per chunk: `source` (filename), `locator` (e.g. "page 3" or "document"), `chunk_id`.

### 2) Indexing / Storage
- **Vector store:** FAISS (index and metadata persisted under `faiss_index/`).
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (local, no API key).
- **Persistence:** Chroma stores embeddings and metadata; no separate BM25 index.

### 3) Retrieval + Grounded Answering
- **Retrieval:** Top-k vector search (k=6) over chunk embeddings.
- **Citations:** Each citation has `source`, `locator`, and `snippet` (short excerpt from the chunk). Built from chunks cited in the model answer (e.g. [1], [2]) or from chunk content.
- **Failure behavior:** If no chunks are retrieved or context is empty, the assistant responds that it couldn't find the answer in the uploaded documents and returns no citations (no hallucination).

### 4) Memory System (Selective)
- **High-signal:** User-specific facts (role, preferences) → `USER_MEMORY.md`; org-wide, reusable learnings → `COMPANY_MEMORY.md`.
- **Not stored:** Raw transcript, PII, secrets, greetings, or vague statements.
- **When we write:** After user messages, an OpenAI call classifies whether to store and as USER vs COMPANY; we only append when the model returns a concise summary. Deduplication by not re-appending the same summary line.
- **Format:** Markdown files with bullet items (`- summary text`).

### 5) Safe Tooling (Open-Meteo) — Feature C
- **Tool:** `get_weather_analysis(location, start_date?, end_date?)` — geocode via Open-Meteo Geocoding API, fetch hourly forecast via Open-Meteo Forecast API (no key), run analytics in a sandbox.
- **Sandbox:** Analytics (rolling mean, volatility, missingness, anomaly flags) run in a subprocess with a strict timeout (~15 s); the child script reads JSON from stdin and prints JSON to stdout — no network, no file access.
- **Integration:** The LLM is given the tool definition; when the user asks about weather or temperature time series, the model can call the tool; we execute it and append the result, then the model summarizes for the user. Direct API: `GET /weather?location=London`.
- **Safety:** Timeout, no arbitrary code execution (only our fixed analytics script), network only to Open-Meteo from the main process.

---

## Tradeoffs & Next Steps
- **Why this design:** FAISS + sentence-transformers keeps setup simple and avoids extra API keys; OpenAI gives strong answers and memory classification; Open-Meteo requires no key. Single top-k retrieval is fast; hybrid retrieval or reranking could improve quality.
- **Improvements with more time:** Hybrid (BM25 + vector) retrieval, reranker, section-aware chunking for PDFs, streaming responses, and evaluation harness.
