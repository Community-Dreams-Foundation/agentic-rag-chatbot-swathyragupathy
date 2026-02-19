"""Load config from environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    OPENAI_API_KEY = None  # Callers can check and raise clear error

# Paths (relative to repo root when running from root)
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = REPO_ROOT / "sample_docs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
USER_MEMORY_PATH = REPO_ROOT / "USER_MEMORY.md"
COMPANY_MEMORY_PATH = REPO_ROOT / "COMPANY_MEMORY.md"
FAISS_INDEX_DIR = REPO_ROOT / "faiss_index"

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 6

# Model
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers, local

# Feature C: weather analytics sandbox
WEATHER_ANALYSIS_TIMEOUT_SEC = 15
