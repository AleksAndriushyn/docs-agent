import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen3.6")
LOCAL_URL = os.getenv("LOCAL_URL", "http://localhost:11434/v1")
# Qwen3 thinking models loop on low temperature; 0.6 is the vendor recommendation
LOCAL_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0.6"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
QDRANT_URL = os.getenv("QDRANT_URL")  # set to use a Qdrant server instead of the embedded store
QDRANT_PATH = os.getenv("QDRANT_PATH", str(PROJECT_DIR / "qdrant_data"))
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "docs")
# cross-encoder for second-stage ranking; empty string disables reranking
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "16"))  # latency grows linearly

SANDBOX_DIR = PROJECT_DIR / "sandbox"

MAX_STEPS = 15
MAX_FILE_CHARS = 20_000
MAX_HITS = 50
