"""
Centralized configuration for the RAG pipeline.

All tunable knobs live here so experiments (chunk size, retrieval weights,
top-k, etc.) are changed in one place rather than scattered across modules.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_PATH = DATA_DIR / "techcorp_docs.json"
EVAL_QUERIES_PATH = DATA_DIR / "eval_queries.json"
INDEX_CACHE_PATH = DATA_DIR / "index_cache.pkl"

# --- Chunking ------------------------------------------------------------
# "fixed" splits on a rolling word window with overlap.
# "sentence" groups whole sentences up to a max size (better semantic boundaries).
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "sentence")  # "fixed" | "sentence"
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", 90))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 20))

# --- Retrieval -----------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 5))
# Weight given to dense (TF-IDF/embedding) vs sparse (BM25) scores when NOT
# using Reciprocal Rank Fusion. Only used by the "weighted" fusion strategy.
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.5))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.5))
# "rrf" (reciprocal rank fusion, recommended, scale-free) | "weighted"
FUSION_STRATEGY = os.getenv("FUSION_STRATEGY", "rrf")
RRF_K = int(os.getenv("RRF_K", 60))  # standard RRF damping constant

# --- Generation ------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-sonnet-4-5")
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", 5))

# --- Embeddings ------------------------------------------------------------
# If sentence-transformers is installed AND USE_SENTENCE_EMBEDDINGS=true,
# use real dense embeddings. Otherwise fall back to TF-IDF, which needs no
# model download and runs fully offline -- useful for CI / restricted networks.
USE_SENTENCE_EMBEDDINGS = os.getenv("USE_SENTENCE_EMBEDDINGS", "false").lower() == "true"
SENTENCE_EMBEDDING_MODEL = os.getenv("SENTENCE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
