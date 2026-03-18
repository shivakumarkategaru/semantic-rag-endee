"""
config.py
----------
Central configuration for the Semantic RAG system.
All runtime settings are defined here and can be overridden via
environment variables or by editing this file directly.
"""

import os

# ─── Endee Vector Database ────────────────────────────────────────────
ENDEE_HOST: str = os.environ.get("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN: str = os.environ.get("ENDEE_AUTH_TOKEN", "")
ENDEE_INDEX_NAME: str = os.environ.get("ENDEE_INDEX_NAME", "rag_knowledge_base")
ENDEE_SPACE_TYPE: str = "cosine"           # cosine | l2 | dot

# ─── Embedding Model ──────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384                   # must match model output
EMBEDDING_BATCH_SIZE: int = 64
EMBEDDING_DEVICE: str = "cpu"             # cpu | cuda | mps

# ─── Document Processing ──────────────────────────────────────────────
CHUNK_SIZE: int = 512                      # characters per chunk
CHUNK_OVERLAP: int = 64                    # overlap between chunks

# ─── Retrieval ────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 5                     # results to retrieve
SEARCH_EF: int = 128                       # HNSW ef parameter

# ─── Generator (optional LLM step) ───────────────────────────────────
GENERATOR_MODEL: str = os.environ.get("GENERATOR_MODEL", "distilgpt2")
USE_LLM: bool = os.environ.get("USE_LLM", "false").lower() == "true"

# ─── Logging ──────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
