# 🔍 Semantic RAG with Endee Vector Database

> A production-grade **Retrieval-Augmented Generation (RAG)** system built in Python, using [Endee](https://github.com/endee-io/endee) as the high-performance vector database backend — powered by **Groq + Llama 3.1** for free, fast AI-generated answers, with a professional **React frontend**.

---

> A production-grade **Retrieval-Augmented Generation (RAG)** system built in Python, using [Endee](https://github.com/endee-io/endee) as the high-performance vector database backend — powered by **Groq + Llama 3.1** for free, fast AI-generated answers, with a professional **React frontend**.

> 🚀 **Live Demo**: Run locally following the setup instructions below. Endee vector database runs via Docker and requires local deployment.

```

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Complete RAG Pipeline](#complete-rag-pipeline)
- [Architecture & System Design](#architecture--system-design)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Frontend](#frontend)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Module Reference](#module-reference)
- [License](#license)

---

## Project Overview

This project implements a complete **Semantic Search + RAG pipeline** with the following capabilities:

| Feature               | Detail                                                                     |
| --------------------- | -------------------------------------------------------------------------- |
| **Embeddings**        | `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, open-source)      |
| **Vector DB**         | [Endee](https://endee.io) — high-performance, HNSW-indexed, runs on Docker |
| **Chunking**          | Sentence-aware sliding window chunking with overlap                        |
| **Similarity Search** | Cosine similarity search with top-k retrieval and metadata filters         |
| **Context Injection** | Retrieved chunks passed as context to LLM for grounded answers             |
| **Answer Generation** | **Groq + Llama 3.1** (free) with full prompt engineering                   |
| **NLP Preprocessing** | Unicode normalization, URL stripping, sentence-aware chunking              |
| **Input Formats**     | Plain text strings, `.txt`, `.json`, `.csv` files                          |
| **Frontend**          | Professional React UI with chat history and source display                 |
| **Architecture**      | Fully modular — each concern lives in its own package                      |

All components run **100% locally and free** — Groq API has a generous free tier with no credit card required.

---

## Complete RAG Pipeline

### 📥 Data Ingestion Phase

```

Documents (.json / .txt / .csv)
│
▼
┌───────────────────────┐
│ NLP Preprocessing │ ← TextCleaner (normalize, clean)
│ src/data_processing/ │
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Chunking │ ← TextChunker (sentence-aware sliding window)
│ Split into chunks │ ← Each chunk ~200-300 tokens with overlap
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Embedding │ ← all-MiniLM-L6-v2 → 384-dim vectors
│ src/embeddings/ │
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Store in Endee │ ← index.upsert(vectors + metadata)
│ src/database/ │ ← HNSW index, cosine similarity
└───────────────────────┘

```

### 🔍 Query Phase

```

User Query
│
▼
┌───────────────────────┐
│ NLP Preprocessing │ ← Clean and normalize query
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Embedding │ ← Convert query → 384-dim vector
│ all-MiniLM-L6-v2 │
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Endee Search │ ← HNSW ANN search
│ top_k=3, filters │ ← Metadata filtering supported
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Retrieve Top Chunks │ ← Top 3 most relevant chunks
│ + similarity scores │ ← Filtered by min_score threshold
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Context Injection │ ← Retrieved chunks added to prompt
│ Prompt Engineering │ ← "Answer using this context: [chunks]"
└──────────┬────────────┘
│
▼
┌───────────────────────┐
│ Groq LLM │ ← Llama 3.1 8B Instant (FREE)
│ groq_generator.py │ ← Context + Question → Answer
└──────────┬────────────┘
│
▼
Final Answer
{ answer, sources, scores }

```

### 🔥 Simplified View

```

Query → Embed → Endee Search (top_k=3) → Add Context → Groq LLM → Answer

```

---

## Architecture & System Design

```

┌─────────────────────────────────────────────────────────────────┐
│ INGESTION PIPELINE │
│ │
│ Raw Text / File │
│ │ │
│ ▼ │
│ ┌─────────────┐ clean + chunk ┌──────────────────────┐ │
│ │ Document │ ─────────────────► │ DocumentProcessor │ │
│ │ Sources │ │ (NLP preprocessing) │ │
│ └─────────────┘ └──────────┬───────────┘ │
│ │ List[Chunk] │
│ ▼ │
│ ┌──────────────────────┐ │
│ │ EmbeddingEncoder │ │
│ │ all-MiniLM-L6-v2 │ │
│ │ 384-dim vectors │ │
│ └──────────┬───────────┘ │
│ │ List[vector] │
│ ▼ │
│ ┌──────────────────────┐ │
│ │ VectorStore │ │
│ │ (Endee Python SDK) │ │
│ │ index.upsert(...) │ │
│ └──────────┬───────────┘ │
│ ▼ │
│ ┌──────────────────────┐ │
│ │ Endee Server │ │
│ │ (Docker / HNSW) │ │
│ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ QUERY PIPELINE │
│ │
│ User Question │
│ │ │
│ ▼ │
│ ┌──────────────┐ encode ┌──────────────────────────────┐ │
│ │ RAGEngine │ ────────► │ EmbeddingEncoder │ │
│ │ (query) │ │ query vector [384-dim] │ │
│ └──────┬───────┘ └──────────────┬───────────────┘ │
│ │ │ │
│ │ ANN search (top_k=3) ▼ │
│ │ ◄───────────────── ┌──────────────────────────┐ │
│ │ top-k chunks │ Endee HNSW Index │ │
│ │ + scores │ cosine similarity │ │
│ │ └──────────────────────────┘ │
│ ▼ │
│ ┌──────────────┐ │
│ │Context Inject│ ── chunks added to prompt │
│ │Prompt Eng. │ ── "Answer using context: [chunk1][chunk2]" │
│ └──────┬───────┘ │
│ ▼ │
│ ┌──────────────┐ │
│ │ Groq LLM │ ── Llama 3.1 8B Instant (FREE) │
│ │ (Answer) │ │
│ └──────┬───────┘ │
│ ▼ │
│ RAGResponse │
│ { answer, sources, scores } │
└─────────────────────────────────────────────────────────────────┘

```

---

## Project Structure

```

semantic-rag-endee/
│
├── main.py # Entry point — demo + interactive loop
├── run.py # Quick start with Groq AI + chat history
├── api.py # Flask REST API for React frontend
├── docker-compose.yml # Endee server via Docker
├── requirements.txt # Python dependencies
├── setup.py # Package setup
├── pytest.ini # Test configuration
├── .env # Environment variables (GROQ_API_KEY)
├── .env.example # Environment variable template
│
├── src/
│ ├── config.py # All settings in one place
│ ├── pipeline.py # Top-level RAGPipeline orchestrator
│ │
│ ├── data_processing/
│ │ ├── **init**.py
│ │ └── processor.py # TextCleaner, TextChunker, DocumentProcessor
│ │
│ ├── embeddings/
│ │ ├── **init**.py
│ │ └── encoder.py # EmbeddingEncoder (sentence-transformers)
│ │
│ ├── database/
│ │ ├── **init**.py
│ │ └── vector_store.py # VectorStore — Endee SDK wrapper
│ │
│ └── query_handler/
│ ├── **init**.py
│ ├── rag_engine.py # RAGEngine + RAGResponse
│ ├── groq_generator.py # Groq + Llama 3.1 (FREE AI answers)
│ ├── gemini_generator.py # Google Gemini integration
│ └── openai_generator.py # OpenAI integration
│
├── data/
│ └── knowledge_base.json # Sample AI/tech knowledge base
│
├── frontend/ # Professional React UI
│ ├── src/
│ │ └── App.js # Main React component
│ └── package.json
│
├── scripts/
│ ├── ingest.py # Standalone ingestion script
│ └── query.py # Standalone query script
│
└── tests/
├── test_data_processing.py # Unit tests for processor
├── test_embeddings.py # Unit tests for encoder
└── test_pipeline.py # Integration tests (mock Endee)

````

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Docker 20.10+ and Docker Compose v2
- Node.js 16+ (for React frontend)
- 4 GB RAM

---

### 1. Start Endee with Docker

```bash
git clone https://github.com/<your-username>/semantic-rag-endee.git
cd semantic-rag-endee

# Start Endee server
docker compose up -d

# Verify it's running
docker ps
# → endee-server   Up X seconds   0.0.0.0:8080->8080/tcp

# Verify API is live
curl http://localhost:8080/api/v1/index/list
# → {"indexes":[]}
````

To stop:

```bash
docker compose down        # stop but keep data
docker compose down -v     # stop and wipe data
```

---

### 2. Python Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install groq flask flask-cors python-dotenv
```

---

### 3. Groq API Key (FREE)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up free → **API Keys** → **Create API Key**
3. Create `.env` file:

```
GROQ_API_KEY=gsk_your-key-here
```

---

## Usage Guide

### Terminal Mode (Quick Start)

```bash
python run.py
```

```
======================================================================
        🤖 AI Assistant powered by Groq + Llama 3.1
======================================================================
You: How do neural networks work?

──────────────────────────────────────────────────────────────────────
  🤖 Neural networks are computing systems inspired by the human brain.
     They consist of layers of interconnected nodes called neurons...
──────────────────────────────────────────────────────────────────────
```

### Full Demo

```bash
python main.py
python main.py --no-interactive   # demo only
python main.py --recreate-index   # rebuild index
python main.py --ingest-only      # load data only
```

### Ingest Documents

```bash
python scripts/ingest.py --file my_docs.json
python scripts/ingest.py --file notes.txt
python scripts/ingest.py --file articles.csv --recreate
```

### Python API

```python
from src.pipeline import RAGPipeline
from src.query_handler.groq_generator import GroqGenerator

# Initialize
pipeline = RAGPipeline()

# STEP 1: Ingest (Documents → Chunking → Embedding → Store in Endee)
pipeline.ingest_file("data/knowledge_base.json")

# STEP 2: Attach Groq LLM
pipeline.engine._openai_gen = GroqGenerator(model="llama-3.1-8b-instant")

# STEP 3: Query (Embed → Search → Context Inject → LLM → Answer)
response = pipeline.ask("How do neural networks work?")
print(response.display())

# Top-k semantic search with filters
results = pipeline.search("machine learning", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.text[:100]}")
```

**Sample output:**

```
──────────────────────────────────────────────────────────────────────
  QUESTION : How do neural networks work?
──────────────────────────────────────────────────────────────────────
  ANSWER   : Neural networks are computing systems inspired by the human
             brain. Each connection has a weight adjusted during training
             using backpropagation...
──────────────────────────────────────────────────────────────────────
  SOURCES  (3 chunks retrieved):
    [1] Score 0.851 ████████░░  [knowledge_base.json / chunk 0]
    [2] Score 0.613 ██████░░░░  [knowledge_base.json / chunk 1]
    [3] Score 0.564 █████░░░░░  [knowledge_base.json / chunk 0]
──────────────────────────────────────────────────────────────────────
```

---

## Frontend

Professional React UI featuring:

- 💬 Real-time chat interface with typing indicator
- 📜 Chat history sidebar
- 📄 Retrieved sources with similarity score bars
- 🔄 Pipeline steps shown after each answer
- 💡 Suggestion cards on welcome screen

### Run the Frontend

**Terminal 1 — Start Flask API:**

```bash
python api.py
# → Running on http://127.0.0.1:5000
```

**Terminal 2 — Start React:**

```bash
cd frontend
npm install
npm start
# → Open http://localhost:3000
```

---

## Configuration

| Variable           | Default                 | Description                             |
| ------------------ | ----------------------- | --------------------------------------- |
| `ENDEE_HOST`       | `http://localhost:8080` | Endee server URL                        |
| `ENDEE_INDEX_NAME` | `rag_knowledge_base`    | Endee index name                        |
| `EMBEDDING_MODEL`  | `all-MiniLM-L6-v2`      | HuggingFace model ID                    |
| `GROQ_API_KEY`     | ``                      | Groq API key (free at console.groq.com) |
| `GROQ_MODEL`       | `llama-3.1-8b-instant`  | Groq Llama model                        |
| `LOG_LEVEL`        | `INFO`                  | Logging verbosity                       |

---

## Running Tests

```bash
pip install pytest pytest-cov

pytest                                        # run all tests
pytest --cov=src --cov-report=term-missing    # with coverage
pytest tests/test_data_processing.py -v
pytest tests/test_embeddings.py -v
pytest tests/test_pipeline.py -v
```

---

## Module Reference

| Module                             | Key Class           | Responsibility                                                  |
| ---------------------------------- | ------------------- | --------------------------------------------------------------- |
| `src.data_processing.processor`    | `DocumentProcessor` | Clean, chunk, structure raw text into `Document` objects        |
| `src.embeddings.encoder`           | `EmbeddingEncoder`  | Convert text to 384-dim L2-normalized embeddings                |
| `src.database.vector_store`        | `VectorStore`       | Endee SDK — index management, upsert, top-k search with filters |
| `src.query_handler.rag_engine`     | `RAGEngine`         | Full RAG — embed query → retrieve → inject context → answer     |
| `src.query_handler.groq_generator` | `GroqGenerator`     | Context injection + prompt engineering + Groq LLM answers       |
| `src.pipeline`                     | `RAGPipeline`       | High-level orchestrator wiring all modules together             |
| `src.config`                       | —                   | Central configuration with env-var overrides                    |
| `api.py`                           | Flask App           | REST API connecting Python backend to React frontend            |

---

## License

MIT License — free to use, modify, and distribute.

---

_Built with [Endee](https://endee.io) · [sentence-transformers](https://www.sbert.net) · [Groq](https://groq.com) · [React](https://react.dev) · Python 3.9+_
