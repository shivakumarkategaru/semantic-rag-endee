"""
tests/test_pipeline.py
-----------------------
Integration-style tests for the RAGPipeline using a mock Endee client.
These tests run without a live Endee server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch


# ─── Lightweight mock for Endee SDK ──────────────────────────────────

class MockIndex:
    def __init__(self):
        self._store = {}

    def upsert(self, vectors):
        for v in vectors:
            self._store[v["id"]] = v

    def query(self, vector, top_k=5, ef=128, include_vectors=False, **kwargs):
        results = []
        for vid, vdata in list(self._store.items())[:top_k]:
            results.append({
                "id": vid,
                "similarity": 0.85,
                "meta": vdata.get("meta", {}),
            })
        return results

    def describe(self):
        return {"vectors_count": len(self._store), "dimension": 384}

    def get_vector(self, vid):
        return self._store.get(vid)


class MockEndeeClient:
    def __init__(self, token=""):
        self._indexes = {}

    def set_base_url(self, url):
        pass

    def list_indexes(self):
        return [{"name": k} for k in self._indexes]

    def create_index(self, name, dimension, space_type, precision=None):
        self._indexes[name] = MockIndex()

    def delete_index(self, name):
        self._indexes.pop(name, None)

    def get_index(self, name):
        if name not in self._indexes:
            self.create_index(name, 384, "cosine")
        return self._indexes[name]


# ─── Tests ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_endee(monkeypatch):
    """Patch the endee module with our mock."""
    mock_module = MagicMock()
    mock_module.Endee = MockEndeeClient
    mock_module.Precision = MagicMock()
    mock_module.Precision.INT8 = "INT8"
    monkeypatch.setattr("src.database.vector_store._endee_module", mock_module)
    return mock_module


@pytest.fixture
def mock_encoder():
    """Return a fast mock encoder with fixed 384-dim vectors."""
    encoder = MagicMock()
    encoder.dimension = 384
    encoder.model_name = "mock-model"
    encoder.encode.return_value = [0.1] * 384
    encoder.encode_batch.return_value = [[0.1] * 384, [0.2] * 384]
    return encoder


class TestVectorStore:
    def test_ensure_index_creates_new(self, mock_endee):
        from src.database.vector_store import VectorStore
        store = VectorStore(index_name="test_idx", dimension=384)
        store.ensure_index()
        assert store.vector_count() >= 0  # index exists

    def test_upsert_and_search(self, mock_endee):
        from src.database.vector_store import VectorStore
        from src.data_processing.processor import Document

        store = VectorStore(index_name="test_idx2", dimension=384)
        store.ensure_index()

        docs = [
            Document(id="d1", text="Hello world", source="test", chunk_index=0),
            Document(id="d2", text="Machine learning", source="test", chunk_index=1),
        ]
        vectors = [[0.1] * 384, [0.2] * 384]
        store.upsert_documents(docs, vectors)

        results = store.search([0.1] * 384, top_k=2)
        assert len(results) >= 1

    def test_search_result_fields(self, mock_endee):
        from src.database.vector_store import VectorStore, SearchResult
        from src.data_processing.processor import Document

        store = VectorStore(index_name="test_idx3", dimension=384)
        store.ensure_index()
        docs = [Document(id="x1", text="Sample text", source="src", chunk_index=0)]
        store.upsert_documents(docs, [[0.5] * 384])

        results = store.search([0.5] * 384)
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.score, float)
            assert isinstance(r.id, str)


class TestRAGEngine:
    def test_query_returns_response(self, mock_endee, mock_encoder):
        from src.database.vector_store import VectorStore
        from src.query_handler.rag_engine import RAGEngine, RAGResponse
        from src.data_processing.processor import Document

        store = VectorStore(index_name="eng_idx", dimension=384)
        store.ensure_index()
        docs = [Document(
            id="e1",
            text="The Transformer model uses attention mechanisms.",
            source="wiki",
            chunk_index=0,
        )]
        store.upsert_documents(docs, [[0.3] * 384])

        engine = RAGEngine(encoder=mock_encoder, vector_store=store, top_k=3)
        response = engine.query("What is a Transformer?")

        assert isinstance(response, RAGResponse)
        assert response.question == "What is a Transformer?"
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_semantic_search(self, mock_endee, mock_encoder):
        from src.database.vector_store import VectorStore
        from src.query_handler.rag_engine import RAGEngine
        from src.data_processing.processor import Document

        store = VectorStore(index_name="ss_idx", dimension=384)
        store.ensure_index()
        docs = [Document(id="s1", text="Python is a language", source="s", chunk_index=0)]
        store.upsert_documents(docs, [[0.1] * 384])

        engine = RAGEngine(encoder=mock_encoder, vector_store=store)
        results = engine.semantic_search("Python programming")
        assert isinstance(results, list)

    def test_rag_response_display(self, mock_endee, mock_encoder):
        from src.database.vector_store import VectorStore, SearchResult
        from src.query_handler.rag_engine import RAGResponse

        sources = [SearchResult(
            id="t1", score=0.9, text="Test context sentence.",
            source="doc.txt", chunk_index=0, metadata={},
        )]
        response = RAGResponse(
            question="Test question?",
            answer="Test answer.",
            sources=sources,
            model_used="template",
        )
        display = response.display()
        assert "Test question?" in display
        assert "Test answer." in display
        assert "doc.txt" in display


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
