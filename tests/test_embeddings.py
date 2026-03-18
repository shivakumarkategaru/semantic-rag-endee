"""
tests/test_embeddings.py
-------------------------
Unit tests for EmbeddingEncoder.
These tests skip automatically if sentence-transformers is not installed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False

pytestmark = pytest.mark.skipif(
    not HAS_ST,
    reason="sentence-transformers not installed",
)

from src.embeddings.encoder import EmbeddingEncoder


@pytest.fixture(scope="module")
def encoder():
    return EmbeddingEncoder(model_name="all-MiniLM-L6-v2", device="cpu")


class TestEmbeddingEncoder:
    def test_dimension(self, encoder):
        assert encoder.dimension == 384

    def test_encode_single(self, encoder):
        vec = encoder.encode("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_encode_returns_normalized(self, encoder):
        import math
        vec = encoder.encode("Normalize me")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-3

    def test_encode_batch(self, encoder):
        texts = ["First text", "Second text", "Third text"]
        vecs = encoder.encode_batch(texts, show_progress=False)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_encode_empty_batch(self, encoder):
        assert encoder.encode_batch([], show_progress=False) == []

    def test_similarity_identical(self, encoder):
        vec = encoder.encode("Test sentence")
        sim = encoder.similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-4

    def test_similarity_different(self, encoder):
        v1 = encoder.encode("The cat sat on the mat")
        v2 = encoder.encode("Quantum mechanics is complex")
        sim = encoder.similarity(v1, v2)
        # Different topics should have low similarity
        assert sim < 0.8

    def test_semantic_similarity(self, encoder):
        v1 = encoder.encode("machine learning algorithms")
        v2 = encoder.encode("deep learning neural networks")
        v3 = encoder.encode("cooking pasta carbonara")
        sim_related = encoder.similarity(v1, v2)
        sim_unrelated = encoder.similarity(v1, v3)
        assert sim_related > sim_unrelated

    def test_repr(self, encoder):
        r = repr(encoder)
        assert "EmbeddingEncoder" in r
        assert "all-MiniLM-L6-v2" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
