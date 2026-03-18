"""
embeddings/encoder.py
----------------------
Converts text into dense vector embeddings using sentence-transformers.

Default model: all-MiniLM-L6-v2
  - 384-dimensional output
  - Lightweight and fast (< 100 MB)
  - Strong semantic similarity performance
  - Fully open-source (Apache 2.0)

The encoder supports batch processing with a configurable batch size
to avoid memory issues on large corpora.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import: sentence_transformers is only loaded when the encoder is first used
_st_module = None


def _get_sentence_transformers():
    global _st_module
    if _st_module is None:
        try:
            import sentence_transformers as st
            _st_module = st
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from exc
    return _st_module


class EmbeddingEncoder:
    """
    Wraps a sentence-transformer model to produce normalized float embeddings.

    Usage:
        encoder = EmbeddingEncoder()
        vector  = encoder.encode("Hello world")          # single → List[float]
        vectors = encoder.encode_batch(["a", "b", "c"]) # batch  → List[List[float]]
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, ~22M params
    DEFAULT_BATCH_SIZE = 64

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ):
        """
        Args:
            model_name:  HuggingFace model id or local path.
            batch_size:  Number of texts encoded per forward pass.
            device:      'cpu' or 'cuda' / 'mps'.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None  # lazily loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (loads model on first access)."""
        return self._get_model().get_sentence_embedding_dimension()

    def encode(self, text: str) -> List[float]:
        """
        Encode a single text string.

        Returns:
            List[float] of length self.dimension (L2-normalized).
        """
        vector = self._get_model().encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Encode a list of texts efficiently in mini-batches.

        Args:
            texts:         List of strings to encode.
            show_progress: Display tqdm progress bar.

        Returns:
            List of embedding vectors, each of length self.dimension.
        """
        if not texts:
            return []

        logger.info("Encoding %d texts with model '%s'", len(texts), self.model_name)
        vectors: np.ndarray = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        Since vectors are L2-normalized, this equals the dot product.
        """
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        return float(np.dot(a, b))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            st = _get_sentence_transformers()
            logger.info("Loading embedding model: %s (device=%s)", self.model_name, self.device)
            self._model = st.SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "Model loaded. Dimension: %d",
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def __repr__(self) -> str:
        return (
            f"EmbeddingEncoder(model={self.model_name!r}, "
            f"batch_size={self.batch_size}, device={self.device!r})"
        )
