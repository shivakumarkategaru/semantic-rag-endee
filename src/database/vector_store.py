"""
database/vector_store.py
-------------------------
Thin abstraction layer over the Endee Python SDK.

Responsibilities:
  - Create / delete indexes
  - Batch-upsert Document embeddings
  - Perform similarity searches
  - Expose a clean interface decoupled from Endee internals

Endee Python SDK reference:
  https://docs.endee.io/python-sdk/usage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import so the module can be imported even without endee installed
_endee_module = None


def _get_endee():
    global _endee_module
    if _endee_module is None:
        try:
            import endee as _e
            _endee_module = _e
        except ImportError as exc:
            raise ImportError(
                "The 'endee' package is required. Install with: pip install endee"
            ) from exc
    return _endee_module


@dataclass
class SearchResult:
    """One search hit returned by Endee."""
    id: str
    score: float
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.id!r}, score={self.score:.4f}, "
            f"source={self.source!r}, text={self.text[:60]!r}...)"
        )


class VectorStore:
    """
    High-level vector database interface backed by Endee.

    Usage:
        store = VectorStore(index_name="rag_docs", dimension=384)
        store.ensure_index()
        store.upsert_documents(docs, vectors)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        index_name: str = "rag_index",
        dimension: int = 384,
        host: str = "http://localhost:8080",
        auth_token: str = "",
        space_type: str = "cosine",
    ):
        """
        Args:
            index_name:  Name of the Endee index to use.
            dimension:   Embedding dimension (must match the encoder output).
            host:        Endee server URL.
            auth_token:  Optional auth token (empty = no auth).
            space_type:  Distance metric: 'cosine' | 'l2' | 'dot'.
        """
        self.index_name = index_name
        self.dimension = dimension
        self.host = host
        self.auth_token = auth_token
        self.space_type = space_type

        self._client = None
        self._index = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            endee = _get_endee()
            self._client = endee.Endee(self.auth_token) if self.auth_token else endee.Endee()
            base_url = f"{self.host.rstrip('/')}/api/v1"
            self._client.set_base_url(base_url)
            logger.info("Connected to Endee at %s", base_url)
        return self._client

    def _get_index(self):
        if self._index is None:
            self._index = self._get_client().get_index(name=self.index_name)
        return self._index

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def ensure_index(self, recreate: bool = False) -> None:
        """
        Create the Endee index if it does not already exist.

        Args:
            recreate: If True, delete existing index and recreate it.
        """
        endee = _get_endee()
        client = self._get_client()

        existing = [idx.get("name") for idx in (client.list_indexes().get("indexes") or [])]

        if self.index_name in existing:
            if recreate:
                logger.info("Deleting existing index '%s' for recreation.", self.index_name)
                client.delete_index(self.index_name)
                self._index = None
            else:
                logger.info("Index '%s' already exists. Reusing.", self.index_name)
                return

        logger.info(
            "Creating index '%s' (dim=%d, space=%s).",
            self.index_name, self.dimension, self.space_type,
        )
        client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            space_type=self.space_type,
            precision=endee.Precision.INT8,
        )
        logger.info("Index '%s' created.", self.index_name)

    def delete_index(self) -> None:
        """Permanently delete the index and all its vectors."""
        self._get_client().delete_index(self.index_name)
        self._index = None
        logger.info("Index '%s' deleted.", self.index_name)

    def describe(self) -> Dict[str, Any]:
        """Return metadata about the index (vector count, dimension, etc.)."""
        return self._get_index().describe()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        documents,          # List[Document] from data_processing
        vectors: List[List[float]],
        batch_size: int = 256,
    ) -> None:
        """
        Store document embeddings in Endee.

        Each vector is stored with rich metadata so search results
        can reconstruct the original text without a secondary lookup.

        Args:
            documents:   Processed Document objects.
            vectors:     Corresponding embedding vectors (same order).
            batch_size:  Number of vectors per upsert call.
        """
        assert len(documents) == len(vectors), (
            f"Mismatch: {len(documents)} documents vs {len(vectors)} vectors."
        )

        index = self._get_index()
        total = len(documents)
        logger.info("Upserting %d vectors into '%s'...", total, self.index_name)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_vecs = vectors[start:end]

            index.upsert([{"id":doc.id,"vector":vec,"meta":{"text":doc.text,"source":doc.source,"chunk_index":doc.chunk_index}} for doc,vec in zip(batch_docs,batch_vecs)])
          
            logger.debug("Upserted batch %d-%d / %d", start, end, total)

        logger.info("Upsert complete. %d vectors stored.", total)

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        ef: int = 128,
        filters: Optional[List[Dict]] = None,
    ) -> List[SearchResult]:
        """
        Find the top-k most similar vectors in Endee.

        Args:
            query_vector: Dense embedding of the query text.
            top_k:        Maximum number of results.
            ef:           Search quality (higher = more accurate, slower).
            filters:      Optional Endee filter list, e.g.:
                          [{"source": {"$eq": "wiki.txt"}}]

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        index = self._get_index()

        query_kwargs: Dict[str, Any] = dict(
            vector=query_vector,
            top_k=top_k,
            ef=ef,
            include_vectors=False,
        )
        if filters:
            query_kwargs["filter"] = filters

        raw_results = index.query(**query_kwargs)

        results: List[SearchResult] = []
        for item in (raw_results or []):
            meta = item.get("meta") or {}
            results.append(
                SearchResult(
                    id=item.get("id", ""),
                    score=float(item.get("similarity", 0.0)),
                    text=meta.get("text", ""),
                    source=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata={k: v for k, v in meta.items()
                              if k not in {"text", "source", "chunk_index"}},
                )
            )
        return results

    def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single vector record by its ID."""
        return self._get_index().get_vector(vector_id)

    def vector_count(self) -> int:
        """Return the number of vectors currently stored in the index."""
        info = self.describe()
        return int(info.get("vectors_count", info.get("count", 0)))

    def __repr__(self) -> str:
        return (
            f"VectorStore(index={self.index_name!r}, "
            f"dim={self.dimension}, host={self.host!r})"
        )
