"""
pipeline.py
-----------
Top-level orchestrator that wires together all modules into a single
easy-to-use interface.

Typical workflow:

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.ingest_texts([
        {"text": "...", "source": "doc1"},
    ])
    response = pipeline.ask("What is quantum entanglement?")
    print(response.display())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src import config
from src.data_processing.processor import DocumentProcessor
from src.embeddings.encoder import EmbeddingEncoder
from src.database.vector_store import VectorStore
from src.query_handler.rag_engine import RAGEngine, RAGResponse, SearchResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline backed by Endee vector database.

    Initialisation sets up all components with values from config.py.
    Override any parameter in the constructor to customise behaviour.
    """

    def __init__(
        self,
        index_name: str = config.ENDEE_INDEX_NAME,
        endee_host: str = config.ENDEE_HOST,
        auth_token: str = config.ENDEE_AUTH_TOKEN,
        embedding_model: str = config.EMBEDDING_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.DEFAULT_TOP_K,
        use_llm: bool = config.USE_LLM,
        recreate_index: bool = False,
    ):
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        )

        logger.info("Initialising RAGPipeline...")

        # Components
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            overlap_size=chunk_overlap,
        )
        self.encoder = EmbeddingEncoder(
            model_name=embedding_model,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            device=config.EMBEDDING_DEVICE,
        )
        self.vector_store = VectorStore(
            index_name=index_name,
            dimension=config.EMBEDDING_DIM,
            host=endee_host,
            auth_token=auth_token,
            space_type=config.ENDEE_SPACE_TYPE,
        )
        self.engine = RAGEngine(
            encoder=self.encoder,
            vector_store=self.vector_store,
            top_k=top_k,
            ef=config.SEARCH_EF,
        )

        # Prepare the index
        self.vector_store.ensure_index(recreate=recreate_index)

        # Optionally load an LLM for answer generation
        if use_llm:
            success = self.engine.load_generator(config.GENERATOR_MODEL)
            if not success:
                logger.warning(
                    "LLM loading failed. Answers will use template extraction."
                )

        logger.info("RAGPipeline ready.")

    # ------------------------------------------------------------------
    # Ingestion API
    # ------------------------------------------------------------------

    def ingest_texts(
        self,
        texts: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> int:
        """
        Ingest a list of text dicts.

        Each dict must have 'text' and optionally 'source' and 'metadata'.

        Example:
            pipeline.ingest_texts([
                {"text": "Python is a programming language...", "source": "python_wiki"},
                {"text": "Machine learning is...", "source": "ml_wiki"},
            ])

        Returns:
            Number of document chunks stored.
        """
        logger.info("Processing %d input texts...", len(texts))
        documents = self.processor.process_batch(texts)

        if not documents:
            logger.warning("No documents produced from input.")
            return 0

        logger.info("Encoding %d document chunks...", len(documents))
        vectors = self.encoder.encode_batch(
            [doc.text for doc in documents],
            show_progress=show_progress,
        )

        self.vector_store.upsert_documents(documents, vectors)
        logger.info("Ingested %d chunks.", len(documents))
        return len(documents)

    def ingest_file(self, filepath: str, show_progress: bool = True) -> int:
        """
        Ingest all documents from a file (.txt, .json, .csv).

        Returns:
            Number of document chunks stored.
        """
        logger.info("Ingesting file: %s", filepath)
        documents = self.processor.process_file(filepath)

        if not documents:
            logger.warning("No content extracted from %s", filepath)
            return 0

        vectors = self.encoder.encode_batch(
            [doc.text for doc in documents],
            show_progress=show_progress,
        )

        self.vector_store.upsert_documents(documents, vectors)
        logger.info("Ingested %d chunks from %s.", len(documents), filepath)
        return len(documents)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[List] = None,
    ) -> RAGResponse:
        """
        Ask a natural language question. Returns a RAGResponse with
        the answer and the retrieved source chunks.
        """
        return self.engine.query(question, top_k=top_k, filters=filters)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Pure semantic search — returns ranked chunks without synthesis.
        """
        return self.engine.semantic_search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return current index statistics."""
        info = self.vector_store.describe()
        return {
            "index_name": self.vector_store.index_name,
            "vector_count": info.get("vectors_count", "?"),
            "dimension": self.vector_store.dimension,
            "space_type": self.vector_store.space_type,
            "embedding_model": self.encoder.model_name,
        }
