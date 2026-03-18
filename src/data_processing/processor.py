"""
data_processing/processor.py
-----------------------------
Handles document ingestion, NLP preprocessing, and text chunking.
Supports plain text, JSON, and CSV document sources.
"""

import re
import json
import csv
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class Document:
    """Represents a processed text chunk ready for embedding."""
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Document(id={self.id!r}, source={self.source!r}, text={preview!r}...)"


class TextCleaner:
    """
    NLP text cleaning utilities.
    Applies normalization, whitespace cleanup, and noise removal.
    """

    @staticmethod
    def clean(text: str) -> str:
        """Full cleaning pipeline: normalize → strip noise → collapse whitespace."""
        text = TextCleaner._normalize_unicode(text)
        text = TextCleaner._remove_noise(text)
        text = TextCleaner._collapse_whitespace(text)
        return text.strip()

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Replace common unicode artifacts with ASCII equivalents."""
        replacements = {
            "\u2019": "'", "\u2018": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--",
            "\u00a0": " ",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    @staticmethod
    def _remove_noise(text: str) -> str:
        """Strip URLs, excessive punctuation, and HTML-like tags."""
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^\w\s.,!?;:()\-'\"]+", " ", text)
        return text

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Reduce all whitespace sequences to a single space."""
        return re.sub(r"\s+", " ", text)


class TextChunker:
    """
    Splits long text into overlapping chunks suitable for embedding.

    Uses a sentence-aware sliding window strategy:
    - Splits by sentence boundaries first
    - Groups sentences until chunk_size is reached
    - Applies overlap_size character overlap between chunks
    """

    def __init__(self, chunk_size: int = 512, overlap_size: int = 64):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk(self, text: str) -> List[str]:
        """Return a list of text chunks from the input text."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sentences = self._split_sentences(text)
        return self._build_chunks(sentences)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text on sentence-ending punctuation."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_chunks(self, sentences: List[str]) -> List[str]:
        """Greedily group sentences into chunks with overlap."""
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.chunk_size and current:
                chunks.append(current.strip())
                # Keep overlap: take last overlap_size chars as context
                current = current[-self.overlap_size:].lstrip() + " " + sentence
            else:
                current = (current + " " + sentence).strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks


class DocumentProcessor:
    """
    Main entry point for document ingestion.

    Pipeline:
        raw text / file → clean → chunk → [Document, ...]

    Supports:
        - Plain strings
        - .txt files
        - .json files  (expects list of {"title": ..., "content": ...})
        - .csv files   (expects columns: title, content)
    """

    def __init__(self, chunk_size: int = 512, overlap_size: int = 64):
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_text(
        self,
        text: str,
        source: str = "inline",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Process a raw text string into Document chunks."""
        cleaned = self.cleaner.clean(text)
        chunks = self.chunker.chunk(cleaned)
        return self._build_documents(chunks, source, metadata or {})

    def process_file(self, filepath: str) -> List[Document]:
        """
        Auto-detect file format and process accordingly.
        Supported: .txt, .json, .csv
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = path.suffix.lower()
        if suffix not in {".txt", ".json", ".csv"}:
            raise ValueError(f"Unsupported file format: {suffix}")
        if suffix == ".txt":
            return self._process_txt(path)
        elif suffix == ".json":
            return self._process_json(path)
        else:
            return self._process_csv(path)

    def process_batch(self, texts: List[Dict[str, str]]) -> List[Document]:
        """
        Process a batch of dicts with 'text' and optional 'source'/'metadata'.

        Example input:
            [{"text": "Hello world", "source": "doc1", "metadata": {"author": "Alice"}}]
        """
        all_docs: List[Document] = []
        for item in texts:
            docs = self.process_text(
                text=item["text"],
                source=item.get("source", "batch"),
                metadata=item.get("metadata", {}),
            )
            all_docs.extend(docs)
        return all_docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_txt(self, path: Path) -> List[Document]:
        text = path.read_text(encoding="utf-8")
        return self.process_text(text, source=path.name)

    def _process_json(self, path: Path) -> List[Document]:
        data = json.loads(path.read_text(encoding="utf-8"))
        all_docs: List[Document] = []
        if isinstance(data, list):
            for entry in data:
                title = entry.get("title", "")
                content = entry.get("content", entry.get("text", ""))
                full_text = f"{title}. {content}" if title else content
                meta = {k: v for k, v in entry.items() if k not in {"title", "content", "text"}}
                all_docs.extend(self.process_text(full_text, source=path.name, metadata=meta))
        else:
            all_docs.extend(self.process_text(str(data), source=path.name))
        return all_docs

    def _process_csv(self, path: Path) -> List[Document]:
        all_docs: List[Document] = []
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get("title", "")
                content = row.get("content", row.get("text", ""))
                full_text = f"{title}. {content}" if title else content
                meta = {k: v for k, v in row.items() if k not in {"title", "content", "text"}}
                all_docs.extend(self.process_text(full_text, source=path.name, metadata=meta))
        return all_docs

    def _build_documents(
        self,
        chunks: List[str],
        source: str,
        metadata: Dict[str, Any],
    ) -> List[Document]:
        docs = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            docs.append(
                Document(
                    id=str(uuid.uuid4()),
                    text=chunk,
                    source=source,
                    chunk_index=i,
                    metadata=metadata,
                )
            )
        return docs
