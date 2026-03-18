"""
tests/test_data_processing.py
-------------------------------
Unit tests for DocumentProcessor, TextCleaner, and TextChunker.
No external dependencies required — runs without Endee or GPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_processing.processor import (
    TextCleaner,
    TextChunker,
    DocumentProcessor,
    Document,
)


# ─── TextCleaner ─────────────────────────────────────────────────────

class TestTextCleaner:
    def test_removes_urls(self):
        text = "Visit https://example.com for details."
        result = TextCleaner.clean(text)
        assert "https://" not in result

    def test_normalizes_smart_quotes(self):
        text = "\u201cHello world\u201d"
        result = TextCleaner.clean(text)
        assert '"' in result

    def test_collapses_whitespace(self):
        text = "Hello    world\n\n  test"
        result = TextCleaner.clean(text)
        assert "  " not in result

    def test_strips_html_tags(self):
        text = "<p>Hello <b>world</b></p>"
        result = TextCleaner.clean(text)
        assert "<" not in result
        assert "Hello" in result

    def test_empty_string(self):
        assert TextCleaner.clean("") == ""


# ─── TextChunker ─────────────────────────────────────────────────────

class TestTextChunker:
    def test_short_text_returns_single_chunk(self):
        chunker = TextChunker(chunk_size=512)
        text = "Short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        chunker = TextChunker(chunk_size=50, overlap_size=10)
        text = "This is a sentence. " * 20
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        chunker = TextChunker(chunk_size=100, overlap_size=10)
        # A long text without sentence breaks — everything lumped into one "sentence"
        text = "word " * 200
        chunks = chunker.chunk(text)
        # Each chunk should not exceed chunk_size by much (overlap aside)
        for chunk in chunks[:-1]:
            assert len(chunk) <= 200  # generous tolerance

    def test_empty_text_returns_empty(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []


# ─── DocumentProcessor ───────────────────────────────────────────────

class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=200, overlap_size=20)

    def test_process_text_returns_documents(self):
        docs = self.processor.process_text("Hello world. This is a test document.")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_document_has_required_fields(self):
        docs = self.processor.process_text("Sample text.", source="test_src")
        assert len(docs) >= 1
        doc = docs[0]
        assert doc.id
        assert doc.text
        assert doc.source == "test_src"
        assert isinstance(doc.chunk_index, int)

    def test_metadata_passed_through(self):
        meta = {"author": "Alice", "year": 2024}
        docs = self.processor.process_text("Some content.", metadata=meta)
        for doc in docs:
            assert doc.metadata["author"] == "Alice"

    def test_process_batch(self):
        batch = [
            {"text": "First document about AI.", "source": "a"},
            {"text": "Second document about ML.", "source": "b"},
        ]
        docs = self.processor.process_batch(batch)
        sources = {d.source for d in docs}
        assert "a" in sources
        assert "b" in sources

    def test_unique_ids(self):
        docs = self.processor.process_text("Word. " * 100, source="test")
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_empty_text_returns_empty(self):
        docs = self.processor.process_text("")
        assert docs == []

    def test_process_json_file(self, tmp_path):
        import json
        data = [
            {"title": "Doc A", "content": "Content about topic A." * 5},
            {"title": "Doc B", "content": "Content about topic B." * 5},
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")
        docs = self.processor.process_file(str(json_file))
        assert len(docs) >= 2

    def test_process_csv_file(self, tmp_path):
        csv_content = "title,content\nDoc A,Content about A.\nDoc B,Content about B.\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        docs = self.processor.process_file(str(csv_file))
        assert len(docs) >= 2

    def test_unsupported_file_raises(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.processor.process_file(str(bad_file))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.processor.process_file("/nonexistent/path/file.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
