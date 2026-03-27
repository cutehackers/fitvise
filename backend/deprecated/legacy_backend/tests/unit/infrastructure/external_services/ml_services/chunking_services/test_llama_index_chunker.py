"""
Unit tests for the LlamaIndexChunker fallback semantics.

These tests run without requiring llama_index to be installed by relying on the
character-based fallback splitter.
"""

import pytest

from app.infrastructure.external_services.ml_services.chunking_services.llama_index_chunker import (
    ChunkingDependencyError,
    LlamaIndexChunker,
    LlamaIndexChunkerConfig,
    SemanticChunk,
)


class TestLlamaIndexChunker:
    """Test suite for the LlamaIndexChunker class."""

    def test_chunk_returns_semantic_chunks_with_metadata_filtering(self):
        """Chunk a sample document and ensure metadata passthrough behaves correctly."""
        config = LlamaIndexChunkerConfig(
            chunk_size=64,
            chunk_overlap=16,
            min_chunk_chars=10,
        )
        chunker = LlamaIndexChunker(config=config, require_llama_index=False)

        text = (
            "Section 1\n"
            "This policy explains the retention guidelines for the finance department.\n\n"
            "Section 2\n"
            "All customer invoices must be archived for seven years in accordance with GAAP."
        )
        metadata = {"document_id": "doc-123", "source_id": "finance-hub", "extra_field": "ignored"}

        chunks = chunker.chunk(text, metadata=metadata)

        assert chunks, "Expected at least one chunk from fallback splitter"
        assert all(isinstance(chunk, SemanticChunk) for chunk in chunks)
        assert chunks[0].metadata["document_id"] == "doc-123"
        assert "extra_field" not in chunks[0].metadata  # filtered out by passthrough rules
        assert chunks[0].text

    def test_chunk_respects_max_chunks_limit(self):
        """Verify max_chunks prevents emitting more chunks than configured."""
        config = LlamaIndexChunkerConfig(
            chunk_size=32,
            chunk_overlap=8,
            min_chunk_chars=8,
            max_chunks=1,
        )
        chunker = LlamaIndexChunker(config=config, require_llama_index=False)
        text = "Paragraph A explains the workflow.\n\nParagraph B adds more operational detail."

        chunks = chunker.chunk(text, metadata={"document_id": "doc-456"})

        assert len(chunks) == 1

    def test_require_llama_index_raises_when_unavailable(self, monkeypatch):
        """Enforcing llama_index dependency should raise when bindings are missing."""
        module_path = (
            "app.infrastructure.external_services.ml_services.chunking_services.llama_index_chunker"
        )
        monkeypatch.setattr(f"{module_path}._LlamaDocument", None)
        monkeypatch.setattr(f"{module_path}._SentenceSplitter", None)

        with pytest.raises(ChunkingDependencyError):
            LlamaIndexChunker(require_llama_index=True)

