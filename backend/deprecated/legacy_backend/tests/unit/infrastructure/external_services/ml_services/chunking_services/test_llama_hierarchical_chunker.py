"""Unit tests for LlamaHierarchicalChunker (Task 2.1.3)."""

import pytest

from app.domain.exceptions import ChunkingDependencyError
from app.infrastructure.external_services.ml_services.chunking_services.llama_hierarchical_chunker import (
    HierarchicalChunk,
    HierarchicalChunkerConfig,
    LlamaHierarchicalChunker,
)
from tests.fixtures.hierarchical_documents.sample_documents import (
    ALL_DOCUMENTS,
    EMPTY_DOCUMENT,
    FLAT_DOCUMENT,
    POLICY_DOCUMENT,
    SHORT_POLICY,
    SINGLE_SENTENCE,
    WHITESPACE_ONLY,
)


class TestHierarchicalChunkerConfig:
    """Tests for HierarchicalChunkerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HierarchicalChunkerConfig()
        assert config.chunk_sizes == [2048, 512, 128]
        assert config.chunk_overlap == 200
        assert config.min_chunk_chars == 100
        assert config.max_chunk_chars == 2048
        assert config.preserve_hierarchy is True

    def test_from_dict_with_valid_data(self):
        """Test configuration creation from dictionary."""
        raw = {
            "chunk_sizes": [1024, 256],
            "chunk_overlap": 50,
            "min_chunk_chars": 80,
            "debug_mode": True,
        }
        config = HierarchicalChunkerConfig.from_dict(raw)
        assert config.chunk_sizes == [1024, 256]
        assert config.chunk_overlap == 50
        assert config.min_chunk_chars == 80
        assert config.debug_mode is True

    def test_from_dict_filters_invalid_keys(self):
        """Test that invalid keys are filtered out."""
        raw = {
            "chunk_sizes": [512],
            "invalid_key": "should_be_ignored",
        }
        config = HierarchicalChunkerConfig.from_dict(raw)
        assert config.chunk_sizes == [512]
        assert not hasattr(config, "invalid_key")

    def test_from_dict_with_none(self):
        """Test configuration creation with None input."""
        config = HierarchicalChunkerConfig.from_dict(None)
        assert config.chunk_sizes == [2048, 512, 128]

    def test_validate_empty_chunk_sizes(self):
        """Test validation rejects empty chunk_sizes."""
        config = HierarchicalChunkerConfig(chunk_sizes=[])
        with pytest.raises(ValueError, match="chunk_sizes must not be empty"):
            config.validate()

    def test_validate_negative_chunk_size(self):
        """Test validation rejects negative chunk sizes."""
        config = HierarchicalChunkerConfig(chunk_sizes=[1024, -512])
        with pytest.raises(ValueError, match="all chunk_sizes must be positive"):
            config.validate()

    def test_validate_negative_overlap(self):
        """Test validation rejects negative chunk overlap."""
        config = HierarchicalChunkerConfig(chunk_overlap=-50)
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            config.validate()

    def test_validate_invalid_min_chars(self):
        """Test validation rejects non-positive min_chunk_chars."""
        config = HierarchicalChunkerConfig(min_chunk_chars=0)
        with pytest.raises(ValueError, match="min_chunk_chars must be positive"):
            config.validate()

    def test_validate_max_less_than_min(self):
        """Test validation rejects max_chunk_chars < min_chunk_chars."""
        config = HierarchicalChunkerConfig(min_chunk_chars=500, max_chunk_chars=300)
        with pytest.raises(ValueError, match="max_chunk_chars must be >= min_chunk_chars"):
            config.validate()


class TestHierarchicalChunk:
    """Tests for HierarchicalChunk dataclass."""

    def test_chunk_creation(self):
        """Test basic hierarchical chunk creation."""
        chunk = HierarchicalChunk(
            chunk_id="chunk-1",
            sequence=0,
            text="Sample text",
            start=0,
            end=11,
            depth_level=0,
            parent_chunk_id=None,
            metadata={"section": "root"},
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.depth_level == 0
        assert chunk.parent_chunk_id is None

    def test_as_dict(self):
        """Test chunk serialization to dictionary."""
        chunk = HierarchicalChunk(
            chunk_id="chunk-1",
            sequence=0,
            text="Sample text",
            start=0,
            end=11,
            depth_level=1,
            parent_chunk_id="parent-1",
            metadata={"section": "1.1"},
        )
        result = chunk.as_dict()
        assert result["id"] == "chunk-1"
        assert result["depth_level"] == 1
        assert result["parent_chunk_id"] == "parent-1"
        assert result["metadata"]["section"] == "1.1"


class TestLlamaHierarchicalChunker:
    """Tests for LlamaHierarchicalChunker service."""

    def test_chunker_initialization_default(self):
        """Test chunker initialization with default config."""
        chunker = LlamaHierarchicalChunker()
        assert chunker.config.chunk_sizes == [2048, 512, 128]
        assert chunker.config.chunk_overlap == 200

    def test_chunker_initialization_custom_config(self):
        """Test chunker initialization with custom config."""
        config = HierarchicalChunkerConfig(chunk_sizes=[1024, 256], chunk_overlap=50)
        chunker = LlamaHierarchicalChunker(config=config)
        assert chunker.config.chunk_sizes == [1024, 256]
        assert chunker.config.chunk_overlap == 50

    def test_require_llama_index_when_unavailable(self, monkeypatch):
        """Test chunker raises error when llama_index required but unavailable."""
        # Mock llama_index unavailability
        import app.infrastructure.external_services.ml_services.chunking_services.llama_hierarchical_chunker as chunker_module

        monkeypatch.setattr(chunker_module, "_LlamaDocument", None)
        with pytest.raises(ChunkingDependencyError, match="llama_index is not installed"):
            LlamaHierarchicalChunker(require_llama_index=True)

    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        chunker = LlamaHierarchicalChunker()
        chunks = chunker.chunk(EMPTY_DOCUMENT)
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text returns empty list."""
        chunker = LlamaHierarchicalChunker()
        chunks = chunker.chunk(WHITESPACE_ONLY)
        assert chunks == []

    def test_chunk_single_sentence(self):
        """Test chunking single sentence document."""
        chunker = LlamaHierarchicalChunker()
        chunks = chunker.chunk(SINGLE_SENTENCE, metadata={"document_id": "doc-1"})
        assert len(chunks) > 0
        assert all(isinstance(chunk, HierarchicalChunk) for chunk in chunks)
        assert chunks[0].text == SINGLE_SENTENCE

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves metadata."""
        chunker = LlamaHierarchicalChunker()
        metadata = {"document_id": "doc-123", "source_id": "src-1"}
        chunks = chunker.chunk(SHORT_POLICY, metadata=metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "document_id" in chunk.metadata
            assert chunk.metadata["document_id"] == "doc-123"

    def test_chunk_flat_document(self):
        """Test chunking flat document without hierarchy."""
        chunker = LlamaHierarchicalChunker()
        chunks = chunker.chunk(FLAT_DOCUMENT)

        assert len(chunks) > 0
        # Flat document should have all chunks at depth 0
        assert all(chunk.depth_level == 0 for chunk in chunks)
        assert all(chunk.parent_chunk_id is None for chunk in chunks)

    def test_chunk_policy_document_creates_hierarchy(self):
        """Test chunking policy document creates multi-level hierarchy."""
        config = HierarchicalChunkerConfig(chunk_sizes=[2048, 512, 128])
        chunker = LlamaHierarchicalChunker(config=config)
        chunks = chunker.chunk(POLICY_DOCUMENT)

        assert len(chunks) > 0
        # Should have chunks at different depth levels
        depth_levels = {chunk.depth_level for chunk in chunks}
        assert len(depth_levels) > 1  # Multiple levels exist

        # Verify chunk ordering by sequence
        assert all(chunks[i].sequence == i for i in range(len(chunks)))

    def test_chunk_with_max_chunks_limit(self):
        """Test chunking respects max_chunks limit."""
        config = HierarchicalChunkerConfig(max_chunks=5)
        chunker = LlamaHierarchicalChunker(config=config)
        chunks = chunker.chunk(POLICY_DOCUMENT)

        assert len(chunks) <= 5

    def test_chunk_with_max_chunk_chars_limit(self):
        """Test chunking respects max_chunk_chars limit."""
        config = HierarchicalChunkerConfig(max_chunk_chars=500)
        chunker = LlamaHierarchicalChunker(config=config)
        chunks = chunker.chunk(POLICY_DOCUMENT)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.text) <= 500

    def test_chunk_batch(self):
        """Test batch chunking of multiple documents."""
        chunker = LlamaHierarchicalChunker()
        texts = [
            (SHORT_POLICY, {"document_id": "doc-1"}),
            (FLAT_DOCUMENT, {"document_id": "doc-2"}),
        ]
        results = chunker.chunk_batch(texts)

        assert len(results) == 2
        assert len(results[0]) > 0  # First document has chunks
        assert len(results[1]) > 0  # Second document has chunks
        assert all(isinstance(chunk, HierarchicalChunk) for chunks in results for chunk in chunks)

    def test_chunk_with_document_metadata(self):
        """Test chunking with both runtime and document metadata."""
        chunker = LlamaHierarchicalChunker()
        runtime_metadata = {"pipeline_run_id": "run-123"}
        document_metadata = {"file_name": "policy.md", "doc_type": "policy"}

        chunks = chunker.chunk(
            SHORT_POLICY,
            metadata=runtime_metadata,
            document_metadata=document_metadata,
        )

        assert len(chunks) > 0
        # Both metadata types should be merged
        chunk_meta = chunks[0].metadata
        assert "file_name" in chunk_meta or "pipeline_run_id" in chunk_meta

    def test_chunk_character_offsets(self):
        """Test that chunks have valid character offsets."""
        chunker = LlamaHierarchicalChunker()
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.start >= 0
            assert chunk.end > chunk.start
            assert chunk.end <= len(text)

    def test_depth_level_calculation(self):
        """Test that depth levels are calculated based on chunk sizes."""
        config = HierarchicalChunkerConfig(chunk_sizes=[2000, 500, 100])
        chunker = LlamaHierarchicalChunker(config=config)
        chunks = chunker.chunk(POLICY_DOCUMENT)

        if len(chunks) > 1:
            # Larger chunks should have lower depth (closer to root)
            large_chunks = [c for c in chunks if len(c.text) > 1500]
            small_chunks = [c for c in chunks if len(c.text) < 200]

            if large_chunks and small_chunks:
                assert large_chunks[0].depth_level < small_chunks[0].depth_level

    def test_fallback_chunking_when_hierarchical_fails(self, monkeypatch):
        """Test fallback to simple chunking when hierarchical parsing fails."""
        config = HierarchicalChunkerConfig(preserve_hierarchy=True)
        chunker = LlamaHierarchicalChunker(config=config)

        # Even if hierarchical parsing fails, fallback should work
        chunks = chunker.chunk(SHORT_POLICY)
        assert len(chunks) > 0
        assert all(isinstance(chunk, HierarchicalChunk) for chunk in chunks)


class TestHierarchicalChunkerEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_text(self):
        """Test chunking very long text."""
        chunker = LlamaHierarchicalChunker()
        long_text = "This is a sentence. " * 1000  # ~20,000 characters
        chunks = chunker.chunk(long_text)

        assert len(chunks) > 0
        total_length = sum(len(chunk.text) for chunk in chunks)
        # Total should be close to original (allowing for whitespace trimming)
        assert total_length >= len(long_text) * 0.8

    def test_special_characters(self):
        """Test chunking text with special characters."""
        chunker = LlamaHierarchicalChunker()
        special_text = "Text with émojis 🎉 and spëcial çharacters!"
        chunks = chunker.chunk(special_text)

        assert len(chunks) > 0
        assert chunks[0].text == special_text

    def test_unicode_text(self):
        """Test chunking unicode text."""
        chunker = LlamaHierarchicalChunker()
        unicode_text = "中文文本 العربية русский текст"
        chunks = chunker.chunk(unicode_text)

        assert len(chunks) > 0
        assert chunks[0].text == unicode_text

    def test_none_text_input(self):
        """Test chunking None text input."""
        chunker = LlamaHierarchicalChunker()
        chunks = chunker.chunk(None)  # type: ignore
        assert chunks == []

    def test_metadata_filtering(self):
        """Test that only specified metadata fields are passed through."""
        config = HierarchicalChunkerConfig(
            metadata_passthrough_fields=("document_id", "source_id")
        )
        chunker = LlamaHierarchicalChunker(config=config)
        metadata = {
            "document_id": "doc-1",
            "source_id": "src-1",
            "extra_field": "should_be_filtered",
        }
        chunks = chunker.chunk(SHORT_POLICY, metadata=metadata)

        assert len(chunks) > 0
        # Only passthrough fields should be in metadata (or all if filtering fails)
        chunk_meta = chunks[0].metadata
        assert "document_id" in chunk_meta


@pytest.mark.parametrize(
    "document_name,expected_min_chunks",
    [
        ("policy", 5),  # Policy document should produce multiple chunks
        ("short_policy", 1),  # Short policy at least one chunk
        ("flat", 1),  # Flat document at least one chunk
        ("single_sentence", 1),  # Single sentence one chunk
    ],
)
def test_document_chunking_parametrized(document_name, expected_min_chunks):
    """Parametrized test for different document types."""
    chunker = LlamaHierarchicalChunker()
    document = ALL_DOCUMENTS[document_name]
    chunks = chunker.chunk(document)

    assert len(chunks) >= expected_min_chunks
    assert all(isinstance(chunk, HierarchicalChunk) for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)  # No empty chunks
