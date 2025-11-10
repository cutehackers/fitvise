"""
Test suite for Chunk entity serialization and deserialization.

This module tests the serialization/deserialization functionality of the Chunk entity,
ensuring that chunks can be properly converted to/from dictionaries for storage and retrieval.
"""
import pytest
from uuid import uuid4

from app.domain.entities.chunk import Chunk
from app.domain.value_objects.chunk_metadata import ChunkMetadata


class TestChunkSerialization:
    """Test cases for Chunk serialization and deserialization."""

    def test_chunk_basic_serialization(self):
        """Test that a chunk can be serialized to a dictionary."""
        # Arrange
        metadata = ChunkMetadata(
            sequence=1,
            start=0,
            end=100,
            section="test-section"
        )

        chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="This is a test chunk for serialization testing.",
            metadata=metadata,
            embedding_vector_id="test_vector_123",
            score=0.95,
            attributes={"test": "value"}
        )

        # Act
        chunk_dict = chunk.as_dict()

        # Assert
        assert isinstance(chunk_dict, dict)
        assert "id" in chunk_dict  # as_dict() uses "id" not "chunk_id"
        assert "document_id" in chunk_dict
        assert "text" in chunk_dict
        assert "metadata" in chunk_dict
        assert chunk_dict["text"] == chunk.text

    def test_chunk_deserialization(self):
        """Test that a chunk can be deserialized from a dictionary."""
        # Arrange
        metadata = ChunkMetadata(
            sequence=1,
            start=0,
            end=100,
            section="test-section"
        )

        original_chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="This is a test chunk for deserialization testing.",
            metadata=metadata,
            embedding_vector_id="test_vector_123",
            score=0.95,
            attributes={"test": "value"}
        )

        # Act
        chunk_dict = original_chunk.as_dict()
        restored_chunk = Chunk.from_dict(chunk_dict)

        # Assert
        assert restored_chunk is not None
        assert restored_chunk.chunk_id == original_chunk.chunk_id

    def test_chunk_roundtrip_serialization(self):
        """Test that a chunk maintains all data through serialize-deserialize cycle."""
        # Arrange
        metadata = ChunkMetadata(
            sequence=1,
            start=0,
            end=100,
            section="test-section"
        )

        original_chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="This is a test chunk for roundtrip testing.",
            metadata=metadata,
            embedding_vector_id="test_vector_123",
            score=0.95,
            attributes={"test": "value", "nested": {"key": "value"}}
        )

        # Act
        chunk_dict = original_chunk.as_dict()
        restored_chunk = Chunk.from_dict(chunk_dict)

        # Assert - Verify all fields match
        assert original_chunk.chunk_id == restored_chunk.chunk_id
        assert str(original_chunk.document_id) == str(restored_chunk.document_id)
        assert original_chunk.text == restored_chunk.text
        assert original_chunk.metadata.sequence == restored_chunk.metadata.sequence
        assert original_chunk.metadata.start == restored_chunk.metadata.start
        assert original_chunk.metadata.end == restored_chunk.metadata.end
        assert original_chunk.metadata.section == restored_chunk.metadata.section
        assert original_chunk.embedding_vector_id == restored_chunk.embedding_vector_id
        assert original_chunk.score == restored_chunk.score
        assert original_chunk.attributes == restored_chunk.attributes

    def test_chunk_deserialization_with_minimal_data(self):
        """Test chunk deserialization with only required fields."""
        # Arrange
        metadata = ChunkMetadata(
            sequence=1,
            start=0,
            end=50
        )

        original_chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="Minimal test chunk.",
            metadata=metadata
        )

        # Act
        chunk_dict = original_chunk.as_dict()
        restored_chunk = Chunk.from_dict(chunk_dict)

        # Assert
        assert restored_chunk.chunk_id == original_chunk.chunk_id
        assert restored_chunk.text == original_chunk.text
        assert restored_chunk.embedding_vector_id is None
        assert restored_chunk.score is None
        assert restored_chunk.attributes == {}  # Default is {}, not None

    def test_chunk_metadata_deserialization(self):
        """Test that chunk metadata is properly deserialized."""
        # Arrange
        metadata = ChunkMetadata(
            sequence=5,
            start=100,
            end=500,
            section="important-section",
            page_number=42
        )

        original_chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="Test chunk with detailed metadata.",
            metadata=metadata
        )

        # Act
        chunk_dict = original_chunk.as_dict()
        restored_chunk = Chunk.from_dict(chunk_dict)

        # Assert - Verify metadata fields
        assert restored_chunk.metadata is not None
        assert restored_chunk.metadata.sequence == 5
        assert restored_chunk.metadata.start == 100
        assert restored_chunk.metadata.end == 500
        assert restored_chunk.metadata.section == "important-section"
        assert restored_chunk.metadata.page_number == 42

    def test_chunk_with_none_optional_fields(self):
        """Test chunk deserialization with None values for optional fields."""
        # Arrange
        metadata = ChunkMetadata(
            sequence=1,
            start=0,
            end=100
        )

        original_chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="Test chunk with None optional fields.",
            metadata=metadata,
            embedding_vector_id=None,
            score=None,
            attributes={}  # Use empty dict instead of None (dataclass default)
        )

        # Act
        chunk_dict = original_chunk.as_dict()
        restored_chunk = Chunk.from_dict(chunk_dict)

        # Assert
        assert restored_chunk.chunk_id == original_chunk.chunk_id
        assert restored_chunk.embedding_vector_id is None
        assert restored_chunk.score is None
        assert restored_chunk.attributes == {}  # Default is {}, not None
