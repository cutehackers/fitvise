"""Unit tests for Embedding entity (Task 2.2.1).

Tests cover:
- Entity creation and validation
- Factory methods (for_chunk, for_query)
- Similarity operations
- Metadata handling
- Entity equality and identity
"""

from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.domain.entities.embedding import Embedding
from app.domain.value_objects.embedding_vector import EmbeddingVector


class TestEmbeddingCreation:
    """Test Embedding entity creation and initialization."""

    def test_create_minimal_embedding(self):
        """Test creating embedding with minimal required fields."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding(
            vector=vector,
            model_name="test-model",
            dimension=3,
        )

        assert embedding.id is not None
        assert isinstance(embedding.id, UUID)
        assert embedding.vector == vector
        assert embedding.model_name == "test-model"
        assert embedding.dimension == 3

    def test_create_full_embedding(self):
        """Test creating embedding with all fields."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        chunk_id = uuid4()
        doc_id = uuid4()
        metadata = {"key": "value"}

        embedding = Embedding(
            vector=vector,
            model_name="test-model",
            model_version="1.0",
            dimension=3,
            chunk_id=chunk_id,
            document_id=doc_id,
            metadata=metadata,
        )

        assert embedding.chunk_id == chunk_id
        assert embedding.document_id == doc_id
        assert embedding.metadata == metadata
        assert embedding.model_version == "1.0"

    def test_created_at_auto_set(self):
        """Test that created_at is automatically set."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        before = datetime.now()

        embedding = Embedding(
            vector=vector,
            model_name="test-model",
            dimension=3,
        )

        after = datetime.now()
        assert before <= embedding.created_at <= after

    def test_default_metadata_empty_dict(self):
        """Test that metadata defaults to empty dict."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding(
            vector=vector,
            model_name="test-model",
            dimension=3,
        )

        assert embedding.metadata == {}


class TestEmbeddingFactoryMethods:
    """Test factory methods for creating embeddings."""

    def test_for_chunk_factory(self):
        """Test creating embedding for chunk using factory method."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        chunk_id = uuid4()
        doc_id = uuid4()

        embedding = Embedding.for_chunk(
            vector=vector,
            chunk_id=chunk_id,
            document_id=doc_id,
            model_name="test-model",
            model_version="1.0",
        )

        assert embedding.chunk_id == chunk_id
        assert embedding.document_id == doc_id
        assert embedding.source_type == "chunk"
        assert embedding.metadata.get("source_type") == "chunk"

    def test_for_query_factory(self):
        """Test creating embedding for query using factory method."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        query_id = uuid4()

        embedding = Embedding.for_query(
            vector=vector,
            query_id=query_id,
            model_name="test-model",
            model_version="1.0",
        )

        assert embedding.query_id == query_id
        assert embedding.source_type == "query"
        assert embedding.metadata.get("source_type") == "query"

    def test_for_chunk_with_metadata(self):
        """Test chunk factory preserves additional metadata."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        chunk_id = uuid4()
        additional_meta = {"custom": "value", "count": 42}

        embedding = Embedding.for_chunk(
            vector=vector,
            chunk_id=chunk_id,
            model_name="test-model",
            metadata=additional_meta,
        )

        assert embedding.metadata["custom"] == "value"
        assert embedding.metadata["count"] == 42
        assert embedding.metadata["source_type"] == "chunk"

    def test_for_query_with_metadata(self):
        """Test query factory preserves additional metadata."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        query_id = uuid4()
        additional_meta = {"user_id": "123", "session": "abc"}

        embedding = Embedding.for_query(
            vector=vector,
            query_id=query_id,
            model_name="test-model",
            metadata=additional_meta,
        )

        assert embedding.metadata["user_id"] == "123"
        assert embedding.metadata["session"] == "abc"
        assert embedding.metadata["source_type"] == "query"


class TestEmbeddingSimilarity:
    """Test similarity operations on embeddings."""

    def test_similarity_to_identical_embedding(self):
        """Test similarity between identical embeddings is 1.0."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        e1 = Embedding(vector=vector, model_name="test", dimension=3)
        e2 = Embedding(vector=vector, model_name="test", dimension=3)

        similarity = e1.similarity_to(e2)

        assert pytest.approx(similarity, abs=1e-6) == 1.0

    def test_similarity_to_different_embedding(self):
        """Test similarity between different embeddings."""
        v1 = EmbeddingVector.from_list([1.0, 0.0, 0.0])
        v2 = EmbeddingVector.from_list([0.0, 1.0, 0.0])
        e1 = Embedding(vector=v1, model_name="test", dimension=3)
        e2 = Embedding(vector=v2, model_name="test", dimension=3)

        similarity = e1.similarity_to(e2)

        assert pytest.approx(similarity, abs=1e-6) == 0.0

    def test_similarity_delegates_to_vector(self):
        """Test that similarity calculation delegates to EmbeddingVector."""
        v1 = EmbeddingVector.from_list([3.0, 4.0, 0.0])
        v2 = EmbeddingVector.from_list([4.0, 3.0, 0.0])
        e1 = Embedding(vector=v1, model_name="test", dimension=3)
        e2 = Embedding(vector=v2, model_name="test", dimension=3)

        embedding_similarity = e1.similarity_to(e2)
        vector_similarity = v1.cosine_similarity(v2)

        assert embedding_similarity == vector_similarity


class TestEmbeddingEquality:
    """Test equality and identity operations."""

    def test_equality_same_id(self):
        """Test that embeddings with same ID are equal."""
        embedding_id = uuid4()
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        e1 = Embedding(id=embedding_id, vector=vector, model_name="test", dimension=3)
        e2 = Embedding(id=embedding_id, vector=vector, model_name="test", dimension=3)

        assert e1 == e2

    def test_inequality_different_id(self):
        """Test that embeddings with different IDs are not equal."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        e1 = Embedding(vector=vector, model_name="test", dimension=3)
        e2 = Embedding(vector=vector, model_name="test", dimension=3)

        assert e1 != e2

    def test_hash_based_on_id(self):
        """Test that hash is based on ID for set/dict usage."""
        embedding_id = uuid4()
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        e1 = Embedding(id=embedding_id, vector=vector, model_name="test", dimension=3)
        e2 = Embedding(id=embedding_id, vector=vector, model_name="test", dimension=3)

        assert hash(e1) == hash(e2)

        # Can use in sets
        embedding_set = {e1, e2}
        assert len(embedding_set) == 1


class TestEmbeddingMetadata:
    """Test metadata handling and operations."""

    def test_metadata_independence(self):
        """Test that metadata dictionaries are independent."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        shared_meta = {"key": "value"}

        e1 = Embedding(vector=vector, model_name="test", dimension=3, metadata=shared_meta)
        e2 = Embedding(vector=vector, model_name="test", dimension=3, metadata=shared_meta)

        # Modifying one should not affect the other
        e1.metadata["new_key"] = "new_value"

        assert "new_key" in e1.metadata
        assert "new_key" not in e2.metadata

    def test_update_metadata(self):
        """Test updating metadata after creation."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding(
            vector=vector,
            model_name="test",
            dimension=3,
            metadata={"initial": "value"},
        )

        embedding.metadata["added"] = "new"
        embedding.metadata["initial"] = "updated"

        assert embedding.metadata["initial"] == "updated"
        assert embedding.metadata["added"] == "new"

    def test_metadata_with_complex_types(self):
        """Test metadata can store complex types."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        complex_meta = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "data"},
        }

        embedding = Embedding(
            vector=vector,
            model_name="test",
            dimension=3,
            metadata=complex_meta,
        )

        assert embedding.metadata == complex_meta


class TestEmbeddingValidation:
    """Test validation and error handling."""

    def test_dimension_matches_vector(self):
        """Test that dimension should match vector dimension."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        # Creating with matching dimension
        embedding = Embedding(vector=vector, model_name="test", dimension=3)
        assert embedding.dimension == 3

        # Creating with mismatched dimension (allowed but inconsistent)
        embedding2 = Embedding(vector=vector, model_name="test", dimension=5)
        assert embedding2.dimension == 5  # Stores what's provided
        assert embedding2.vector.dimension == 3  # But vector has actual dimension

    def test_none_vector_allowed(self):
        """Test that None vector is allowed for placeholder entities."""
        embedding = Embedding(
            vector=None,
            model_name="test-model",
            dimension=384,
        )

        assert embedding.vector is None
        assert embedding.dimension == 384


class TestEmbeddingSourceTypes:
    """Test source type classification."""

    def test_chunk_source_type(self):
        """Test chunk embeddings have correct source type."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding.for_chunk(
            vector=vector,
            chunk_id=uuid4(),
            model_name="test",
        )

        assert embedding.source_type == "chunk"
        assert embedding.chunk_id is not None
        assert embedding.query_id is None

    def test_query_source_type(self):
        """Test query embeddings have correct source type."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding.for_query(
            vector=vector,
            query_id=uuid4(),
            model_name="test",
        )

        assert embedding.source_type == "query"
        assert embedding.query_id is not None
        assert embedding.chunk_id is None

    def test_generic_source_type(self):
        """Test generic embeddings have default source type."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding(
            vector=vector,
            model_name="test",
            dimension=3,
        )

        assert embedding.source_type == "unknown"


class TestEmbeddingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_dimension_embedding(self):
        """Test creating embedding with high dimensions (4096)."""
        vector = EmbeddingVector.from_numpy(np.random.rand(4096).astype(np.float32))
        embedding = Embedding(
            vector=vector,
            model_name="test-model",
            dimension=4096,
        )

        assert embedding.dimension == 4096
        assert embedding.vector.dimension == 4096

    def test_empty_model_name(self):
        """Test that empty model name is allowed but not recommended."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        embedding = Embedding(
            vector=vector,
            model_name="",
            dimension=3,
        )

        assert embedding.model_name == ""

    def test_special_characters_in_metadata(self):
        """Test metadata with special characters and unicode."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        metadata = {
            "unicode": "こんにちは",
            "special": "!@#$%^&*()",
            "emoji": "🚀🎉",
        }

        embedding = Embedding(
            vector=vector,
            model_name="test",
            dimension=3,
            metadata=metadata,
        )

        assert embedding.metadata["unicode"] == "こんにちは"
        assert embedding.metadata["emoji"] == "🚀🎉"


class TestEmbeddingRealisticScenarios:
    """Test realistic usage scenarios."""

    def test_create_document_chunk_embedding(self):
        """Test creating embedding for document chunk (realistic scenario)."""
        # Simulate all-MiniLM-L6-v2 output
        vector = EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))
        chunk_id = uuid4()
        doc_id = uuid4()

        embedding = Embedding.for_chunk(
            vector=vector,
            chunk_id=chunk_id,
            document_id=doc_id,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_version="1.0",
            metadata={
                "text": "Exercise routine for lower back pain",
                "doc_type": "pdf",
                "sequence": 5,
                "token_count": 12,
            },
        )

        assert embedding.dimension == 384
        assert embedding.chunk_id == chunk_id
        assert embedding.document_id == doc_id
        assert embedding.metadata["doc_type"] == "pdf"
        assert embedding.metadata["sequence"] == 5

    def test_create_user_query_embedding(self):
        """Test creating embedding for user query (realistic scenario)."""
        # Simulate query embedding
        vector = EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))
        query_id = uuid4()

        embedding = Embedding.for_query(
            vector=vector,
            query_id=query_id,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_version="1.0",
            metadata={
                "query_text": "What exercises help with lower back pain?",
                "user_id": "user_123",
                "session_id": "session_456",
            },
        )

        assert embedding.dimension == 384
        assert embedding.query_id == query_id
        assert "query_text" in embedding.metadata
        assert embedding.source_type == "query"

    def test_compare_query_to_chunks(self):
        """Test comparing query embedding to chunk embeddings."""
        # Query embedding
        query_vector = EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))
        query_emb = Embedding.for_query(
            vector=query_vector,
            query_id=uuid4(),
            model_name="test",
        )

        # Chunk embeddings
        chunk_embeddings = []
        for i in range(5):
            chunk_vector = EmbeddingVector.from_numpy(
                np.random.rand(384).astype(np.float32)
            )
            chunk_emb = Embedding.for_chunk(
                vector=chunk_vector,
                chunk_id=uuid4(),
                model_name="test",
            )
            chunk_embeddings.append(chunk_emb)

        # Calculate similarities
        similarities = [query_emb.similarity_to(chunk) for chunk in chunk_embeddings]

        # All similarities should be valid
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)
        assert len(similarities) == 5
