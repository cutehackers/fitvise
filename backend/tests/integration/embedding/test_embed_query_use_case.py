"""Integration tests for EmbedQueryUseCase (Task 2.2.1).

Tests cover:
- Query embedding with caching
- Performance metrics tracking
- Error handling and recovery
- Integration with repository and service layers
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from app.application.use_cases.embedding.embed_query import (
    EmbedQueryRequest,
    EmbedQueryUseCase,
)
from app.domain.entities.embedding import Embedding
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = AsyncMock()
    service.model_name = "Alibaba-NLP/gte-multilingual-base"
    service.model_dimension = 768
    service.is_loaded = True

    # Mock embed_query to return a vector
    async def mock_embed_query(query: str, use_cache: bool = True):
        return EmbeddingVector.from_numpy(np.random.rand(768).astype(np.float32))

    service.embed_query = mock_embed_query

    # Mock cache stats
    async def mock_get_cache_stats():
        return {
            "cache_size": 10,
            "cache_hits": 5,
            "cache_misses": 3,
            "hit_rate": 0.625,
            "total_embeddings": 8,
        }

    service.get_cache_stats = mock_get_cache_stats

    return service


@pytest.fixture
def mock_repository():
    """Mock embedding repository for testing."""
    repository = AsyncMock()

    # Mock save to succeed
    async def mock_save(embedding: Embedding):
        pass

    repository.save = mock_save

    return repository


@pytest.fixture
def mock_domain_service():
    """Mock domain service for testing."""
    service = AsyncMock()
    return service


@pytest.fixture
def use_case(mock_embedding_service, mock_repository, mock_domain_service):
    """Create use case with mocked dependencies."""
    return EmbedQueryUseCase(
        embedding_service=mock_embedding_service,
        embedding_repository=mock_repository,
        domain_service=mock_domain_service,
    )


class TestEmbedQueryBasicOperation:
    """Test basic query embedding operation."""

    @pytest.mark.asyncio
    async def test_embed_simple_query(self, use_case):
        """Test embedding a simple query successfully."""
        request = EmbedQueryRequest(query="What exercises for lower back pain?")

        response = await use_case.execute(request)

        assert response.success is True
        assert response.query_id is not None
        assert response.vector_dimension == 768
        assert response.processing_time_ms >= 0
        assert response.error is None

    @pytest.mark.asyncio
    async def test_embed_query_without_storage(self, use_case):
        """Test embedding query without storing to database."""
        request = EmbedQueryRequest(
            query="Test query",
            store_embedding=False,
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.stored is False
        assert response.embedding_id is None

    @pytest.mark.asyncio
    async def test_embed_query_with_storage(self, use_case, mock_repository):
        """Test embedding query with storage enabled."""
        request = EmbedQueryRequest(
            query="Test query",
            store_embedding=True,
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.stored is True
        assert response.embedding_id is not None
        # Verify repository save was called
        mock_repository.save.assert_called_once()


class TestEmbedQueryCaching:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_query_with_cache_enabled(self, use_case, mock_embedding_service):
        """Test query embedding with cache enabled."""
        request = EmbedQueryRequest(
            query="Cached query",
            use_cache=True,
        )

        response = await use_case.execute(request)

        assert response.success is True
        # Processing time should be relatively fast (heuristic for cache)
        # Note: In real scenario, cache hit would be < 50ms

    @pytest.mark.asyncio
    async def test_query_with_cache_disabled(self, use_case):
        """Test query embedding with cache disabled."""
        request = EmbedQueryRequest(
            query="Non-cached query",
            use_cache=False,
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.cache_hit is False

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, use_case):
        """Test retrieving performance metrics."""
        metrics = await use_case.get_performance_metrics()

        assert "cache_size" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert "hit_rate" in metrics
        assert "total_queries" in metrics
        assert metrics["hit_rate"] == 0.625


class TestEmbedQueryMetadata:
    """Test metadata handling."""

    @pytest.mark.asyncio
    async def test_embed_query_with_metadata(self, use_case, mock_repository):
        """Test embedding query with custom metadata."""
        custom_meta = {
            "user_id": "user_123",
            "session_id": "session_456",
            "context": "fitness",
        }

        request = EmbedQueryRequest(
            query="Test query",
            query_metadata=custom_meta,
            store_embedding=True,
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.stored is True

        # Verify metadata was included in saved embedding
        call_args = mock_repository.save.call_args
        saved_embedding = call_args[0][0]
        assert saved_embedding.metadata["user_id"] == "user_123"
        assert saved_embedding.metadata["session_id"] == "session_456"
        assert saved_embedding.metadata["context"] == "fitness"

    @pytest.mark.asyncio
    async def test_embed_query_stores_processing_time(self, use_case, mock_repository):
        """Test that processing time is stored in metadata."""
        request = EmbedQueryRequest(
            query="Test query",
            store_embedding=True,
        )

        response = await use_case.execute(request)

        # Verify processing time in metadata
        call_args = mock_repository.save.call_args
        saved_embedding = call_args[0][0]
        assert "processing_time_ms" in saved_embedding.metadata
        assert saved_embedding.metadata["processing_time_ms"] >= 0


class TestEmbedQueryValidation:
    """Test input validation and error handling."""

    @pytest.mark.asyncio
    async def test_embed_empty_query(self, use_case):
        """Test that empty query returns error."""
        request = EmbedQueryRequest(query="")

        response = await use_case.execute(request)

        assert response.success is False
        assert response.error == "Empty query text"
        assert response.vector_dimension == 0

    @pytest.mark.asyncio
    async def test_embed_whitespace_query(self, use_case):
        """Test that whitespace-only query returns error."""
        request = EmbedQueryRequest(query="   \n\t  ")

        response = await use_case.execute(request)

        assert response.success is False
        assert response.error == "Empty query text"

    @pytest.mark.asyncio
    async def test_embedding_generation_error(self, use_case, mock_embedding_service):
        """Test handling of embedding generation errors."""
        from app.domain.exceptions.embedding_exceptions import EmbeddingGenerationError

        # Mock service to raise error
        async def mock_error(query: str, use_cache: bool = True):
            raise EmbeddingGenerationError(message="Model failed")

        mock_embedding_service.embed_query = mock_error

        request = EmbedQueryRequest(query="Test query")

        response = await use_case.execute(request)

        assert response.success is False
        assert "Failed to generate query embedding" in response.error
        assert response.processing_time_ms > 0


class TestEmbedQueryModelConfiguration:
    """Test model configuration handling."""

    @pytest.mark.asyncio
    async def test_embed_with_custom_model(self, use_case):
        """Test embedding with custom model name."""
        request = EmbedQueryRequest(
            query="Test query",
            model_name="custom/model",
            model_version="2.0",
        )

        response = await use_case.execute(request)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_embed_with_default_model(self, use_case):
        """Test embedding with default model configuration."""
        request = EmbedQueryRequest(query="Test query")

        response = await use_case.execute(request)

        assert response.success is True
        # Default model name should be used


class TestEmbedQueryStorageFailure:
    """Test storage failure handling."""

    @pytest.mark.asyncio
    async def test_storage_failure_non_critical(self, use_case, mock_repository):
        """Test that storage failure doesn't fail the entire operation."""
        # Mock repository to raise error
        async def mock_save_error(embedding):
            raise Exception("Storage failed")

        mock_repository.save = mock_save_error

        request = EmbedQueryRequest(
            query="Test query",
            store_embedding=True,
        )

        response = await use_case.execute(request)

        # Embedding should succeed even if storage fails
        assert response.success is True
        assert response.stored is False
        assert response.embedding_id is None
        assert response.vector_dimension == 768


class TestEmbedQueryPerformanceMetrics:
    """Test performance metrics tracking."""

    @pytest.mark.asyncio
    async def test_processing_time_tracked(self, use_case):
        """Test that processing time is accurately tracked."""
        request = EmbedQueryRequest(query="Test query")

        response = await use_case.execute(request)

        assert response.processing_time_ms >= 0
        assert response.processing_time_ms < 10000  # Should be under 10 seconds

    @pytest.mark.asyncio
    async def test_cache_hit_heuristic(self, use_case):
        """Test cache hit detection heuristic."""
        request = EmbedQueryRequest(
            query="Fast query",
            use_cache=True,
        )

        response = await use_case.execute(request)

        # If processing time < 50ms, likely a cache hit
        if response.processing_time_ms < 50:
            assert response.cache_hit is True


class TestEmbedQueryRealisticScenarios:
    """Test realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_embed_fitness_query(self, use_case):
        """Test embedding a realistic fitness query."""
        request = EmbedQueryRequest(
            query="What are the best exercises for lower back pain relief?",
            model_name="Alibaba-NLP/gte-multilingual-base",
            use_cache=True,
            store_embedding=True,
            query_metadata={
                "user_id": "user_123",
                "category": "fitness",
                "intent": "exercise_recommendation",
            },
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.vector_dimension == 768
        assert response.query_id is not None

    @pytest.mark.asyncio
    async def test_embed_multiple_queries_sequence(self, use_case):
        """Test embedding multiple queries in sequence."""
        queries = [
            "Lower back pain exercises",
            "Core strengthening routine",
            "Stretching for flexibility",
        ]

        responses = []
        for query_text in queries:
            request = EmbedQueryRequest(query=query_text, use_cache=True)
            response = await use_case.execute(request)
            responses.append(response)

        # All should succeed
        assert all(r.success for r in responses)
        # All should have unique query IDs
        query_ids = [r.query_id for r in responses]
        assert len(set(query_ids)) == 3

    @pytest.mark.asyncio
    async def test_embed_long_query(self, use_case):
        """Test embedding a long query (approaching limits)."""
        long_query = " ".join(["exercise"] * 100)  # ~100 words

        request = EmbedQueryRequest(query=long_query)

        response = await use_case.execute(request)

        assert response.success is True
        assert response.vector_dimension == 768
