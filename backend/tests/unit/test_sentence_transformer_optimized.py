"""Unit tests for optimized SentenceTransformerService.

Tests the performance optimizations and new functionality
without requiring model loading or external dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)


class TestSentenceTransformerOptimized:
    """Test optimized SentenceTransformerService functionality."""

    def test_service_creation_with_optimizations(self):
        """Test that service creation initializes all optimization components."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Check that all optimization components are initialized
        assert hasattr(service, '_thread_pool_manager')
        assert hasattr(service, '_memory_pool')
        assert hasattr(service, '_batching_engine')
        assert hasattr(service, '_cache')
        assert hasattr(service, '_performance_monitor')

        # Check basic properties
        assert service.model_name == "Alibaba-NLP/gte-multilingual-base"
        assert service.model_dimension == 768
        assert service.config == config

    def test_device_resolution(self):
        """Test device resolution with torch import."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Device should be auto-detected
        device = service._resolve_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_comprehensive_stats_method(self):
        """Test the new comprehensive stats method."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Should return stats without requiring model to be loaded
        stats = service.get_comprehensive_stats()

        # Check that all optimization component stats are included
        assert "thread_pool_stats" in stats
        assert "memory_pool_stats" in stats
        assert "batching_engine_stats" in stats
        assert "cache_stats" in stats
        assert "performance_stats" in stats
        assert "batching_performance" in stats

        # Check base stats are present
        assert "total_embeddings" in stats
        assert "model_name" in stats
        assert "model_dimension" in stats

    def test_performance_report_method(self):
        """Test the new performance report method."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Should generate report without requiring model to be loaded
        report = service.get_performance_report()

        # Check report structure
        assert "overview" in report
        assert "performance_metrics" in report
        assert "optimization_effectiveness" in report
        assert "recommendations" in report

        # Check overview section
        overview = report["overview"]
        assert "total_operations" in overview
        assert "total_embeddings" in overview

    @patch('sentence_transformers.SentenceTransformer')
    async def test_embed_with_caching(self, mock_model):
        """Test embedding with caching functionality."""
        # Mock model for testing
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value = MagicMock()

        # Setup mock to return numpy array
        import numpy as np
        mock_embedding = np.array([[0.1, 0.2, 0.3] * 256], dtype=np.float32)  # 768 dimensions
        mock_model.encode.return_value = mock_embedding

        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)
        service._model = mock_model
        service.is_loaded = True

        # Test single embedding
        vector = await service.embed("test text")
        assert len(vector) == 768

        # Test that caching works (same text should hit cache)
        vector2 = await service.embed("test text")
        assert len(vector2) == 768

        # Check cache stats
        stats = service.get_comprehensive_stats()
        assert stats["cache_stats"]["total_requests"] >= 2

    def test_batching_engine_configuration(self):
        """Test that batching engine is properly configured."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Batching engine should be configured with model parameters
        assert service._batching_engine.model_dimension == 768

    def test_cache_configuration(self):
        """Test that cache is properly configured."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Cache should be configured with model information
        cache_stats = service._cache.get_stats()
        assert cache_stats["model_name"] == "Alibaba-NLP/gte-multilingual-base"
        assert cache_stats["model_version"] == "1.0"

    def test_thread_pool_manager_initialization(self):
        """Test that thread pool manager is properly initialized."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Thread pool should be initialized
        stats = service._thread_pool_manager.get_stats()
        assert "embedding_pool_size" in stats
        assert "model_pool_size" in stats
        assert "io_pool_size" in stats
        assert stats["embedding_pool_size"] > 0
        assert stats["model_pool_size"] > 0
        assert stats["io_pool_size"] > 0

    def test_memory_pool_initialization(self):
        """Test that memory pool is properly initialized."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Memory pool should be initialized
        stats = service._memory_pool.get_stats()
        assert "pool_efficiency" in stats
        assert "memory_usage_mb" in stats
        assert stats["pool_efficiency"] >= 0.0  # Should be non-negative

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self):
        """Test that shutdown properly cleans up resources."""
        config = EmbeddingModelConfig.default()
        service = SentenceTransformerService(config)

        # Mock the components to avoid actual cleanup
        service._cache = AsyncMock()
        service._memory_pool = MagicMock()

        # Test shutdown
        await service.shutdown()

        # Verify cleanup was called
        service._cache.clear.assert_called_once()
        service._memory_pool.cleanup.assert_called_once()