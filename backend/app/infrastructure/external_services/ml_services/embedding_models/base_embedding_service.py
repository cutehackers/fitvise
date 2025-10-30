"""Base embedding service infrastructure (Task 2.2.1).

This module defines the abstract base class for all embedding services,
providing common interface and utilities for model loading, caching,
and error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.domain.exceptions.embedding_exceptions import (
    EmbeddingGenerationError,
    ModelLoadError,
)
from app.domain.value_objects.embedding_vector import EmbeddingVector

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services (Task 2.2.1).

    Defines common interface for embedding generation with support for
    batch and real-time processing, model management, and performance tracking.

    Attributes:
        config: Embedding model configuration
        model_name: Name of the embedding model
        model_dimension: Embedding vector dimension
        is_loaded: Whether model is loaded and ready

    Examples:
        >>> config = EmbeddingModelConfig.default()
        >>> service = SentenceTransformerService(config)
        >>> await service.initialize()
        >>> service.is_loaded
        True

        >>> # Single embedding
        >>> vector = await service.embed("Hello world")
        >>> len(vector)
        768

        >>> # Batch embedding
        >>> vectors = await service.embed_batch(["text1", "text2", "text3"])
        >>> len(vectors)
        3
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        """Initialize base embedding service.

        Args:
            config: Embedding model configuration
        """
        self.config = config
        self.model_name = config.model_name
        self.model_dimension = config.model_dimension
        self.is_loaded = False
        self._model: Optional[Any] = None
        self._cache: Dict[str, EmbeddingVector] = {}
        self._stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize and load the embedding model.

        Raises:
            ModelLoadError: If model initialization fails
        """
        pass

    @abstractmethod
    async def embed(
        self,
        text: str,
        normalize: Optional[bool] = None,
    ) -> EmbeddingVector:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            normalize: Whether to L2-normalize (defaults to config)

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to embed
            batch_size: Batch size (defaults to config)
            show_progress: Show progress bar (defaults to config)
            normalize: Whether to L2-normalize (defaults to config)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingGenerationError: If batch embedding fails
        """
        pass

    async def embed_query(
        self,
        query: str,
        use_cache: bool = True,
    ) -> EmbeddingVector:
        """Generate embedding optimized for query (real-time).

        Args:
            query: Query text
            use_cache: Whether to use cache for repeated queries

        Returns:
            Query embedding vector

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        # Check cache if enabled
        if use_cache and query in self._cache:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._cache[query]

        self._stats["cache_misses"] += 1

        # Generate embedding
        try:
            vector = await self.embed(query)

            # Cache if enabled
            if use_cache and self.config.cache_strategy.value != "none":
                self._cache[query] = vector
                logger.debug(f"Cached embedding for query: {query[:50]}...")

            return vector

        except Exception as e:
            self._stats["errors"] += 1
            raise EmbeddingGenerationError(
                message=f"Failed to embed query: {query[:50]}...",
                model_name=self.model_name,
                details=str(e),
            ) from e

    async def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[EmbeddingVector]:
        """Generate embeddings optimized for documents (batch processing).

        Args:
            texts: List of document texts
            batch_size: Batch size (defaults to config)

        Returns:
            List of document embedding vectors

        Raises:
            EmbeddingGenerationError: If batch embedding fails
        """
        return await self.embed_batch(
            texts=texts,
            batch_size=batch_size,
            show_progress=True,
        )

    def validate_model_loaded(self) -> None:
        """Validate that model is loaded.

        Raises:
            ModelLoadError: If model is not loaded
        """
        if not self.is_loaded or self._model is None:
            raise ModelLoadError(
                message="Model not loaded. Call initialize() first.",
                model_name=self.model_name,
            )

    def clear_cache(self) -> int:
        """Clear embedding cache.

        Returns:
            Number of cached items cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cached embeddings")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Dictionary with service statistics
        """
        cache_hit_rate = (
            self._stats["cache_hits"]
            / (self._stats["cache_hits"] + self._stats["cache_misses"])
            if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
            else 0.0
        )

        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": cache_hit_rate,
            "model_name": self.model_name,
            "model_dimension": self.model_dimension,
            "is_loaded": self.is_loaded,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary
        """
        health = {
            "service": self.__class__.__name__,
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "status": "healthy" if self.is_loaded else "not_initialized",
        }

        # Try a test embedding if loaded
        if self.is_loaded:
            try:
                test_vector = await self.embed("health check test")
                health["test_embedding_dim"] = len(test_vector)
                health["test_passed"] = len(test_vector) == self.model_dimension
            except Exception as e:
                health["status"] = "unhealthy"
                health["error"] = str(e)

        return health

    def __repr__(self) -> str:
        """String representation of service."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"dim={self.model_dimension}, "
            f"loaded={self.is_loaded})"
        )
