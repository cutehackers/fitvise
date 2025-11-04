"""Sentence-Transformers embedding service (Task 2.2.1).

This module implements the concrete embedding service using Sentence-Transformers
with Alibaba-NLP/gte-multilingual-base model, supporting batch and real-time embedding with
CPU optimization and multilingual support.

Performance Optimizations:
- Dedicated thread pools for concurrent operations
- Zero-copy vector conversion with memory pooling
- Adaptive batching for optimal throughput
- High-performance caching with collision resistance
- Comprehensive performance monitoring
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
import numpy as np
import time
import torch

from app.config.ml_models.embedding_model_configs import (
    DeviceType,
    EmbeddingModelConfig,
)
from app.domain.exceptions.embedding_exceptions import (
    EmbeddingGenerationError,
    ModelLoadError,
)
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.base_embedding_service import (
    BaseEmbeddingService,
)
from app.infrastructure.external_services.ml_services.embedding_models.thread_pool_manager import (
    get_thread_pool_manager,
)
from app.infrastructure.external_services.ml_services.embedding_models.memory_pool import (
    get_memory_pool,
)
from app.infrastructure.external_services.ml_services.embedding_models.adaptive_batching import (
    AdaptiveBatchingEngine,
)
from app.infrastructure.external_services.ml_services.embedding_models.embedding_cache import (
    EmbeddingCache,
)
from app.infrastructure.external_services.ml_services.embedding_models.performance_monitor import (
    get_global_monitor,
)

logger = logging.getLogger(__name__)


class SentenceTransformerService(BaseEmbeddingService):
    """Sentence-Transformers embedding service implementation (Task 2.2.1).

    Implements embedding generation using Sentence-Transformers library
    with Alibaba-NLP/gte-multilingual-base model (768 dimensions), optimized for CPU.

    Performance Optimizations:
        - Thread Pool Management: Dedicated pools for different operation types
        - Memory Pooling: Zero-copy vector conversion with pre-allocated memory
        - Adaptive Batching: Dynamic batch sizing based on system constraints
        - Intelligent Caching: Collision-resistant caching with LRU eviction
        - Performance Monitoring: Real-time metrics and automated alerting

    Performance Targets:
        - Batch: ≥1000 chunks/minute
        - Real-time: <100ms latency
        - CPU-optimized with thread pooling
        - Memory: 50% reduction through pooling
        - Concurrency: 40-50% improvement

    Examples:
        >>> config = EmbeddingModelConfig.default()
        >>> service = SentenceTransformerService(config)
        >>> await service.initialize()
        >>> service.is_loaded
        True

        >>> # High-performance single embedding
        >>> vector = await service.embed("Hello world")
        >>> len(vector)
        768

        >>> # Optimized batch embedding
        >>> texts = ["text1", "text2", "text3"]
        >>> vectors = await service.embed_batch(texts)
        >>> len(vectors)
        3
        >>> all(len(v) == 768 for v in vectors)
        True
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        """Initialize Sentence-Transformers service with performance optimizations.

        Args:
            config: Embedding model configuration
        """
        super().__init__(config)
        self._device = self._resolve_device()

        # Performance optimization components
        self._thread_pool_manager = get_thread_pool_manager()
        self._memory_pool = get_memory_pool()
        self._batching_engine = AdaptiveBatchingEngine()
        self._cache = EmbeddingCache(max_size=10000, max_memory_mb=512)
        self._performance_monitor = get_global_monitor()

        # Configure cache with model information
        self._cache.set_model_info(self.model_name, "1.0")

        # Configure batching engine with model parameters
        self._batching_engine.update_model_parameters(
            dimension=self.model_dimension,
            base_memory_mb=0.5  # Will be updated after model loading
        )

        logger.info(f"Optimized embedding service initialized for {self.model_name}")

    def _resolve_device(self) -> str:
        """Resolve target compute device.

        Returns:
            Device string ("cpu", "cuda", "mps")
        """
        if self.config.device == DeviceType.AUTO:
            # Auto-detect best available device
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        return self.config.device.value

    async def initialize(self) -> None:
        """Initialize and load Sentence-Transformers model with performance optimization.

        Raises:
            ModelLoadError: If model loading fails
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_name} already loaded")
            return

        async with self._performance_monitor.measure_operation("model_initialization"):
            try:
                logger.info(
                    f"Loading Sentence-Transformers model: {self.model_name} "
                    f"on device: {self._device}"
                )

                # Import here to avoid loading at module level
                from sentence_transformers import SentenceTransformer

                # Load model in dedicated model thread pool
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    self._thread_pool_manager.model_pool,
                    lambda: SentenceTransformer(
                        self.model_name,
                        device=self._device,
                        trust_remote_code=True,
                        **self.config.model_kwargs,
                    ),
                )

                # Verify model dimension with minimal testing
                test_embedding = await loop.run_in_executor(
                    self._thread_pool_manager.model_pool,
                    lambda: self._model.encode(
                        ["test"],
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        batch_size=1,
                        show_progress_bar=False,
                    )
                )
                actual_dim = test_embedding.shape[1]

                if actual_dim != self.model_dimension:
                    logger.warning(
                        f"Model dimension mismatch: expected {self.model_dimension}, "
                        f"got {actual_dim}. Updating config."
                    )
                    self.model_dimension = actual_dim

                # Update batching engine with actual model parameters
                self._batching_engine.update_model_parameters(
                    dimension=self.model_dimension,
                    base_memory_mb=max(0.1, self.model_dimension * 4 / (1024 * 1024))
                )

                self.is_loaded = True
                logger.info(
                    f"Successfully loaded {self.model_name} "
                    f"(dim={self.model_dimension}, device={self._device})"
                )

            except ImportError as e:
                raise ModelLoadError(
                    message="sentence-transformers not installed",
                    model_name=self.model_name,
                    details=str(e),
                ) from e
            except Exception as e:
                raise ModelLoadError(
                    message=f"Failed to load model {self.model_name}",
                    model_name=self.model_name,
                    details=str(e),
                ) from e

    async def embed(
        self,
        text: str,
        normalize: Optional[bool] = None,
    ) -> EmbeddingVector:
        """Generate embedding for a single text with caching optimization.

        Args:
            text: Input text to embed
            normalize: Whether to L2-normalize (defaults to config)

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        self.validate_model_loaded()

        if not text or not text.strip():
            raise EmbeddingGenerationError(
                message="Cannot embed empty text",
                model_name=self.model_name,
            )

        # Check cache first
        cached_vector = await self._cache.get(text, normalize or False)
        if cached_vector is not None:
            return cached_vector

        should_normalize = (
            normalize if normalize is not None else self.config.normalize_embeddings
        )

        # Generate embedding with performance monitoring
        async with self._performance_monitor.measure_operation("single_embedding", 1):
            try:
                loop = asyncio.get_event_loop()
                embedding_array = await loop.run_in_executor(
                    self._thread_pool_manager.embedding_pool,
                    lambda: self._model.encode(
                        [text],
                        convert_to_numpy=True,
                        normalize_embeddings=should_normalize,
                        batch_size=1,
                        show_progress_bar=False,
                        **self.config.encode_kwargs,
                    ),
                )

                # Zero-copy vector creation with memory pooling
                vectors = EmbeddingVector.from_model_output_pooled(
                    embedding_array,
                    self._memory_pool
                )
                vector = vectors[0]

                # Cache the result
                await self._cache.put(text, vector, should_normalize)

                self._stats["total_embeddings"] += 1
                return vector

            except Exception as e:
                self._stats["errors"] += 1
                raise EmbeddingGenerationError(
                    message=f"Failed to embed text: {text[:50]}...",
                    model_name=self.model_name,
                    details=str(e),
                ) from e

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts with adaptive batching optimization.

        Args:
            texts: List of input texts to embed
            batch_size: Batch size (uses adaptive sizing if None)
            show_progress: Show progress bar (defaults to config)
            normalize: Whether to L2-normalize (defaults to config)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingGenerationError: If batch embedding fails
        """
        self.validate_model_loaded()

        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise EmbeddingGenerationError(
                message="All texts are empty",
                model_name=self.model_name,
            )

        should_normalize = (
            normalize if normalize is not None else self.config.normalize_embeddings
        )
        should_show_progress = (
            show_progress if show_progress is not None else self.config.show_progress
        )

        # Get optimal batch size recommendation
        if batch_size is None:
            recommendation = self._batching_engine.calculate_optimal_batch_size(valid_texts)
            effective_batch_size = recommendation.optimal_batch_size
            logger.info(
                f"Adaptive batching: using size {effective_batch_size} - "
                f"estimated throughput: {recommendation.estimated_throughput:.1f} items/sec"
            )
        else:
            effective_batch_size = batch_size

        # Process in optimized batches
        start_time = time.perf_counter()
        all_vectors = []

        async with self._performance_monitor.measure_operation("batch_embedding", len(valid_texts)):
            try:
                # Check cache for each text
                uncached_texts = []
                uncached_indices = []
                cache_results = {}

                for i, text in enumerate(valid_texts):
                    cached_vector = await self._cache.get(text, should_normalize)
                    if cached_vector is not None:
                        cache_results[i] = cached_vector
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)

                if uncached_texts:
                    logger.info(
                        f"Processing {len(uncached_texts)} uncached texts in batches of {effective_batch_size}"
                    )

                    # Process uncached texts in batches
                    loop = asyncio.get_event_loop()
                    embeddings_array = await loop.run_in_executor(
                        self._thread_pool_manager.embedding_pool,
                        lambda: self._model.encode(
                            uncached_texts,
                            convert_to_numpy=True,
                            normalize_embeddings=should_normalize,
                            batch_size=effective_batch_size,
                            show_progress_bar=should_show_progress,
                            **self.config.encode_kwargs,
                        ),
                    )

                    # Zero-copy vector creation with memory pooling
                    uncached_vectors = EmbeddingVector.from_model_output_pooled(
                        embeddings_array,
                        self._memory_pool
                    )

                    # Cache results and organize final output
                    for text, vector, original_index in zip(uncached_texts, uncached_vectors, uncached_indices):
                        await self._cache.put(text, vector, should_normalize)
                        cache_results[original_index] = vector

                # Assemble final results in original order
                all_vectors = [cache_results[i] for i in range(len(valid_texts))]

                # Record batch performance
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                memory_usage_mb = self._memory_pool.get_stats()["memory_usage_mb"]

                self._batching_engine.record_batch_performance(
                    batch_size=effective_batch_size,
                    processing_time_ms=processing_time_ms,
                    memory_usage_mb=memory_usage_mb,
                    success_count=len(all_vectors),
                    total_count=len(valid_texts),
                    texts=valid_texts
                )

                self._stats["total_embeddings"] += len(all_vectors)
                logger.info(f"Successfully embedded {len(all_vectors)} texts in {processing_time_ms:.1f}ms")

                return all_vectors

            except Exception as e:
                self._stats["errors"] += 1
                raise EmbeddingGenerationError(
                    message=f"Failed to embed batch of {len(texts)} texts",
                    model_name=self.model_name,
                    details=str(e),
                ) from e

    async def shutdown(self) -> None:
        """Shutdown service and release resources with comprehensive cleanup."""
        if self._model is not None:
            # Clear model from memory
            del self._model
            self._model = None
            self.is_loaded = False
            logger.info(f"Shutdown {self.model_name} service")

        # Cleanup optimized components
        self._memory_pool.cleanup()
        await self._cache.clear()

        logger.info("Optimized embedding service shutdown complete")

    def get_stats(self) -> dict:
        """Get service statistics with custom cache support.

        Returns:
            Dictionary with service statistics
        """
        cache_stats = self._cache.get_stats()
        cache_hit_rate = (
            self._stats["cache_hits"]
            / (self._stats["cache_hits"] + self._stats["cache_misses"])
            if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
            else 0.0
        )

        return {
            **self._stats,
            "cache_size": cache_stats.get("cache_size", 0),
            "cache_hit_rate": cache_hit_rate,
            "model_name": self.model_name,
            "model_dimension": self.model_dimension,
            "is_loaded": self.is_loaded,
        }

    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive service statistics including all optimization components.

        Returns:
            Comprehensive statistics dictionary
        """
        # Get base stats
        base_stats = self.get_stats()
        cache_stats = self._cache.get_stats()

        return {
            **base_stats,
            "thread_pool_stats": self._thread_pool_manager.get_stats(),
            "memory_pool_stats": self._memory_pool.get_stats(),
            "batching_engine_stats": self._batching_engine.get_stats(),
            "cache_stats": cache_stats,
            "performance_stats": self._performance_monitor.get_current_metrics(),
            "batching_performance": self._batching_engine.get_performance_summary(),
        }

    def get_performance_report(self) -> dict:
        """Get detailed performance report for optimization analysis.

        Returns:
            Performance report with insights and recommendations
        """
        summary = self._performance_monitor.get_performance_summary()
        cache_stats = self._cache.get_stats()
        memory_stats = self._memory_pool.get_stats()

        report = {
            "overview": {
                "total_operations": summary["current"]["total_operations"],
                "total_embeddings": summary["current"]["total_embeddings"],
                "uptime_hours": summary["uptime_hours"],
                "avg_embeddings_per_operation": summary["avg_embeddings_per_operation"],
            },
            "performance_metrics": {
                "current_processing_time_ms": summary["current"]["rolling_averages"]["processing_time_ms"],
                "current_memory_mb": summary["current"]["current_memory_mb"],
                "current_cpu_percent": summary["current"]["current_cpu_percent"],
                "recent_5min": summary["recent_5min"],
                "performance_trend": summary.get("performance_trend_ms"),
                "performance_status": summary.get("performance_status", "unknown"),
            },
            "optimization_effectiveness": {
                "cache_hit_rate": cache_stats["hit_rate"],
                "cache_utilization_percent": cache_stats["utilization_percent"],
                "memory_pool_efficiency": memory_stats["pool_efficiency"],
                "thread_pool_utilization": {
                    pool: stats["active"] / stats["size"] * 100
                    for pool, stats in summary["current"]["rolling_averages"].items()
                    if "pool" in pool
                },
            },
            "alerts": {
                "total_alerts_triggered": summary["alerts_triggered"],
                "active_alerts": {
                    name: alert for name, alert in summary["current"]["alerts"].items()
                    if alert["cooldown_remaining"] == 0
                }
            },
            "recommendations": self._generate_performance_recommendations(summary, cache_stats, memory_stats)
        }

        return report

    def _generate_performance_recommendations(
        self,
        summary: dict,
        cache_stats: dict,
        memory_stats: dict
    ) -> List[str]:
        """Generate performance optimization recommendations.

        Args:
            summary: Performance summary
            cache_stats: Cache statistics
            memory_stats: Memory pool statistics

        Returns:
            List of performance recommendations
        """
        recommendations = []

        # Cache recommendations
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache size for better hit rates")
        if cache_stats["memory_utilization_percent"] > 80:
            recommendations.append("Cache memory usage is high, consider increasing memory limits")

        # Memory pool recommendations
        if memory_stats["pool_efficiency"] < 0.7:
            recommendations.append("Memory pool efficiency is low, check allocation patterns")
        if memory_stats["memory_usage_mb"] > 400:
            recommendations.append("Memory usage is high, consider more aggressive cleanup")

        # Performance trend recommendations
        trend = summary.get("performance_trend_ms")
        if trend and trend > 100:
            recommendations.append("Processing time is increasing, investigate bottlenecks")
        elif trend and trend < -100:
            recommendations.append("Performance is improving, current optimizations are effective")

        # Throughput recommendations
        recent = summary.get("recent_5min")
        if recent and recent["throughput_ops_per_second"] < 20:
            recommendations.append("Low throughput detected, consider increasing batch sizes")

        if not recommendations:
            recommendations.append("Performance is optimal, no recommendations needed")

        return recommendations


def create_sentence_transformer_service(
    config: Optional[EmbeddingModelConfig] = None,
) -> SentenceTransformerService:
    """Create Sentence-Transformers embedding service.

    Args:
        config: Optional embedding model configuration

    Returns:
        SentenceTransformerService instance
    """
    if config is None:
        config = EmbeddingModelConfig.default()

    return SentenceTransformerService(config)
