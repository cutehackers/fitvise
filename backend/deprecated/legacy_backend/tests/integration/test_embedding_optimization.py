#!/usr/bin/env python3
"""Integration test for optimized embedding service performance.

This script tests the complete optimization stack including:
- Thread pool management
- Memory pooling and zero-copy operations
- Adaptive batching
- High-performance caching
- Performance monitoring
"""

import asyncio
import logging
import sys
import time
from typing import List

import pytest

# Add the project root to path
sys.path.insert(0, '/Users/junhyounglee/workspace/fitvise/backend')

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    create_sentence_transformer_service,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
async def service():
    """Create and initialize optimized embedding service for testing."""
    logger.info("Initializing optimized embedding service...")
    config = EmbeddingModelConfig.default()
    service = create_sentence_transformer_service(config)

    # Initialize model
    logger.info("Loading embedding model...")
    await service.initialize()
    logger.info("Model loaded successfully")

    yield service

    # Cleanup
    logger.info("Shutting down service...")
    await service.shutdown()
    logger.info("Service shutdown complete")


async def test_basic_functionality(service):
    """Test basic embedding functionality."""
    logger.info("Testing basic functionality...")

    # Test single embedding
    start_time = time.perf_counter()
    vector = await service.embed("Hello world, this is a test.")
    single_time = (time.perf_counter() - start_time) * 1000

    logger.info(f"Single embedding completed in {single_time:.2f}ms")
    logger.info(f"Vector dimension: {len(vector)}")
    assert len(vector) == 768, f"Expected 768 dimensions, got {len(vector)}"

    # Test caching by embedding same text again
    start_time = time.perf_counter()
    cached_vector = await service.embed("Hello world, this is a test.")
    cached_time = (time.perf_counter() - start_time) * 1000

    logger.info(f"Cached embedding completed in {cached_time:.2f}ms")
    logger.info(f"Cache speedup: {single_time / cached_time:.1f}x")
    # Allow some tolerance for cache timing - cache should generally be faster but timing can vary
    assert cached_time <= single_time * 1.2, "Cached embedding should be faster or comparable"


async def test_adaptive_batching(service):
    """Test adaptive batching with different text sizes."""
    logger.info("Testing adaptive batching...")

    # Generate test texts of varying lengths
    test_texts = [
        "Short text",
        "This is a medium length text with more content to process.",
        "This is a much longer text that contains multiple sentences and should require more processing time and memory resources to embed properly using the sentence transformers model.",
        "Brief",
        "Another medium sized text for testing the adaptive batching functionality with some additional words here and there.",
        "The quick brown fox jumps over the lazy dog. This pangram sentence contains all letters of the English alphabet and is commonly used for testing font rendering and text processing capabilities.",
        "Tiny",
        "Medium length text about artificial intelligence and machine learning applications in modern technology.",
    ] * 10  # 80 texts total

    logger.info(f"Testing batch embedding with {len(test_texts)} texts")

    start_time = time.perf_counter()
    vectors = await service.embed_batch(test_texts)
    batch_time = (time.perf_counter() - start_time) * 1000

    logger.info(f"Batch embedding completed in {batch_time:.2f}ms")
    logger.info(f"Throughput: {len(test_texts) / (batch_time / 1000):.1f} texts/second")
    logger.info(f"Average time per text: {batch_time / len(test_texts):.2f}ms")

    assert len(vectors) == len(test_texts), "Batch should return same number of vectors"
    assert all(len(v) == 768 for v in vectors), "All vectors should have correct dimensions"

    # Test performance summary
    stats = service.get_comprehensive_stats()
    logger.info(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.1%}")
    logger.info(f"Memory pool efficiency: {stats['memory_pool_stats']['pool_efficiency']:.1%}")


async def test_concurrent_operations(service):
    """Test concurrent embedding operations."""
    logger.info("Testing concurrent operations...")

    async def embed_worker(worker_id: int, texts: List[str]) -> List:
        """Worker function for concurrent embedding."""
        results = []
        for i, text in enumerate(texts):
            start = time.perf_counter()
            vector = await service.embed(f"Worker {worker_id}: {text}")
            elapsed = (time.perf_counter() - start) * 1000
            results.append((worker_id, i, elapsed))
        return results

    # Create multiple concurrent workers
    worker_texts = [
        ["concurrent test 1", "concurrent test 2", "concurrent test 3"],
        ["concurrent test 4", "concurrent test 5", "concurrent test 6"],
        ["concurrent test 7", "concurrent test 8", "concurrent test 9"],
    ]

    start_time = time.perf_counter()
    tasks = [
        embed_worker(i, texts)
        for i, texts in enumerate(worker_texts)
    ]
    results = await asyncio.gather(*tasks)
    total_time = (time.perf_counter() - start_time) * 1000

    total_embeddings = sum(len(worker_result) for worker_result in results)
    avg_time = sum(elapsed for worker_result in results for _, _, elapsed in worker_result) / total_embeddings

    logger.info(f"Concurrent operations completed in {total_time:.2f}ms")
    logger.info(f"Total embeddings: {total_embeddings}")
    logger.info(f"Average time per embedding: {avg_time:.2f}ms")
    logger.info(f"Concurrent throughput: {total_embeddings / (total_time / 1000):.1f} embeddings/second")

    # Check thread pool stats
    stats = service.get_comprehensive_stats()
    logger.info(f"Thread pool stats: {stats['thread_pool_stats']}")


async def test_performance_monitoring(service):
    """Test performance monitoring and reporting."""
    logger.info("Testing performance monitoring...")

    # Generate some activity for monitoring
    await service.embed("Test text for performance monitoring")
    await service.embed_batch(["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"])

    # Get comprehensive performance report
    report = service.get_performance_report()

    logger.info("Performance Report Summary:")
    logger.info(f"  Total operations: {report['overview']['total_operations']}")
    logger.info(f"  Total embeddings: {report['overview']['total_embeddings']}")
    logger.info(f"  Uptime: {report['overview']['uptime_hours']:.2f} hours")
    logger.info(f"  Cache hit rate: {report['optimization_effectiveness']['cache_hit_rate']:.1%}")
    logger.info(f"  Memory pool efficiency: {report['optimization_effectiveness']['memory_pool_efficiency']:.1%}")

    if report['performance_metrics']['recent_5min']:
        recent = report['performance_metrics']['recent_5min']
        logger.info(f"  Recent throughput: {recent['throughput_ops_per_second']:.1f} ops/sec")
        if 'success_rate' in recent:
            logger.info(f"  Recent success rate: {recent['success_rate']:.1%}")

    logger.info("Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        logger.info(f"  {i}. {rec}")

    # Basic validation that the report contains expected fields
    assert "overview" in report
    assert "performance_metrics" in report
    assert "optimization_effectiveness" in report
    assert "recommendations" in report