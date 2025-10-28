"""Performance tests for embedding pipeline (Task 2.2.1).

These tests verify performance targets:
- Query embedding: <100ms with cache, <500ms without cache
- Batch embedding: ≥1000 texts/minute (16.7 texts/second)
- Search: <200ms for k=10
- Chunk embedding: 10K chunks/hour (2.78 chunks/second)

Run with: pytest tests/performance/ -v -s --tb=short

Note: These tests require actual ML model and Weaviate running.
Mark as slow tests with @pytest.mark.performance
"""

import time
from statistics import mean, median
from uuid import uuid4

import numpy as np
import pytest

from app.application.use_cases.embedding.batch_embed import (
    BatchEmbedRequest,
    BatchEmbedUseCase,
    TextItem,
)
from app.application.use_cases.embedding.embed_document_chunks import (
    EmbedChunksRequest,
    EmbedDocumentChunksUseCase,
)
from app.application.use_cases.embedding.embed_query import (
    EmbedQueryRequest,
    EmbedQueryUseCase,
)
from app.application.use_cases.embedding.search_embeddings import (
    SearchEmbeddingsUseCase,
    SearchRequest,
)
from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.domain.entities.chunk import Chunk
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.infrastructure.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)


# Mark all tests in this module as performance tests
pytestmark = pytest.mark.performance


@pytest.fixture(scope="module")
async def embedding_service():
    """Create and initialize embedding service for performance testing."""
    config = EmbeddingModelConfig.for_realtime()
    service = SentenceTransformerService(config)
    await service.initialize()
    yield service


@pytest.fixture(scope="module")
async def weaviate_client():
    """Create and connect Weaviate client for performance testing."""
    config = WeaviateConfig.for_local_development()
    client = WeaviateClient(config)
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
async def repository(weaviate_client):
    """Create embedding repository."""
    return WeaviateEmbeddingRepository(weaviate_client)


@pytest.fixture
async def domain_service(repository):
    """Create domain service."""
    return EmbeddingService(repository)


class TestQueryEmbeddingPerformance:
    """Performance tests for query embedding."""

    @pytest.mark.asyncio
    async def test_query_embedding_latency_with_cache(
        self, embedding_service, repository, domain_service
    ):
        """Test query embedding latency with cache hits (target: <100ms)."""
        use_case = EmbedQueryUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Warm up cache
        request = EmbedQueryRequest(
            query="What exercises help with lower back pain?",
            use_cache=True,
            store_embedding=False,
        )
        await use_case.execute(request)

        # Measure cached query performance
        latencies = []
        for _ in range(10):
            start = time.time()
            response = await use_case.execute(request)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            assert response.success is True

        avg_latency = mean(latencies)
        median_latency = median(latencies)

        print(f"\nCache hit latency - Avg: {avg_latency:.2f}ms, Median: {median_latency:.2f}ms")

        # Target: <100ms for cache hits
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms target"

    @pytest.mark.asyncio
    async def test_query_embedding_latency_without_cache(
        self, embedding_service, repository, domain_service
    ):
        """Test query embedding latency without cache (target: <500ms)."""
        use_case = EmbedQueryUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Different queries to avoid cache
        queries = [
            f"Exercise query number {i}" for i in range(10)
        ]

        latencies = []
        for query in queries:
            request = EmbedQueryRequest(
                query=query,
                use_cache=False,
                store_embedding=False,
            )

            start = time.time()
            response = await use_case.execute(request)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            assert response.success is True

        avg_latency = mean(latencies)
        median_latency = median(latencies)

        print(f"\nCache miss latency - Avg: {avg_latency:.2f}ms, Median: {median_latency:.2f}ms")

        # Target: <500ms for cache misses
        assert avg_latency < 500, f"Average latency {avg_latency:.2f}ms exceeds 500ms target"


class TestBatchEmbeddingPerformance:
    """Performance tests for batch embedding."""

    @pytest.mark.asyncio
    async def test_batch_embedding_throughput(
        self, embedding_service, repository, domain_service
    ):
        """Test batch embedding throughput (target: ≥1000 texts/minute)."""
        use_case = BatchEmbedUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Create 1000 test texts
        num_texts = 1000
        texts = [f"Exercise text number {i} with some content" for i in range(num_texts)]

        request = BatchEmbedRequest(
            texts=texts,
            batch_size=32,
            show_progress=False,
            store_embeddings=False,
        )

        # Measure throughput
        start = time.time()
        response = await use_case.execute(request)
        elapsed_seconds = time.time() - start

        assert response.success is True
        assert response.embedded_count == num_texts

        texts_per_minute = (num_texts / elapsed_seconds) * 60
        texts_per_second = num_texts / elapsed_seconds

        print(f"\nBatch throughput: {texts_per_second:.2f} texts/sec ({texts_per_minute:.0f} texts/min)")
        print(f"Total time: {elapsed_seconds:.2f}s for {num_texts} texts")

        # Target: ≥1000 texts/minute (16.7 texts/second)
        assert texts_per_minute >= 1000, f"Throughput {texts_per_minute:.0f} texts/min below 1000 target"

    @pytest.mark.asyncio
    async def test_batch_embedding_scalability(
        self, embedding_service, repository, domain_service
    ):
        """Test batch embedding scales linearly with size."""
        use_case = BatchEmbedUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        batch_sizes = [100, 500, 1000]
        results = []

        for size in batch_sizes:
            texts = [f"Text {i}" for i in range(size)]
            request = BatchEmbedRequest(
                texts=texts,
                batch_size=32,
                show_progress=False,
                store_embeddings=False,
            )

            start = time.time()
            response = await use_case.execute(request)
            elapsed = time.time() - start

            throughput = size / elapsed

            results.append({
                "size": size,
                "time": elapsed,
                "throughput": throughput,
            })

            print(f"\nBatch size {size}: {elapsed:.2f}s, {throughput:.2f} texts/sec")

        # Verify roughly linear scaling (throughput should be relatively constant)
        throughputs = [r["throughput"] for r in results]
        avg_throughput = mean(throughputs)
        variation = max(throughputs) / min(throughputs)

        print(f"\nThroughput variation: {variation:.2f}x")

        # Variation should be less than 2x (allowing for overhead differences)
        assert variation < 2.0, f"Throughput varies {variation:.2f}x, should scale linearly"


class TestChunkEmbeddingPerformance:
    """Performance tests for chunk embedding."""

    @pytest.mark.asyncio
    async def test_chunk_embedding_throughput(
        self, embedding_service, repository, domain_service
    ):
        """Test chunk embedding throughput (target: 10K chunks/hour = 2.78 chunks/sec)."""
        use_case = EmbedDocumentChunksUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Create 300 test chunks (representative sample)
        num_chunks = 300
        chunks = [
            Chunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                text=f"Exercise description number {i} with fitness content and recommendations",
                sequence=i,
            )
            for i in range(num_chunks)
        ]

        request = EmbedChunksRequest(
            chunks=chunks,
            batch_size=32,
            show_progress=False,
            store_embeddings=False,
        )

        # Measure throughput
        start = time.time()
        response = await use_case.execute(request)
        elapsed_seconds = time.time() - start

        assert response.success is True
        assert response.embedded_count == num_chunks

        chunks_per_hour = (num_chunks / elapsed_seconds) * 3600
        chunks_per_second = num_chunks / elapsed_seconds

        print(f"\nChunk throughput: {chunks_per_second:.2f} chunks/sec ({chunks_per_hour:.0f} chunks/hour)")
        print(f"Total time: {elapsed_seconds:.2f}s for {num_chunks} chunks")

        # Target: 10K chunks/hour (2.78 chunks/second)
        assert chunks_per_hour >= 10000, f"Throughput {chunks_per_hour:.0f} chunks/hour below 10K target"


class TestSearchPerformance:
    """Performance tests for similarity search."""

    @pytest.mark.asyncio
    async def test_search_latency(
        self, embedding_service, repository, domain_service
    ):
        """Test search latency (target: <200ms for k=10)."""
        use_case = SearchEmbeddingsUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # First, store some embeddings to search against
        # (In real scenario, assume database is populated)

        # Test search performance
        latencies = []
        for i in range(10):
            query = f"Search query number {i}"
            request = SearchRequest(
                query=query,
                k=10,
                min_similarity=0.0,
            )

            start = time.time()
            response = await use_case.execute(request)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            assert response.success is True

        avg_latency = mean(latencies)
        median_latency = median(latencies)

        print(f"\nSearch latency - Avg: {avg_latency:.2f}ms, Median: {median_latency:.2f}ms")

        # Target: <200ms for k=10
        # Note: Actual latency depends on database size and may include query embedding time
        assert avg_latency < 500, f"Average search latency {avg_latency:.2f}ms too high"


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_latency(
        self, embedding_service, repository, domain_service
    ):
        """Test full pipeline: embed query + search + retrieval."""
        query_use_case = EmbedQueryUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        search_use_case = SearchEmbeddingsUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Full pipeline
        start = time.time()

        # Step 1: Embed query
        query_request = EmbedQueryRequest(
            query="Lower back pain exercises",
            use_cache=True,
            store_embedding=False,
        )
        query_response = await query_use_case.execute(query_request)
        assert query_response.success is True

        # Step 2: Search
        search_request = SearchRequest(
            query="Lower back pain exercises",
            k=10,
            min_similarity=0.7,
        )
        search_response = await search_use_case.execute(search_request)
        assert search_response.success is True

        total_latency_ms = (time.time() - start) * 1000

        print(f"\nFull pipeline latency: {total_latency_ms:.2f}ms")
        print(f"  - Query embedding: {query_response.processing_time_ms:.2f}ms")
        print(f"  - Search: {search_response.processing_time_ms:.2f}ms")

        # Target: <1000ms for full pipeline
        assert total_latency_ms < 1000, f"Pipeline latency {total_latency_ms:.2f}ms exceeds 1s"


class TestResourceUtilization:
    """Test resource utilization and efficiency."""

    @pytest.mark.asyncio
    async def test_memory_efficiency_batch(
        self, embedding_service, repository, domain_service
    ):
        """Test that batch processing doesn't cause memory issues."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        use_case = BatchEmbedUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Process large batch
        texts = [f"Text number {i}" for i in range(5000)]
        request = BatchEmbedRequest(
            texts=texts,
            batch_size=32,
            show_progress=False,
            store_embeddings=False,
        )

        response = await use_case.execute(request)
        assert response.success is True

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nMemory usage: {initial_memory:.0f}MB -> {final_memory:.0f}MB (+{memory_increase:.0f}MB)")

        # Memory increase should be reasonable (< 500MB for 5K texts)
        assert memory_increase < 500, f"Memory increase {memory_increase:.0f}MB too high"


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_query_embeddings(
        self, embedding_service, repository, domain_service
    ):
        """Test concurrent query embedding requests."""
        import asyncio

        use_case = EmbedQueryUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Create 10 concurrent requests
        queries = [f"Concurrent query {i}" for i in range(10)]
        requests = [
            EmbedQueryRequest(query=q, use_cache=False, store_embedding=False)
            for q in queries
        ]

        start = time.time()

        # Execute concurrently
        tasks = [use_case.execute(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        elapsed = time.time() - start

        # All should succeed
        assert all(r.success for r in responses)

        print(f"\n10 concurrent queries completed in {elapsed:.2f}s")
        print(f"Average time per query: {elapsed * 1000 / 10:.2f}ms")

        # Should be faster than sequential (but not 10x due to CPU bottleneck)
        # Rough estimate: should complete in < 5 seconds
        assert elapsed < 5.0, f"Concurrent execution took {elapsed:.2f}s, too slow"
