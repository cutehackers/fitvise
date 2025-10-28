"""Embed query use case (Task 2.2.1).

This use case generates embeddings for user queries with real-time
optimization and caching for sub-100ms latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from app.domain.entities.embedding import Embedding
from app.domain.exceptions.embedding_exceptions import EmbeddingGenerationError
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)


@dataclass
class EmbedQueryRequest:
    """Request to embed a user query.

    Optimized for real-time processing with <100ms target latency.
    """

    query: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_version: str = "1.0"
    use_cache: bool = True
    store_embedding: bool = False
    query_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedQueryResponse:
    """Response from embedding a query."""

    success: bool
    query_id: UUID
    embedding_id: Optional[UUID] = None
    vector_dimension: int = 0
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    stored: bool = False
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "query_id": str(self.query_id),
            "embedding_id": str(self.embedding_id) if self.embedding_id else None,
            "vector_dimension": self.vector_dimension,
            "processing_time_ms": self.processing_time_ms,
            "cache_hit": self.cache_hit,
            "stored": self.stored,
            "error": self.error,
        }


class EmbedQueryUseCase:
    """Use case for embedding user queries (Task 2.2.1).

    Generates embeddings for user queries with real-time optimization,
    caching for repeated queries, and optional storage for retrieval history.

    Performance Target: <100ms latency for cache hits, <500ms for cache misses

    Examples:
        >>> use_case = EmbedQueryUseCase(
        ...     embedding_service=embedding_service,
        ...     embedding_repository=repository,
        ...     domain_service=domain_service
        ... )
        >>> request = EmbedQueryRequest(query="What exercises for lower back pain?")
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> response.processing_time_ms < 100  # Cache hit
        True
        >>> response.cache_hit
        True
    """

    def __init__(
        self,
        embedding_service: SentenceTransformerService,
        embedding_repository: EmbeddingRepository,
        domain_service: EmbeddingService,
    ) -> None:
        """Initialize embed query use case.

        Args:
            embedding_service: Service for generating embeddings
            embedding_repository: Repository for storing embeddings
            domain_service: Domain service for coordination
        """
        self._embedding_service = embedding_service
        self._repository = embedding_repository
        self._domain_service = domain_service

    async def execute(self, request: EmbedQueryRequest) -> EmbedQueryResponse:
        """Execute query embedding with real-time optimization.

        Args:
            request: Embed query request

        Returns:
            Response with embedding results and performance metrics
        """
        query_id = uuid4()
        start_time = datetime.now()

        # Validate query
        if not request.query or not request.query.strip():
            return EmbedQueryResponse(
                success=False,
                query_id=query_id,
                error="Empty query text",
            )

        try:
            # Step 1: Generate embedding with caching
            # The embed_query method uses internal cache for performance
            vector = await self._embedding_service.embed_query(
                query=request.query,
                use_cache=request.use_cache,
            )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Check if result was from cache
            cache_stats = await self._embedding_service.get_cache_stats()
            cache_hit = processing_time < 50.0  # Heuristic: <50ms indicates cache hit

            # Step 2: Optionally store embedding
            embedding_id = None
            stored = False

            if request.store_embedding:
                try:
                    # Create query embedding entity
                    metadata = {
                        "query_text": request.query,
                        "source_type": "query",
                        "cache_hit": cache_hit,
                        "processing_time_ms": processing_time,
                        **request.query_metadata,
                    }

                    embedding = Embedding.for_query(
                        vector=vector,
                        query_id=query_id,
                        model_name=request.model_name,
                        model_version=request.model_version,
                        metadata=metadata,
                    )

                    # Store in repository
                    await self._repository.save(embedding)
                    embedding_id = embedding.id
                    stored = True

                except Exception as e:
                    # Non-critical failure - embedding was generated successfully
                    # Just couldn't store for history
                    pass

            return EmbedQueryResponse(
                success=True,
                query_id=query_id,
                embedding_id=embedding_id,
                vector_dimension=vector.dimension,
                processing_time_ms=processing_time,
                cache_hit=cache_hit,
                stored=stored,
            )

        except EmbeddingGenerationError as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return EmbedQueryResponse(
                success=False,
                query_id=query_id,
                processing_time_ms=processing_time,
                error=f"Failed to generate query embedding: {str(e)}",
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return EmbedQueryResponse(
                success=False,
                query_id=query_id,
                processing_time_ms=processing_time,
                error=f"Unexpected error: {str(e)}",
            )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for query embeddings.

        Returns:
            Dictionary with performance statistics
        """
        cache_stats = await self._embedding_service.get_cache_stats()

        return {
            "cache_size": cache_stats.get("cache_size", 0),
            "cache_hits": cache_stats.get("cache_hits", 0),
            "cache_misses": cache_stats.get("cache_misses", 0),
            "hit_rate": cache_stats.get("hit_rate", 0.0),
            "total_queries": cache_stats.get("total_embeddings", 0),
        }
