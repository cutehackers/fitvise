"""Semantic search use case for RAG retrieval (Task 2.4.1).

This module implements the core semantic search functionality that
integrates query embedding, Weaviate similarity search, and result
processing to deliver relevant document chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.application.use_cases.embedding.embed_query import (
    EmbedQueryRequest,
    EmbedQueryResponse,
    EmbedQueryUseCase,
)
from app.domain.entities.search_query import SearchQuery
from app.domain.entities.search_result import SearchResult
from app.domain.exceptions.retrieval_exceptions import (
    RetrievalError,
    SearchExecutionError,
    SimilaritySearchError,
)
from app.domain.repositories.search_repository import SearchRepository
from app.domain.services.retrieval_service import RetrievalService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.domain.value_objects.similarity_score import SimilarityScore, SimilarityMethod


@dataclass
class SemanticSearchRequest:
    """Request for semantic search operation.

    Encapsulates the search query and configuration options.
    """

    query_text: str
    top_k: int = 10
    min_similarity: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    search_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_search_query(self) -> SearchQuery:
        """Convert to SearchQuery domain entity.

        Returns:
            SearchQuery domain entity
        """
        from app.domain.value_objects.search_filters import SearchFilters

        # Convert dict filters to SearchFilters if provided
        search_filters = None
        if self.filters:
            search_filters = SearchFilters(
                doc_types=set(self.filters.get("doc_types", [])),
                departments=set(self.filters.get("departments", [])),
                tags=set(self.filters.get("tags", [])),
                min_token_count=self.filters.get("min_token_count"),
                max_token_count=self.filters.get("max_token_count"),
                quality_threshold=self.filters.get("quality_threshold"),
            )

        return SearchQuery.create(
            text=self.query_text,
            filters=search_filters,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            include_metadata=self.include_metadata,
            session_id=self.session_id,
            user_id=self.user_id,
            metadata=self.search_metadata,
        )


@dataclass
class SemanticSearchResponse:
    """Response from semantic search operation.

    Contains the search results with metadata and performance metrics.
    """

    success: bool
    query_id: UUID
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    processing_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    query_vector_dimension: int = 0
    avg_similarity_score: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the search response
        """
        return {
            "success": self.success,
            "query_id": str(self.query_id),
            "results": [result.as_summary_dict() for result in self.results],
            "total_results": self.total_results,
            "processing_time_ms": self.processing_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "search_time_ms": self.search_time_ms,
            "query_vector_dimension": self.query_vector_dimension,
            "avg_similarity_score": self.avg_similarity_score,
            "error": self.error,
            "metadata": self.metadata,
        }


class SemanticSearchUseCase:
    """Use case for semantic search in RAG system (Task 2.4.1).

    Implements the core semantic search workflow:
    1. Embed user query using existing EmbedQueryUseCase
    2. Search Weaviate for similar document chunks
    3. Process and rank results with similarity scores
    4. Return formatted search response

    Performance Targets:
    - Query embedding: <100ms (cache hits), <500ms (cache misses)
    - Similarity search: <200ms for typical queries
    - Total processing: <500ms end-to-end

    Examples:
        >>> use_case = SemanticSearchUseCase(
        ...     embed_query_uc=embed_query_uc,
        ...     search_repository=search_repo,
        ...     retrieval_service=retrieval_service
        ... )
        >>> request = SemanticSearchRequest(
        ...     query_text="What exercises help with lower back pain?",
        ...     top_k=5
        ... )
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> len(response.results)
        5
        >>> response.results[0].similarity_score.score > 0.8
        True
    """

    def __init__(
        self,
        embed_query_uc: EmbedQueryUseCase,
        search_repository: SearchRepository,
        retrieval_service: RetrievalService,
    ) -> None:
        """Initialize semantic search use case.

        Args:
            embed_query_uc: Use case for embedding queries
            search_repository: Repository for search operations
            retrieval_service: Domain service for retrieval coordination
        """
        self._embed_query_uc = embed_query_uc
        self._search_repository = search_repository
        self._retrieval_service = retrieval_service

    async def execute(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        """Execute semantic search with end-to-end processing.

        Args:
            request: Semantic search request

        Returns:
            Search response with results and metrics
        """
        start_time = datetime.now()
        query_id = uuid4()

        # Validate request
        if not request.query_text or not request.query_text.strip():
            return SemanticSearchResponse(
                success=False,
                query_id=query_id,
                error="Query text cannot be empty",
            )

        try:
            # Step 1: Convert to domain entity and validate
            search_query = request.to_search_query()
            await self._search_repository.validate_query(search_query)

            # Step 2: Embed the query
            embed_start = datetime.now()
            embed_request = EmbedQueryRequest(
                query=request.query_text,
                use_cache=True,
                store_embedding=False,  # Don't store query embeddings by default
            )
            embed_response = await self._embed_query_uc.execute(embed_request)
            embedding_time = (datetime.now() - embed_start).total_seconds() * 1000

            if not embed_response.success:
                return SemanticSearchResponse(
                    success=False,
                    query_id=query_id,
                    embedding_time_ms=embedding_time,
                    error=f"Query embedding failed: {embed_response.error}",
                )

            # Step 3: Perform similarity search
            search_start = datetime.now()
            raw_results = await self._search_repository.semantic_search(search_query)
            search_time = (datetime.now() - search_start).total_seconds() * 1000

            # Step 4: Process and rank results
            processed_results = await self._retrieval_service.process_search_results(
                raw_results=raw_results,
                query=search_query,
                query_embedding_dimension=embed_response.vector_dimension,
            )

            # Step 5: Calculate metrics and build response
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            avg_score = (
                sum(r.similarity_score.score for r in processed_results) / len(processed_results)
                if processed_results
                else 0.0
            )

            return SemanticSearchResponse(
                success=True,
                query_id=query_id,
                results=processed_results,
                total_results=len(processed_results),
                processing_time_ms=total_time,
                embedding_time_ms=embedding_time,
                search_time_ms=search_time,
                query_vector_dimension=embed_response.vector_dimension,
                avg_similarity_score=avg_score,
                metadata={
                    "cache_hit": embed_response.cache_hit,
                    "query_length": len(request.query_text),
                    "filter_count": len(search_query.filters.as_dict()),
                },
            )

        except RetrievalError as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            return SemanticSearchResponse(
                success=False,
                query_id=query_id,
                processing_time_ms=total_time,
                error=f"Search failed: {str(e)}",
            )

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            return SemanticSearchResponse(
                success=False,
                query_id=query_id,
                processing_time_ms=total_time,
                error=f"Unexpected error: {str(e)}",
            )

    async def find_similar_chunks(
        self,
        chunk_ids: List[str],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Find chunks similar to given chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to find similar items for
            top_k: Maximum number of results per chunk
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar chunks
        """
        try:
            return await self._search_repository.find_similar_chunks(
                chunk_ids=chunk_ids,
                top_k=top_k,
                min_similarity=min_similarity,
            )
        except Exception as e:
            raise SimilaritySearchError(
                message=f"Failed to find similar chunks: {str(e)}",
                details=f"chunk_ids: {chunk_ids[:3]}..., top_k: {top_k}",
            ) from e

    async def get_search_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 5,
        min_similarity: float = 0.3,
    ) -> List[str]:
        """Get search suggestions based on partial query.

        Args:
            partial_query: Partial search query text
            max_suggestions: Maximum number of suggestions
            min_similarity: Minimum similarity for suggestions

        Returns:
            List of suggested query completions
        """
        try:
            if not partial_query or len(partial_query.strip()) < 2:
                return []

            return await self._search_repository.get_search_suggestions(
                partial_query=partial_query.strip(),
                max_suggestions=max_suggestions,
                min_similarity=min_similarity,
            )
        except Exception as e:
            raise SearchExecutionError(
                message=f"Failed to get search suggestions: {str(e)}",
                details=f"partial_query: '{partial_query[:50]}...'",
            ) from e

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for semantic search.

        Returns:
            Dictionary with performance statistics
        """
        try:
            # Get embedding metrics
            embed_metrics = await self._embed_query_uc.get_performance_metrics()

            # Get search repository health and stats
            search_health = await self._search_repository.health_check()
            search_stats = await self._search_repository.get_search_statistics()

            return {
                "embedding_metrics": embed_metrics,
                "search_health": search_health,
                "search_statistics": search_stats,
                "overall_status": "healthy" if search_health.get("status") == "healthy" else "degraded",
            }
        except Exception as e:
            return {
                "error": f"Failed to get performance metrics: {str(e)}",
                "embedding_metrics": {},
                "search_health": {"status": "unknown"},
                "search_statistics": {},
            }

    async def log_search_feedback(
        self,
        query_id: UUID,
        result_ids: List[str],
        clicked_result_id: Optional[str] = None,
        feedback_score: Optional[float] = None,
    ) -> None:
        """Log user interaction feedback for search improvement.

        Args:
            query_id: ID of the search query
            result_ids: IDs of returned results
            clicked_result_id: ID of result user clicked (if any)
            feedback_score: User feedback score (1-5, if provided)
        """
        try:
            await self._search_repository.log_search_interaction(
                query_id=str(query_id),
                result_ids=result_ids,
                clicked_result_id=clicked_result_id,
                feedback_score=feedback_score,
            )
        except Exception as e:
            # Non-critical failure - log but don't raise
            print(f"Warning: Failed to log search feedback: {str(e)}")