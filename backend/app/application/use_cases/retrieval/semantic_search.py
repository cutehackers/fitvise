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
from app.domain.value_objects.similarity_score import SimilarityScore


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
        retrieval_service: RetrievalService,
    ) -> None:
        """Initialize semantic search use case.

        Args:
            retrieval_service: Domain service for complete semantic search orchestration
        """
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
            # Delegate to RetrievalService for complete semantic search workflow
            search_results_with_metrics = await self._retrieval_service.semantic_search_with_metrics(
                query=request.query_text.strip(),
                top_k=request.top_k,
                min_similarity=request.min_similarity,
                filters=request.filters,
                use_cache=True,
                include_metadata=request.include_metadata,
            )

            results = search_results_with_metrics["results"]
            metrics = search_results_with_metrics["metrics"]

            # Calculate additional metrics
            avg_score = (
                sum(r.similarity_score.score for r in results) / len(results)
                if results
                else 0.0
            )

            return SemanticSearchResponse(
                success=True,
                query_id=query_id,
                results=results,
                total_results=len(results),
                processing_time_ms=metrics["total_processing_time_ms"],
                embedding_time_ms=0.0,  # Embedded in total time now
                search_time_ms=metrics["total_processing_time_ms"],  # Single operation now
                query_vector_dimension=0,  # Not tracked at use case level anymore
                avg_similarity_score=avg_score,
                metadata={
                    "cache_hit": metrics["cache_hit"],
                    "query_length": len(request.query_text),
                    "min_similarity_threshold": request.min_similarity,
                    "top_k_requested": request.top_k,
                    "top_k_returned": len(results),
                    "retrieval_metrics": metrics,
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
            return await self._retrieval_service.find_similar_chunks(
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
            return await self._retrieval_service.get_search_suggestions(
                partial_query=partial_query,
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
            # Get retrieval service metrics (includes embedding and search)
            retrieval_metrics = await self._retrieval_service.get_retrieval_metrics()

            return {
                "retrieval_service_metrics": retrieval_metrics,
                "semantic_search_status": "operational",
                "overall_status": retrieval_metrics.get("overall_status", "unknown"),
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
            # This functionality could be moved to the RetrievalService
            # For now, implement a simple logging mechanism
            feedback_data = {
                "query_id": str(query_id),
                "result_ids": result_ids,
                "clicked_result_id": clicked_result_id,
                "feedback_score": feedback_score,
                "timestamp": datetime.utcnow().isoformat(),
            }
            # In a real implementation, this would be stored in a feedback repository
            print(f"Search feedback logged: {feedback_data}")
        except Exception as e:
            # Non-critical failure - log but don't raise
            print(f"Warning: Failed to log search feedback: {str(e)}")