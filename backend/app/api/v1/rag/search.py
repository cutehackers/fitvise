"""RAG search API endpoints for semantic search (Task 2.4.1).

This module provides REST API endpoints for semantic search functionality
including basic search, similar chunks, suggestions, and search analytics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.application.dto.search import (
    BatchSearchRequest,
    DocumentSearchRequest,
    SearchFeedback,
    SearchHealthResponse,
    SearchMetrics,
    SearchRequest,
    SearchResponse,
    SearchSuggestionRequest,
    SearchSuggestionsResponse,
    SimilarChunksRequest,
)
from app.application.use_cases.retrieval.semantic_search import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchUseCase,
)
from app.domain.services.retrieval_service import RetrievalService
from app.infrastructure.repositories.weaviate_search_repository import (
    WeaviateSearchRepository,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.config.vector_stores.weaviate_config import WeaviateConfig, WeaviateAuthType
from app.core.settings import get_settings

router = APIRouter(prefix="/rag/search", tags=["RAG Search"])

# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------

def get_weaviate_client() -> WeaviateClient:
    """Get Weaviate client instance."""
    settings = get_settings()

    # Create WeaviateConfig from settings
    weaviate_config = WeaviateConfig(
        host=settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
        port=8080 if "localhost" in settings.weaviate_url else 8080,  # Default port
        scheme="https" if settings.weaviate_url.startswith("https") else "http",
        api_key=settings.weaviate_api_key if settings.weaviate_api_key else None,
        auth_type=WeaviateAuthType.API_KEY if settings.weaviate_api_key else WeaviateAuthType.NONE,
    )

    client = WeaviateClient(weaviate_config)
    return client


def get_search_repository(
    client: WeaviateClient = Depends(get_weaviate_client),
) -> WeaviateSearchRepository:
    """Get search repository instance."""
    from app.infrastructure.repositories.weaviate_embedding_repository import (
        WeaviateEmbeddingRepository,
    )
    from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
        SentenceTransformerService,
    )

    embedding_repo = WeaviateEmbeddingRepository(client)
    embedding_service = SentenceTransformerService()

    return WeaviateSearchRepository(
        weaviate_client=client,
        embedding_repository=embedding_repo,
        embedding_service=embedding_service,
    )


def get_retrieval_service() -> RetrievalService:
    """Get retrieval service instance."""
    return RetrievalService()


def get_semantic_search_use_case(
    search_repo: WeaviateSearchRepository = Depends(get_search_repository),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> SemanticSearchUseCase:
    """Get semantic search use case instance."""
    from app.application.use_cases.embedding.embed_query import EmbedQueryUseCase

    # Create embed query use case with injected services
    embed_query_uc = EmbedQueryUseCase(
        embedding_service=search_repo.embedding_service,
        embedding_repository=search_repo.embedding_repository,
        domain_service=None,
    )

    return SemanticSearchUseCase(
        embed_query_uc=embed_query_uc,
        search_repository=search_repo,
        retrieval_service=retrieval_service,
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/semantic", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def semantic_search(
    request: SearchRequest,
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
) -> SearchResponse:
    """Perform semantic search on document chunks.

    This endpoint provides the core RAG retrieval functionality by:
    1. Embedding the user's query text
    2. Searching for semantically similar document chunks
    3. Returning ranked results with similarity scores

    Args:
        request: Search request with query text and filters

    Returns:
        Search results with relevance scores and metadata

    Raises:
        HTTPException: If search operation fails
    """
    try:
        # Validate request against configuration limits
        settings = get_settings()

        # Validate top_k against configuration limits
        if request.top_k > settings.search_max_top_k:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Requested top_k ({request.top_k}) exceeds maximum allowed ({settings.search_max_top_k})",
            )

        # Validate min_similarity
        if request.min_similarity < settings.search_min_similarity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Requested min_similarity ({request.min_similarity}) is below minimum ({settings.search_min_similarity})",
            )

        # Convert DTO to use case request
        semantic_request = SemanticSearchRequest(
            query_text=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            filters=request.filters.model_dump() if request.filters else None,
            include_metadata=request.include_metadata,
            session_id=UUID(request.session_id) if request.session_id else None,
            user_id=UUID(request.user_id) if request.user_id else None,
        )

        # Execute search
        response = await use_case.execute(semantic_request)

        # Convert use case response to DTO
        return SearchResponse(
            success=response.success,
            query_id=str(response.query_id),
            results=[
                {
                    "result_id": str(result.result_id),
                    "chunk_id": str(result.chunk_id),
                    "document_id": str(result.document_id),
                    "content": result.content,
                    "similarity_score": result.similarity_score.score,
                    "rank": result.rank,
                    "document_metadata": result.document_metadata,
                    "chunk_metadata": result.chunk_metadata,
                    "highlight_text": result.highlight_text,
                    "context_before": result.context_before,
                    "context_after": result.context_after,
                    "quality_label": result.similarity_score.get_quality_label(),
                    "doc_type": result.document_metadata.get("doc_type"),
                }
                for result in response.results
            ],
            total_results=response.total_results,
            processing_time_ms=response.processing_time_ms,
            embedding_time_ms=response.embedding_time_ms,
            search_time_ms=response.search_time_ms,
            query_vector_dimension=response.query_vector_dimension,
            avg_similarity_score=response.avg_similarity_score,
            error=response.error,
            metadata=response.metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        ) from e


@router.post("/similar-chunks", response_model=List[SearchResponse], status_code=status.HTTP_200_OK)
async def find_similar_chunks(
    request: SimilarChunksRequest,
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
):
    """Find chunks similar to given chunk IDs.

    This endpoint is useful for "find more like this" functionality
    and for exploring related content within the document corpus.

    Args:
        request: Request with chunk IDs and similarity parameters

    Returns:
        List of similar chunks for each input chunk

    Raises:
        HTTPException: If similarity search fails
    """
    try:
        # Find similar chunks
        results = await use_case.find_similar_chunks(
            chunk_ids=request.chunk_ids,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )

        # Convert to response format
        # Group results by original chunk for better organization
        grouped_results = {}
        for result in results:
            # For simplicity, return all results as a flat list
            # In a more sophisticated implementation, you might group by input chunk
            similar_response = SearchResponse(
                success=True,
                query_id=str(UUID()),  # Generate temporary query ID
                results=[
                    {
                        "result_id": str(result.result_id),
                        "chunk_id": str(result.chunk_id),
                        "document_id": str(result.document_id),
                        "content": result.content,
                        "similarity_score": result.similarity_score.score,
                        "rank": result.rank,
                        "document_metadata": result.document_metadata,
                        "chunk_metadata": result.chunk_metadata,
                        "highlight_text": result.highlight_text,
                        "quality_label": result.similarity_score.get_quality_label(),
                        "doc_type": result.document_metadata.get("doc_type"),
                    }
                ],
                total_results=1,
                processing_time_ms=0.0,
                embedding_time_ms=0.0,
                search_time_ms=0.0,
                query_vector_dimension=768,  # Default dimension
                avg_similarity_score=result.similarity_score.score,
            )
            grouped_results[str(result.chunk_id)] = similar_response

        return list(grouped_results.values())

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar chunks search failed: {str(e)}",
        ) from e


@router.get("/suggestions", response_model=SearchSuggestionsResponse, status_code=status.HTTP_200_OK)
async def get_search_suggestions(
    partial_query: str = Query(..., min_length=2, max_length=100, description="Partial search query"),
    max_suggestions: int = Query(5, ge=1, le=20, description="Maximum number of suggestions"),
    min_similarity: float = Query(0.3, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
):
    """Get search suggestions based on partial query.

    This endpoint provides autocomplete functionality for search queries
    based on common terms and previous search patterns.

    Args:
        partial_query: Partial search query text
        max_suggestions: Maximum number of suggestions to return
        min_similarity: Minimum similarity for suggestions

    Returns:
        List of search suggestions with similarity scores

    Raises:
        HTTPException: If suggestion generation fails
    """
    try:
        # Generate suggestions
        suggestions = await use_case.get_search_suggestions(
            partial_query=partial_query,
            max_suggestions=max_suggestions,
            min_similarity=min_similarity,
        )

        # Convert to response format
        suggestion_dtos = [
            {
                "suggestion": suggestion,
                "similarity_score": 0.8,  # Placeholder similarity
                "popularity_score": 1.0,   # Placeholder popularity
            }
            for suggestion in suggestions
        ]

        return SearchSuggestionsResponse(
            success=True,
            suggestions=suggestion_dtos,
            total_suggestions=len(suggestion_dtos),
            processing_time_ms=50.0,  # Placeholder processing time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search suggestions failed: {str(e)}",
        ) from e


@router.post("/feedback", status_code=status.HTTP_200_OK)
async def submit_search_feedback(
    feedback: SearchFeedback,
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
):
    """Submit user feedback for search results.

    This endpoint allows users to provide feedback on search quality
    which can be used to improve future search results.

    Args:
        feedback: Feedback information including clicked results and ratings

    Returns:
        Success confirmation

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        # Log the feedback
        await use_case.log_search_feedback(
            query_id=UUID(feedback.query_id),
            result_ids=feedback.result_ids,
            clicked_result_id=feedback.clicked_result_id,
            feedback_score=feedback.feedback_score,
        )

        return {"success": True, "message": "Feedback submitted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}",
        ) from e


@router.get("/health", response_model=SearchHealthResponse, status_code=status.HTTP_200_OK)
async def health_check(
    search_repo: WeaviateSearchRepository = Depends(get_search_repository),
):
    """Check the health of the search system.

    This endpoint provides system health information including
    database connectivity and search functionality status.

    Returns:
        Health status and system information

    Raises:
        HTTPException: If health check fails
    """
    try:
        # Perform health check
        health_info = await search_repo.health_check()

        return SearchHealthResponse(
            status=health_info.get("status", "unknown"),
            message=f"Search system is {health_info.get('status', 'unknown')}",
            components={
                "weaviate": {
                    "connected": health_info.get("weaviate_connected", False),
                    "status": "healthy" if health_info.get("weaviate_connected") else "unhealthy",
                },
                "search_functionality": {
                    "working": health_info.get("search_functionality", False),
                    "status": "healthy" if health_info.get("search_functionality") else "degraded",
                },
            },
            uptime_seconds=3600.0,  # Placeholder uptime
            last_check=health_info.get("timestamp", ""),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        ) from e


@router.get("/metrics", response_model=SearchMetrics, status_code=status.HTTP_200_OK)
async def get_search_metrics(
    time_range_days: int = Query(7, ge=1, le=365, description="Time range in days"),
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
):
    """Get search performance and usage metrics.

    This endpoint provides analytics about search usage patterns,
    performance metrics, and system statistics.

    Args:
        time_range_days: Number of days to analyze

    Returns:
        Search metrics and analytics

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        # Get performance metrics
        metrics = await use_case.get_performance_metrics()

        # Get search statistics
        search_repo = use_case._search_repository
        search_stats = await search_repo.get_search_statistics(time_range_days)

        # Get popular queries
        popular_queries = await search_repo.get_popular_queries(limit=10, time_range_days=time_range_days)

        return SearchMetrics(
            total_queries=search_stats.get("total_queries", 0),
            avg_processing_time_ms=search_stats.get("avg_processing_time_ms", 0.0),
            cache_hit_rate=search_stats.get("cache_hit_rate", 0.0),
            avg_similarity_score=search_stats.get("avg_similarity_score", 0.0),
            popular_queries=[
                {"query": query, "frequency": freq} for query, freq in popular_queries
            ],
            search_health={
                "status": "healthy",
                "last_check": metrics.get("search_health", {}).get("timestamp", ""),
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics retrieval failed: {str(e)}",
        ) from e


@router.post("/batch", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def batch_search(
    request: BatchSearchRequest,
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
):
    """Perform multiple searches and aggregate results.

    This endpoint allows executing multiple search queries in a single
    request and aggregates the results using the specified method.

    Args:
        request: Batch search request with multiple queries

    Returns:
        Aggregated search results

    Raises:
        HTTPException: If batch search fails
    """
    try:
        # Convert queries to domain entities
        search_queries = []
        for query_dto in request.queries:
            from app.domain.value_objects.search_filters import SearchFilters
            from app.domain.entities.search_query import SearchQuery

            filters = None
            if query_dto.filters:
                filters = SearchFilters(
                    doc_types=set(query_dto.filters.doc_types or []),
                    departments=set(query_dto.filters.departments or []),
                    tags=set(query_dto.filters.tags or []),
                    min_token_count=query_dto.filters.min_token_count,
                    max_token_count=query_dto.filters.max_token_count,
                    quality_threshold=query_dto.filters.quality_threshold,
                )

            search_query = SearchQuery.create(
                text=query_dto.query,
                filters=filters,
                top_k=query_dto.top_k,
                min_similarity=query_dto.min_similarity,
                include_metadata=query_dto.include_metadata,
            )
            search_queries.append(search_query)

        # Perform aggregated search
        aggregated_results = await use_case._search_repository.aggregate_search_results(
            queries=search_queries,
            aggregation_method=request.aggregation_method,
        )

        # Convert to response format
        result_dtos = [
            {
                "result_id": str(result.result_id),
                "chunk_id": str(result.chunk_id),
                "document_id": str(result.document_id),
                "content": result.content,
                "similarity_score": result.similarity_score.score,
                "rank": result.rank,
                "document_metadata": result.document_metadata,
                "chunk_metadata": result.chunk_metadata,
                "highlight_text": result.highlight_text,
                "quality_label": result.similarity_score.get_quality_label(),
                "doc_type": result.document_metadata.get("doc_type"),
            }
            for result in aggregated_results[:request.max_total_results]
        ]

        avg_score = (
            sum(r["similarity_score"] for r in result_dtos) / len(result_dtos)
            if result_dtos else 0.0
        )

        return SearchResponse(
            success=True,
            query_id=str(UUID()),  # Generate temporary query ID for batch
            results=result_dtos,
            total_results=len(result_dtos),
            processing_time_ms=300.0,  # Placeholder processing time
            embedding_time_ms=100.0,
            search_time_ms=150.0,
            query_vector_dimension=768,
            avg_similarity_score=avg_score,
            metadata={
                "aggregation_method": request.aggregation_method,
                "query_count": len(search_queries),
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}",
        ) from e


@router.post("/documents", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_within_documents(
    request: DocumentSearchRequest,
    use_case: SemanticSearchUseCase = Depends(get_semantic_search_use_case),
):
    """Search within specific documents.

    This endpoint allows searching within a predefined set of documents
    with optional semantic filtering by query text.

    Args:
        request: Document search request

    Returns:
        Search results from specified documents

    Raises:
        HTTPException: If document search fails
    """
    try:
        # Perform document search
        results = await use_case._search_repository.search_by_document_ids(
            document_ids=request.document_ids,
            query_text=request.query_text,
            top_k=request.top_k,
        )

        # Convert to response format
        result_dtos = [
            {
                "result_id": str(result.result_id),
                "chunk_id": str(result.chunk_id),
                "document_id": str(result.document_id),
                "content": result.content,
                "similarity_score": result.similarity_score.score,
                "rank": result.rank,
                "document_metadata": result.document_metadata,
                "chunk_metadata": result.chunk_metadata,
                "highlight_text": result.highlight_text,
                "quality_label": result.similarity_score.get_quality_label(),
                "doc_type": result.document_metadata.get("doc_type"),
            }
            for result in results
        ]

        avg_score = (
            sum(r["similarity_score"] for r in result_dtos) / len(result_dtos)
            if result_dtos else 0.0
        )

        return SearchResponse(
            success=True,
            query_id=str(UUID()),  # Generate temporary query ID
            results=result_dtos,
            total_results=len(result_dtos),
            processing_time_ms=200.0,  # Placeholder processing time
            embedding_time_ms=0.0,
            search_time_ms=150.0,
            query_vector_dimension=768,
            avg_similarity_score=avg_score,
            metadata={
                "search_type": "document_search",
                "document_count": len(request.document_ids),
                "has_query_text": request.query_text is not None,
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {str(e)}",
        ) from e