"""Search API models for request/response validation (Task 2.4.1).

This module defines Pydantic models for search API request/response
validation and serialization following the project's style guide.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class SearchFilter(BaseModel):
    """Search filter criteria model."""

    doc_types: Optional[List[str]] = Field(None, description="Document types to include")
    exclude_doc_types: Optional[List[str]] = Field(None, description="Document types to exclude")
    source_types: Optional[List[str]] = Field(None, description="Source types to filter by")
    departments: Optional[List[str]] = Field(None, description="Departments to filter by")
    categories: Optional[List[str]] = Field(None, description="Categories to filter by")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    min_token_count: Optional[int] = Field(None, ge=1, description="Minimum token count")
    max_token_count: Optional[int] = Field(None, ge=1, description="Maximum token count")
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum quality score")
    language_codes: Optional[List[str]] = Field(None, description="Language codes to filter by")

    @field_validator('max_token_count')
    @classmethod
    def validate_token_counts(cls, v, info):  # type: ignore[override]
        """Validate token count range."""
        if v is not None and 'min_token_count' in info.data and info.data['min_token_count'] is not None:
            if v < info.data['min_token_count']:
                raise ValueError('max_token_count cannot be less than min_token_count')
        return v


class SearchRequest(BaseModel):
    """Semantic search request model."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    filters: Optional[SearchFilter] = Field(None, description="Search filters")
    include_metadata: bool = Field(True, description="Whether to include full metadata")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):  # type: ignore[override]
        """Validate query text."""
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result model."""

    result_id: str = Field(..., description="Unique result identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Result content text")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Result rank")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    highlight_text: Optional[str] = Field(None, description="Highlighted text snippet")
    context_before: Optional[str] = Field(None, description="Text before main content")
    context_after: Optional[str] = Field(None, description="Text after main content")
    quality_label: Optional[str] = Field(None, description="Quality assessment label")
    doc_type: Optional[str] = Field(None, description="Document type")

    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )


class SearchResponse(BaseModel):
    """Semantic search response model."""

    success: bool = Field(..., description="Whether search was successful")
    query_id: str = Field(..., description="Unique query identifier")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results")
    processing_time_ms: float = Field(..., ge=0.0, description="Total processing time")
    embedding_time_ms: float = Field(..., ge=0.0, description="Query embedding time")
    search_time_ms: float = Field(..., ge=0.0, description="Search execution time")
    query_vector_dimension: int = Field(..., ge=0, description="Query embedding dimension")
    avg_similarity_score: float = Field(..., ge=0.0, le=1.0, description="Average similarity score")
    error: Optional[str] = Field(None, description="Error message if search failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SimilarChunksRequest(BaseModel):
    """Find similar chunks request model."""

    chunk_ids: List[str] = Field(..., min_length=1, max_length=50, description="Chunk IDs to find similar items for")
    top_k: int = Field(10, ge=1, le=100, description="Maximum results per chunk")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")

    @field_validator('chunk_ids')
    @classmethod
    def validate_chunk_ids(cls, v):  # type: ignore[override]
        """Validate chunk IDs."""
        if not v:
            raise ValueError('At least one chunk ID must be provided')
        unique_ids = list(set(v))  # Remove duplicates
        if len(unique_ids) != len(v):
            raise ValueError('Duplicate chunk IDs are not allowed')
        return unique_ids


class SearchSuggestionRequest(BaseModel):
    """Search suggestions request model."""

    partial_query: str = Field(..., min_length=2, max_length=100, description="Partial search query")
    max_suggestions: int = Field(5, ge=1, le=20, description="Maximum number of suggestions")
    min_similarity: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity for suggestions")

    @field_validator('partial_query')
    @classmethod
    def validate_partial_query(cls, v):  # type: ignore[override]
        """Validate partial query."""
        if not v or not v.strip():
            raise ValueError('Partial query cannot be empty')
        return v.strip()


class SearchSuggestion(BaseModel):
    """Individual search suggestion model."""

    suggestion: str = Field(..., description="Suggested query text")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity to partial query")
    popularity_score: Optional[float] = Field(None, ge=0.0, description="Usage frequency score")


class SearchSuggestionsResponse(BaseModel):
    """Search suggestions response model."""

    success: bool = Field(..., description="Whether suggestion generation was successful")
    suggestions: List[SearchSuggestion] = Field(default_factory=list, description="Search suggestions")
    total_suggestions: int = Field(..., ge=0, description="Total number of suggestions")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class SearchFeedback(BaseModel):
    """Search feedback submission model."""

    query_id: str = Field(..., description="Search query identifier")
    result_ids: List[str] = Field(..., min_length=1, description="Returned result IDs")
    clicked_result_id: Optional[str] = Field(None, description="ID of result user clicked")
    feedback_score: Optional[int] = Field(None, ge=1, le=5, description="User feedback score (1-5)")
    feedback_text: Optional[str] = Field(None, max_length=500, description="Text feedback")

    @field_validator('clicked_result_id')
    @classmethod
    def validate_clicked_result(cls, v, info):  # type: ignore[override]
        """Validate clicked result ID."""
        if v is not None and 'result_ids' in info.data and v not in info.data['result_ids']:
            raise ValueError('clicked_result_id must be one of the returned result_ids')
        return v


class SearchMetrics(BaseModel):
    """Search performance metrics model."""

    total_queries: int = Field(..., ge=0, description="Total number of queries")
    avg_processing_time_ms: float = Field(..., ge=0.0, description="Average processing time")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    avg_similarity_score: float = Field(..., ge=0.0, le=1.0, description="Average similarity score")
    popular_queries: List[Dict[str, Any]] = Field(default_factory=list, description="Popular search queries")
    search_health: Dict[str, Any] = Field(default_factory=dict, description="Search system health")


class SearchHealthResponse(BaseModel):
    """Search health check response model."""

    status: str = Field(..., description="Health status: healthy, degraded, or unhealthy")
    message: str = Field(..., description="Health status message")
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Component health details")
    uptime_seconds: float = Field(..., ge=0.0, description="System uptime in seconds")
    last_check: str = Field(..., description="Last health check timestamp")


class BatchSearchRequest(BaseModel):
    """Batch search request model."""

    queries: List[SearchRequest] = Field(..., min_length=1, max_length=10, description="Search queries to execute")
    aggregation_method: str = Field("reciprocal_rank", description="Method for combining results")
    max_total_results: int = Field(50, ge=1, le=200, description="Maximum total results across all queries")

    @field_validator('queries')
    @classmethod
    def validate_queries(cls, v):  # type: ignore[override]
        """Validate query list."""
        if not v:
            raise ValueError('At least one query must be provided')
        return v

    @field_validator('aggregation_method')
    @classmethod
    def validate_aggregation_method(cls, v):  # type: ignore[override]
        """Validate aggregation method."""
        allowed_methods = ['reciprocal_rank', 'average_score', 'max_score']
        if v not in allowed_methods:
            raise ValueError(f'aggregation_method must be one of: {allowed_methods}')
        return v


class DocumentSearchRequest(BaseModel):
    """Search within specific documents request model."""

    document_ids: List[str] = Field(..., min_length=1, max_length=100, description="Document IDs to search within")
    query_text: Optional[str] = Field(None, min_length=1, max_length=1000, description="Optional query text")
    top_k: int = Field(50, ge=1, le=200, description="Maximum results per document")

    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v):  # type: ignore[override]
        """Validate document IDs."""
        if not v:
            raise ValueError('At least one document ID must be provided')
        unique_ids = list(set(v))
        if len(unique_ids) != len(v):
            raise ValueError('Duplicate document IDs are not allowed')
        return unique_ids