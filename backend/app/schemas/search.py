"""Pydantic schemas for search API validation (Task 2.4.1).

This module provides Pydantic schemas for validating and serializing
search API requests and responses, with comprehensive field validation
and error handling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# Request Schemas

class SearchFilterSchema(BaseModel):
    """Schema for search filter criteria."""

    doc_types: Optional[List[str]] = Field(
        None,
        description="Document types to include in search (e.g., ['pdf', 'docx'])",
        example=["pdf", "docx", "txt"],
    )
    exclude_doc_types: Optional[List[str]] = Field(
        None,
        description="Document types to exclude from search",
        example=["image", "video"],
    )
    source_types: Optional[List[str]] = Field(
        None,
        description="Source types to filter by (e.g., ['file', 'database'])",
        example=["file", "web"],
    )
    departments: Optional[List[str]] = Field(
        None,
        description="Department names to filter by",
        example=["fitness", "nutrition", "physical_therapy"],
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Content categories to filter by",
        example=["exercise", "diet", "wellness"],
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags to filter by",
        example=["beginner", "advanced", "rehabilitation"],
    )
    min_token_count: Optional[int] = Field(
        None,
        ge=1,
        description="Minimum token count for chunks",
        example=50,
    )
    max_token_count: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum token count for chunks",
        example=500,
    )
    quality_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum quality score threshold",
        example=0.7,
    )
    language_codes: Optional[List[str]] = Field(
        None,
        description="Language codes to filter by (e.g., ['en', 'es'])",
        example=["en"],
    )

    @field_validator('max_token_count')
    def validate_token_counts(cls, v, values):  # type: ignore[override]
        """Validate that max_token_count is not less than min_token_count."""
        if v is not None and 'min_token_count' in values and values['min_token_count'] is not None:
            if v < values['min_token_count']:
                raise ValueError('max_token_count cannot be less than min_token_count')
        return v


class SearchRequestSchema(BaseModel):
    """Schema for semantic search request."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text",
        example="What exercises help with lower back pain?",
    )
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
        example=10,
    )
    min_similarity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)",
        example=0.7,
    )
    filters: Optional[SearchFilterSchema] = Field(
        None,
        description="Search filters to apply",
    )
    include_metadata: bool = Field(
        True,
        description="Whether to include full metadata in results",
    )
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for conversation context",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    user_id: Optional[str] = Field(
        None,
        description="User identifier for personalization",
        example="user123",
    )

    @field_validator('query')
    def validate_query(cls, v):  # type: ignore[override]
        """Validate that query text is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()


class SimilarChunksRequestSchema(BaseModel):
    """Schema for finding similar chunks request."""

    chunk_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of chunk IDs to find similar items for",
        example=["550e8400-e29b-41d4-a716-446655440000", "660e8400-e29b-41d4-a716-446655440001"],
    )
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of results per chunk",
        example=5,
    )
    min_similarity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
        example=0.6,
    )

    @field_validator('chunk_ids')
    def validate_chunk_ids(cls, v):  # type: ignore[override]
        """Validate that chunk IDs are unique."""
        if not v:
            raise ValueError('At least one chunk ID must be provided')
        unique_ids = list(set(v))
        if len(unique_ids) != len(v):
            raise ValueError('Duplicate chunk IDs are not allowed')
        return unique_ids


class SearchSuggestionRequestSchema(BaseModel):
    """Schema for search suggestions request."""

    partial_query: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Partial search query for autocomplete",
        example="lower back",
    )
    max_suggestions: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of suggestions to return",
        example=5,
    )
    min_similarity: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for suggestions",
        example=0.4,
    )

    @field_validator('partial_query')
    def validate_partial_query(cls, v):  # type: ignore[override]
        """Validate that partial query is not empty."""
        if not v or not v.strip():
            raise ValueError('Partial query cannot be empty')
        return v.strip()


class SearchFeedbackSchema(BaseModel):
    """Schema for search feedback submission."""

    query_id: str = Field(
        ...,
        description="Search query identifier",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    result_ids: List[str] = Field(
        ...,
        min_items=1,
        description="List of returned result IDs",
        example=["660e8400-e29b-41d4-a716-446655440001", "770e8400-e29b-41d4-a716-446655440002"],
    )
    clicked_result_id: Optional[str] = Field(
        None,
        description="ID of the result the user clicked on",
        example="660e8400-e29b-41d4-a716-446655440001",
    )
    feedback_score: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="User feedback score (1-5, where 5 is excellent)",
        example=4,
    )
    feedback_text: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional text feedback from user",
        example="Results were very relevant and helpful",
    )

    @field_validator('clicked_result_id')
    def validate_clicked_result(cls, v, values):  # type: ignore[override]
        """Validate that clicked result ID is in the returned results."""
        if v is not None and 'result_ids' in values and v not in values['result_ids']:
            raise ValueError('clicked_result_id must be one of the returned result_ids')
        return v


class BatchSearchRequestSchema(BaseModel):
    """Schema for batch search request."""

    queries: List[SearchRequestSchema] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of search queries to execute",
    )
    aggregation_method: str = Field(
        "reciprocal_rank",
        description="Method for combining results",
        example="reciprocal_rank",
    )
    max_total_results: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum total results across all queries",
        example=50,
    )

    @field_validator('queries')
    def validate_queries(cls, v):  # type: ignore[override]
        """Validate that queries list is not empty."""
        if not v:
            raise ValueError('At least one query must be provided')
        return v

    @field_validator('aggregation_method')
    def validate_aggregation_method(cls, v):  # type: ignore[override]
        """Validate that aggregation method is supported."""
        allowed_methods = ['reciprocal_rank', 'average_score', 'max_score']
        if v not in allowed_methods:
            raise ValueError(f'aggregation_method must be one of: {allowed_methods}')
        return v


class DocumentSearchRequestSchema(BaseModel):
    """Schema for searching within specific documents."""

    document_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Document IDs to search within",
        example=["550e8400-e29b-41d4-a716-446655440000", "660e8400-e29b-41d4-a716-446655440001"],
    )
    query_text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=1000,
        description="Optional query text for semantic filtering",
        example="strength training exercises",
    )
    top_k: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum results per document",
        example=20,
    )

    @field_validator('document_ids')
    def validate_document_ids(cls, v):  # type: ignore[override]
        """Validate that document IDs are unique."""
        if not v:
            raise ValueError('At least one document ID must be provided')
        unique_ids = list(set(v))
        if len(unique_ids) != len(v):
            raise ValueError('Duplicate document IDs are not allowed')
        return unique_ids


# Response Schemas

class SearchResultSchema(BaseModel):
    """Schema for individual search result."""

    result_id: str = Field(
        ...,
        description="Unique result identifier",
        example="770e8400-e29b-41d4-a716-446655440003",
    )
    chunk_id: str = Field(
        ...,
        description="Chunk identifier",
        example="880e8400-e29b-41d4-a716-446655440004",
    )
    document_id: str = Field(
        ...,
        description="Source document identifier",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    content: str = Field(
        ...,
        description="Result content text",
        example="Regular plank exercises help strengthen core muscles and support the lower back...",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between query and result",
        example=0.85,
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Result rank in search results",
        example=1,
    )
    document_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata",
        example={
            "doc_type": "pdf",
            "source_type": "file",
            "file_name": "fitness_guide.pdf",
            "created_at": "2024-01-15T10:30:00Z",
        },
    )
    chunk_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata",
        example={
            "sequence": 15,
            "token_count": 125,
            "section": "core_exercises",
        },
    )
    highlight_text: Optional[str] = Field(
        None,
        description="Highlighted text snippet showing relevance",
        example="...<mark>plank exercises</mark> help strengthen <mark>core muscles</mark>...",
    )
    context_before: Optional[str] = Field(
        None,
        description="Text before the main content",
        example="Core strengthening is essential for back health...",
    )
    context_after: Optional[str] = Field(
        None,
        description="Text after the main content",
        example="Hold the plank position for 30-60 seconds...",
    )
    quality_label: Optional[str] = Field(
        None,
        description="Quality assessment label",
        example="very_good",
    )
    doc_type: Optional[str] = Field(
        None,
        description="Document type",
        example="pdf",
    )


class SearchResponseSchema(BaseModel):
    """Schema for search response."""

    success: bool = Field(
        ...,
        description="Whether search was successful",
        example=True,
    )
    query_id: str = Field(
        ...,
        description="Unique query identifier",
        example="990e8400-e29b-41d4-a716-446655440005",
    )
    results: List[SearchResultSchema] = Field(
        default_factory=list,
        description="Search results ranked by relevance",
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Total number of results returned",
        example=10,
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds",
        example=245.5,
    )
    embedding_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Query embedding time in milliseconds",
        example=85.2,
    )
    search_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Search execution time in milliseconds",
        example=120.3,
    )
    query_vector_dimension: int = Field(
        ...,
        ge=0,
        description="Dimension of query embedding vector",
        example=768,
    )
    avg_similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average similarity score across results",
        example=0.78,
    )
    error: Optional[str] = Field(
        None,
        description="Error message if search failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
        example={
            "cache_hit": True,
            "query_length": 35,
            "filter_count": 2,
        },
    )


class SearchSuggestionSchema(BaseModel):
    """Schema for individual search suggestion."""

    suggestion: str = Field(
        ...,
        description="Suggested query text",
        example="lower back pain exercises",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score to partial query",
        example=0.85,
    )
    popularity_score: Optional[float] = Field(
        None,
        ge=0.0,
        description="Usage frequency score",
        example=0.75,
    )


class SearchSuggestionsResponseSchema(BaseModel):
    """Schema for search suggestions response."""

    success: bool = Field(
        ...,
        description="Whether suggestion generation was successful",
        example=True,
    )
    suggestions: List[SearchSuggestionSchema] = Field(
        default_factory=list,
        description="Search suggestions",
    )
    total_suggestions: int = Field(
        ...,
        ge=0,
        description="Total number of suggestions",
        example=5,
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
        example=45.0,
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed",
    )


class SearchHealthResponseSchema(BaseModel):
    """Schema for search health check response."""

    status: str = Field(
        ...,
        description="Health status: healthy, degraded, or unhealthy",
        example="healthy",
    )
    message: str = Field(
        ...,
        description="Health status message",
        example="Search system is healthy",
    )
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Component health details",
        example={
            "weaviate": {"connected": True, "status": "healthy"},
            "search_functionality": {"working": True, "status": "healthy"},
        },
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="System uptime in seconds",
        example=86400.0,
    )
    last_check: str = Field(
        ...,
        description="Last health check timestamp",
        example="2024-01-15T10:30:00Z",
    )


class SearchMetricsSchema(BaseModel):
    """Schema for search performance metrics."""

    total_queries: int = Field(
        ...,
        ge=0,
        description="Total number of queries in time range",
        example=1250,
    )
    avg_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average processing time in milliseconds",
        example=245.5,
    )
    cache_hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cache hit rate",
        example=0.65,
    )
    avg_similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average similarity score",
        example=0.78,
    )
    popular_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most popular search queries",
        example=[
            {"query": "lower back pain exercises", "frequency": 150},
            {"query": "strength training routine", "frequency": 120},
        ],
    )
    search_health: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search system health information",
        example={"status": "healthy", "last_check": "2024-01-15T10:30:00Z"},
    )


# Error Response Schemas

class ErrorSchema(BaseModel):
    """Schema for error responses."""

    error: str = Field(
        ...,
        description="Error message",
        example="Search query cannot be empty",
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code",
        example="INVALID_QUERY",
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details",
        example={"field": "query", "constraint": "min_length=1"},
    )
    timestamp: str = Field(
        ...,
        description="Error timestamp",
        example="2024-01-15T10:30:00Z",
    )