"""API schemas for embedding endpoints (Task 2.2.1).

Pydantic models for request/response validation and OpenAPI documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Setup Infrastructure Schemas
# ============================================================================


class SetupInfrastructureRequest(BaseModel):
    """Request to setup embedding infrastructure."""

    embedding_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional embedding model configuration",
    )
    weaviate_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Weaviate connection configuration",
    )
    vector_dimension: int = Field(
        default=384,
        ge=1,
        le=4096,
        description="Embedding vector dimension",
    )
    recreate_schema: bool = Field(
        default=False,
        description="Whether to recreate Weaviate schema if exists",
    )


class SetupInfrastructureResponse(BaseModel):
    """Response from setup embedding infrastructure."""

    success: bool = Field(description="Whether setup was successful")
    embedding_service: Dict[str, Any] = Field(
        description="Embedding service status"
    )
    weaviate: Dict[str, Any] = Field(description="Weaviate status")
    schema_created: bool = Field(description="Whether schema was created")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


# ============================================================================
# Embed Document Chunks Schemas
# ============================================================================


class ChunkInput(BaseModel):
    """Input for a single chunk to embed."""

    chunk_id: UUID = Field(description="Unique chunk identifier")
    document_id: Optional[UUID] = Field(default=None, description="Source document ID")
    text: str = Field(min_length=1, max_length=10000, description="Chunk text content")
    sequence: int = Field(default=0, ge=0, description="Chunk sequence number")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata",
    )


class EmbedChunksRequest(BaseModel):
    """Request to embed document chunks."""

    chunks: List[ChunkInput] = Field(
        min_length=1,
        max_length=1000,
        description="Chunks to embed",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    model_version: str = Field(default="1.0", description="Model version")
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for processing",
    )
    show_progress: bool = Field(
        default=True,
        description="Whether to show progress",
    )
    store_embeddings: bool = Field(
        default=True,
        description="Whether to store embeddings in Weaviate",
    )


class ChunkEmbeddingResult(BaseModel):
    """Result for single chunk embedding."""

    chunk_id: UUID = Field(description="Chunk identifier")
    embedding_id: Optional[UUID] = Field(default=None, description="Embedding identifier")
    success: bool = Field(description="Whether embedding was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class EmbedChunksResponse(BaseModel):
    """Response from embedding document chunks."""

    success: bool = Field(description="Overall operation success")
    total_chunks: int = Field(ge=0, description="Total chunks processed")
    embedded_count: int = Field(ge=0, description="Number successfully embedded")
    stored_count: int = Field(ge=0, description="Number stored in Weaviate")
    results: List[ChunkEmbeddingResult] = Field(description="Per-chunk results")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


# ============================================================================
# Embed Query Schemas
# ============================================================================


class EmbedQueryRequest(BaseModel):
    """Request to embed a user query."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Query text to embed",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    model_version: str = Field(default="1.0", description="Model version")
    use_cache: bool = Field(
        default=True,
        description="Whether to use query cache",
    )
    store_embedding: bool = Field(
        default=False,
        description="Whether to store query embedding",
    )
    query_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional query metadata",
    )


class EmbedQueryResponse(BaseModel):
    """Response from embedding a query."""

    success: bool = Field(description="Whether embedding was successful")
    query_id: UUID = Field(description="Query identifier")
    embedding_id: Optional[UUID] = Field(default=None, description="Embedding identifier")
    vector_dimension: int = Field(ge=0, description="Vector dimension")
    processing_time_ms: float = Field(ge=0, description="Processing time in milliseconds")
    cache_hit: bool = Field(description="Whether result was from cache")
    stored: bool = Field(description="Whether embedding was stored")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# Batch Embed Schemas
# ============================================================================


class TextItemInput(BaseModel):
    """Input for a single text item to embed."""

    text: str = Field(
        min_length=1,
        max_length=10000,
        description="Text content",
    )
    item_id: Optional[UUID] = Field(default=None, description="Optional item identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class BatchEmbedRequest(BaseModel):
    """Request for batch embedding operation."""

    texts: Optional[List[str]] = Field(
        default=None,
        max_length=10000,
        description="Simple text list (alternative to text_items)",
    )
    text_items: Optional[List[TextItemInput]] = Field(
        default=None,
        max_length=10000,
        description="Structured text items with metadata",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    model_version: str = Field(default="1.0", description="Model version")
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for processing",
    )
    show_progress: bool = Field(
        default=True,
        description="Whether to show progress",
    )
    store_embeddings: bool = Field(
        default=False,
        description="Whether to store embeddings",
    )
    storage_batch_size: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Batch size for storage operations",
    )

    @field_validator("text_items", "texts")
    @classmethod
    def validate_input(cls, v, info):
        """Ensure at least one input method is provided."""
        texts = info.data.get("texts")
        text_items = info.data.get("text_items")

        if not texts and not text_items:
            raise ValueError("Either texts or text_items must be provided")

        return v


class BatchEmbedResult(BaseModel):
    """Result for single batch embedding item."""

    item_id: UUID = Field(description="Item identifier")
    embedding_id: Optional[UUID] = Field(default=None, description="Embedding identifier")
    success: bool = Field(description="Whether embedding was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchEmbedResponse(BaseModel):
    """Response from batch embedding operation."""

    success: bool = Field(description="Overall operation success")
    total_items: int = Field(ge=0, description="Total items processed")
    embedded_count: int = Field(ge=0, description="Number successfully embedded")
    stored_count: int = Field(ge=0, description="Number stored in database")
    failed_count: int = Field(ge=0, description="Number that failed")
    processing_time_ms: float = Field(ge=0, description="Total processing time")
    throughput_items_per_second: float = Field(
        ge=0, description="Processing throughput"
    )
    results: List[BatchEmbedResult] = Field(description="Per-item results")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


# ============================================================================
# Search Embeddings Schemas
# ============================================================================


class SearchRequest(BaseModel):
    """Request for similarity search."""

    query: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Query text (alternative to query_vector)",
    )
    query_vector: Optional[List[float]] = Field(
        default=None,
        description="Query embedding vector (alternative to query)",
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters",
    )
    include_vectors: bool = Field(
        default=False,
        description="Whether to include vectors in response",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for query embedding (if query provided)",
    )

    @field_validator("query_vector", "query")
    @classmethod
    def validate_query_input(cls, v, info):
        """Ensure at least one query method is provided."""
        query = info.data.get("query")
        query_vector = info.data.get("query_vector")

        if not query and not query_vector:
            raise ValueError("Either query or query_vector must be provided")

        return v


class SearchResultItem(BaseModel):
    """Single search result with similarity score."""

    embedding_id: UUID = Field(description="Embedding identifier")
    chunk_id: Optional[UUID] = Field(default=None, description="Chunk identifier")
    document_id: Optional[UUID] = Field(default=None, description="Document identifier")
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Similarity score"
    )
    rank: int = Field(ge=1, description="Result rank")
    metadata: Dict[str, Any] = Field(description="Embedding metadata")
    vector_dimension: Optional[int] = Field(
        default=None, description="Vector dimension (if included)"
    )


class SearchResponse(BaseModel):
    """Response from similarity search."""

    success: bool = Field(description="Whether search was successful")
    query: Optional[str] = Field(default=None, description="Query text if provided")
    total_results: int = Field(ge=0, description="Number of results returned")
    processing_time_ms: float = Field(ge=0, description="Search processing time")
    results: List[SearchResultItem] = Field(description="Search results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
