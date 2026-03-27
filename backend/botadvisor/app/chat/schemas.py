"""Request and response schemas for the canonical BotAdvisor API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Source citation returned by retrieval-backed API responses."""

    index: int
    content: str
    similarity_score: float
    document_id: str
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request schema for retrieval-only queries."""

    query: str = Field(min_length=1)
    platform: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response schema for retrieval-only queries."""

    query: str
    total_results: int
    results: list[SourceCitation]


class ChatRequest(BaseModel):
    """Request schema for retrieval-backed chat responses."""

    message: str = Field(min_length=1)
    platform: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Response schema for retrieval-backed chat responses."""

    answer: str
    total_sources: int
    sources: list[SourceCitation]


class ChatResponseChunk(BaseModel):
    """Streaming chunk schema for NDJSON chat responses."""

    delta: str | None = None
    answer: str | None = None
    total_sources: int | None = None
    sources: list[SourceCitation] | None = None
    done: bool = False


class HealthResponse(BaseModel):
    """Minimal runtime health response for the canonical API."""

    status: str
    service: str
    retrieval_available: bool
    langfuse_enabled: bool
