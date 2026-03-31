"""Retrieval tool adapter for the canonical agent runtime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from botadvisor.app.core.entity.retriever_request import RetrieverRequest
from botadvisor.app.tools.contracts import ToolDefinition


class RetrievalToolInput(BaseModel):
    """Input payload for the canonical retrieval tool."""

    query: str = Field(min_length=1)
    platform: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalToolCitation(BaseModel):
    """Citation payload returned by the retrieval tool."""

    document_id: str
    chunk_id: str
    content: str
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalToolResult(BaseModel):
    """Structured output returned by the canonical retrieval tool."""

    tool_name: str = "retrieval"
    total_results: int
    results: list[RetrievalToolCitation]


def build_retrieval_tool_definition(retrieval_service: Any) -> ToolDefinition:
    """Return the canonical retrieval tool definition."""

    def execute_retrieval(payload: RetrievalToolInput) -> RetrievalToolResult:
        chunks = retrieval_service.retrieve(
            RetrieverRequest(query=payload.query, platform=payload.platform, top_k=payload.top_k)
        )
        results = [
            RetrievalToolCitation(
                document_id=chunk.metadata.doc_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                similarity_score=chunk.score or 0.0,
                metadata={
                    "source_id": chunk.metadata.source_id,
                    "platform": chunk.metadata.platform,
                    "source_url": chunk.metadata.source_url,
                    "page": chunk.metadata.page,
                    "section": chunk.metadata.section,
                },
            )
            for chunk in chunks
        ]
        return RetrievalToolResult(total_results=len(results), results=results)

    return ToolDefinition(
        name="retrieval",
        description="Retrieve supporting chunks and citation metadata for a user question.",
        input_model=RetrievalToolInput,
        output_model=RetrievalToolResult,
        handler=execute_retrieval,
    )
