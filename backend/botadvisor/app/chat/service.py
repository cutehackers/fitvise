"""Thin retrieval-backed chat service for the canonical BotAdvisor API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.retriever_request import RetrieverRequest
from botadvisor.app.observability.langfuse import get_tracer
from botadvisor.app.chat.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)


@dataclass
class RetrievalChatService:
    """Coordinate retrieval results into query and chat responses."""

    retrieval_service: Any
    tracer: Any = None

    def __post_init__(self) -> None:
        if self.tracer is None:
            self.tracer = get_tracer()

    def query(self, request: QueryRequest) -> QueryResponse:
        trace = self._start_trace("api_query", {"query": request.query, "top_k": request.top_k})
        try:
            chunks = self.retrieval_service.retrieve(
                RetrieverRequest(query=request.query, platform=request.platform, top_k=request.top_k)
            )
            citations = self._to_citations(chunks)
            response = QueryResponse(query=request.query, total_results=len(citations), results=citations)
            self._finish_trace(trace, "success", {"total_results": response.total_results})
            return response
        except Exception as exc:
            self._finish_trace(trace, "error", {"error": str(exc)})
            raise

    def chat(self, request: ChatRequest) -> ChatResponse:
        chunks = self.retrieval_service.retrieve(
            RetrieverRequest(query=request.message, platform=request.platform, top_k=request.top_k)
        )
        citations = self._to_citations(chunks)
        answer = self._build_answer(citations)
        return ChatResponse(answer=answer, total_sources=len(citations), sources=citations)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatResponseChunk]:
        trace = self._start_trace("api_chat", {"message": request.message, "top_k": request.top_k})
        try:
            response = self.chat(request)
            segments = response.answer.splitlines() or [response.answer]

            for segment in segments:
                if segment:
                    yield ChatResponseChunk(delta=f"{segment}\n", done=False)

            final_chunk = ChatResponseChunk(
                answer=response.answer,
                total_sources=response.total_sources,
                sources=response.sources,
                done=True,
            )
            self._finish_trace(trace, "success", {"total_sources": response.total_sources})
            yield final_chunk
        except Exception as exc:
            self._finish_trace(trace, "error", {"error": str(exc)})
            raise

    def _to_citations(self, chunks: list[Chunk]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        for index, chunk in enumerate(chunks, start=1):
            citations.append(
                SourceCitation(
                    index=index,
                    content=chunk.content,
                    similarity_score=chunk.score or 0.0,
                    document_id=chunk.metadata.doc_id,
                    chunk_id=chunk.chunk_id,
                    metadata={
                        "source_id": chunk.metadata.source_id,
                        "platform": chunk.metadata.platform,
                        "source_url": chunk.metadata.source_url,
                        "page": chunk.metadata.page,
                        "section": chunk.metadata.section,
                    },
                )
            )
        return citations

    def _build_answer(self, citations: list[SourceCitation]) -> str:
        if not citations:
            return "No supporting context was found for this question."

        parts = ["Retrieved context summary:"]
        for citation in citations:
            parts.append(f"[{citation.index}] {citation.content}")
        return "\n".join(parts)

    def _start_trace(self, name: str, metadata: dict[str, Any]) -> Any:
        if self.tracer and self.tracer.is_enabled():
            return self.tracer.trace(name=name, trace_type="api", metadata=metadata)
        return None

    @staticmethod
    def _finish_trace(trace: Any, status: str, output: dict[str, Any]) -> None:
        if trace:
            trace.update(status=status, output=output)
