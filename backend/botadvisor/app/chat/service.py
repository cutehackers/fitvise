"""Thin retrieval-backed chat service for the canonical BotAdvisor API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.retriever_request import RetrieverRequest
from botadvisor.app.chat.prompting import build_chat_prompt_messages
from botadvisor.app.chat.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from botadvisor.app.observability.langfuse import get_tracer


@dataclass
class RetrievalChatService:
    """Coordinate retrieval results into query and chat responses."""

    retrieval_service: Any
    llm_service: Any = None
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
        answer = self._generate_answer(request.message, citations)
        return ChatResponse(answer=answer, total_sources=len(citations), sources=citations)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatResponseChunk]:
        trace = self._start_trace("api_chat", {"message": request.message, "top_k": request.top_k})
        try:
            chunks = self.retrieval_service.retrieve(
                RetrieverRequest(query=request.message, platform=request.platform, top_k=request.top_k)
            )
            citations = self._to_citations(chunks)

            if self.llm_service is not None:
                prompt_messages = build_chat_prompt_messages(question=request.message, citations=citations)
                answer_parts: list[str] = []
                async for token in self.llm_service.generate_stream(prompt_messages):
                    answer_parts.append(token)
                    yield ChatResponseChunk(delta=token, done=False)
                final_answer = "".join(answer_parts)
            else:
                final_answer = self._build_answer(citations)
                segments = final_answer.splitlines() or [final_answer]

                for segment in segments:
                    if segment:
                        yield ChatResponseChunk(delta=f"{segment}\n", done=False)

            final_chunk = ChatResponseChunk(
                answer=final_answer,
                total_sources=len(citations),
                sources=citations,
                done=True,
            )
            self._finish_trace(trace, "success", {"total_sources": len(citations)})
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

    def _generate_answer(self, question: str, citations: list[SourceCitation]) -> str:
        if self.llm_service is None:
            return self._build_answer(citations)

        prompt_messages = build_chat_prompt_messages(question=question, citations=citations)
        return self.llm_service.generate(prompt_messages)

    def _start_trace(self, name: str, metadata: dict[str, Any]) -> Any:
        if self.tracer and self.tracer.is_enabled():
            return self.tracer.trace(name=name, trace_type="api", metadata=metadata)
        return None

    @staticmethod
    def _finish_trace(trace: Any, status: str, output: dict[str, Any]) -> None:
        if trace:
            trace.update(status=status, output=output)
