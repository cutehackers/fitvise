"""Thin retrieval-backed chat service for the canonical BotAdvisor API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.retriever_request import RetrieverRequest
from botadvisor.app.chat.schemas import ChatRequest, ChatResponse, QueryRequest, QueryResponse, SourceCitation


@dataclass
class RetrievalChatService:
    """Coordinate retrieval results into query and chat responses."""

    retrieval_service: Any

    def query(self, request: QueryRequest) -> QueryResponse:
        chunks = self.retrieval_service.retrieve(
            RetrieverRequest(query=request.query, platform=request.platform, top_k=request.top_k)
        )
        citations = self._to_citations(chunks)
        return QueryResponse(query=request.query, total_results=len(citations), results=citations)

    def chat(self, request: ChatRequest) -> ChatResponse:
        chunks = self.retrieval_service.retrieve(
            RetrieverRequest(query=request.message, platform=request.platform, top_k=request.top_k)
        )
        citations = self._to_citations(chunks)
        answer = self._build_answer(citations)
        return ChatResponse(answer=answer, total_sources=len(citations), sources=citations)

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
