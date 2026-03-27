from __future__ import annotations

from unittest.mock import Mock

import pytest

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document_metadata import DocumentMetadata


def make_chunk(*, chunk_id: str, content: str, score: float, page: int | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata=DocumentMetadata(
            doc_id="doc-1",
            source_id="file-1",
            platform="filesystem",
            source_url="file:///tmp/sample.txt",
            page=page,
            section="chunk_0",
        ),
        score=score,
    )


def test_query_returns_citation_shaped_results():
    from botadvisor.app.chat.schemas import QueryRequest
    from botadvisor.app.chat.service import RetrievalChatService

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [
        make_chunk(chunk_id="chunk-1", content="first chunk", score=0.91, page=1),
        make_chunk(chunk_id="chunk-2", content="second chunk", score=0.72, page=2),
    ]

    service = RetrievalChatService(retrieval_service=retrieval_service)
    response = service.query(QueryRequest(query="protein intake", platform="filesystem", top_k=2))

    assert response.query == "protein intake"
    assert response.total_results == 2
    assert [item.index for item in response.results] == [1, 2]
    assert response.results[0].content == "first chunk"
    assert response.results[0].metadata["source_url"] == "file:///tmp/sample.txt"


def test_chat_builds_answer_from_retrieved_context():
    from botadvisor.app.chat.schemas import ChatRequest
    from botadvisor.app.chat.service import RetrievalChatService

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [
        make_chunk(chunk_id="chunk-1", content="first chunk", score=0.91, page=1),
        make_chunk(chunk_id="chunk-2", content="second chunk", score=0.72, page=2),
    ]

    service = RetrievalChatService(retrieval_service=retrieval_service)
    response = service.chat(ChatRequest(message="what is a good protein intake?", platform="filesystem", top_k=2))

    assert "first chunk" in response.answer
    assert "second chunk" in response.answer
    assert "[1]" in response.answer
    assert "[2]" in response.answer
    assert response.total_sources == 2
    assert response.sources[0].document_id == "doc-1"


def test_chat_uses_llm_service_when_configured():
    from botadvisor.app.chat.schemas import ChatRequest
    from botadvisor.app.chat.service import RetrievalChatService

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [
        make_chunk(chunk_id="chunk-1", content="first chunk", score=0.91, page=1),
        make_chunk(chunk_id="chunk-2", content="second chunk", score=0.72, page=2),
    ]

    llm_service = Mock()
    llm_service.generate.return_value = "Aim for roughly 1.6 to 2.2 g/kg per day [1]."

    service = RetrievalChatService(retrieval_service=retrieval_service, llm_service=llm_service)
    response = service.chat(ChatRequest(message="what is a good protein intake?", platform="filesystem", top_k=2))

    assert response.answer == "Aim for roughly 1.6 to 2.2 g/kg per day [1]."
    llm_service.generate.assert_called_once()
    prompt_messages = llm_service.generate.call_args.args[0]
    assert prompt_messages[0].content.startswith("You are Fitvise")
    assert "first chunk" in prompt_messages[0].content
    assert "second chunk" in prompt_messages[0].content
    assert prompt_messages[1].content == "what is a good protein intake?"


@pytest.mark.asyncio
async def test_stream_chat_yields_ndjson_ready_chunks():
    from botadvisor.app.chat.schemas import ChatRequest
    from botadvisor.app.chat.service import RetrievalChatService

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [
        make_chunk(chunk_id="chunk-1", content="first chunk", score=0.91, page=1),
        make_chunk(chunk_id="chunk-2", content="second chunk", score=0.72, page=2),
    ]

    service = RetrievalChatService(retrieval_service=retrieval_service)
    chunks = [chunk async for chunk in service.stream_chat(ChatRequest(message="protein intake?", top_k=2))]

    assert len(chunks) >= 2
    assert chunks[-1].done is True
    assert chunks[-1].answer is not None
    assert chunks[-1].sources is not None


@pytest.mark.asyncio
async def test_stream_chat_uses_llm_stream_when_available():
    from botadvisor.app.chat.schemas import ChatRequest
    from botadvisor.app.chat.service import RetrievalChatService

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [
        make_chunk(chunk_id="chunk-1", content="first chunk", score=0.91, page=1),
    ]

    async def fake_stream(_messages):
        yield "Aim for "
        yield "1.6 g/kg."

    llm_service = Mock()
    llm_service.generate_stream = fake_stream

    service = RetrievalChatService(retrieval_service=retrieval_service, llm_service=llm_service)
    chunks = [chunk async for chunk in service.stream_chat(ChatRequest(message="protein intake?", top_k=1))]

    assert [chunk.delta for chunk in chunks[:-1]] == ["Aim for ", "1.6 g/kg."]
    assert chunks[-1].done is True
    assert chunks[-1].answer == "Aim for 1.6 g/kg."
    assert chunks[-1].total_sources == 1


def test_query_creates_langfuse_trace_when_enabled():
    from botadvisor.app.chat.schemas import QueryRequest
    from botadvisor.app.chat.service import RetrievalChatService

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [make_chunk(chunk_id="chunk-1", content="first chunk", score=0.91)]

    trace = Mock()
    tracer = Mock()
    tracer.is_enabled.return_value = True
    tracer.trace.return_value = trace

    service = RetrievalChatService(retrieval_service=retrieval_service, tracer=tracer)
    response = service.query(QueryRequest(query="protein intake"))

    assert response.total_results == 1
    tracer.trace.assert_called_once()
    trace.update.assert_called_once()
