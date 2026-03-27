from __future__ import annotations

from fastapi.testclient import TestClient


class FakeChatService:
    def query(self, request):
        from botadvisor.app.chat.schemas import QueryResponse, SourceCitation

        return QueryResponse(
            query=request.query,
            total_results=1,
            results=[
                SourceCitation(
                    index=1,
                    content="retrieved content",
                    similarity_score=0.88,
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    metadata={"source_url": "file:///tmp/sample.txt"},
                )
            ],
        )

    def chat(self, request):
        from botadvisor.app.chat.schemas import ChatResponse, SourceCitation

        return ChatResponse(
            answer="Based on retrieved context [1]: retrieved content",
            total_sources=1,
            sources=[
                SourceCitation(
                    index=1,
                    content="retrieved content",
                    similarity_score=0.88,
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    metadata={"source_url": "file:///tmp/sample.txt"},
                )
            ],
        )

    async def stream_chat(self, request):
        from botadvisor.app.chat.schemas import ChatResponseChunk, SourceCitation

        yield ChatResponseChunk(delta="Based on retrieved context [1]: ", done=False)
        yield ChatResponseChunk(
            delta="retrieved content",
            answer="Based on retrieved context [1]: retrieved content",
            total_sources=1,
            sources=[
                SourceCitation(
                    index=1,
                    content="retrieved content",
                    similarity_score=0.88,
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    metadata={"source_url": "file:///tmp/sample.txt"},
                )
            ],
            done=True,
        )


class FakeHealthService:
    async def get_status(self):
        from botadvisor.app.chat.schemas import HealthResponse

        return HealthResponse(
            status="healthy",
            service="botadvisor-api",
            retrieval_available=True,
            langfuse_enabled=False,
        )


def test_health_endpoint_returns_runtime_status():
    from botadvisor.app.main import create_app

    client = TestClient(create_app(chat_service=FakeChatService(), health_service=FakeHealthService()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "botadvisor-api",
        "retrieval_available": True,
        "langfuse_enabled": False,
    }


def test_query_endpoint_returns_retrieved_sources():
    from botadvisor.app.main import create_app

    client = TestClient(create_app(chat_service=FakeChatService(), health_service=FakeHealthService()))

    response = client.post(
        "/query",
        json={"query": "protein intake", "platform": "filesystem", "top_k": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "protein intake"
    assert data["total_results"] == 1
    assert data["results"][0]["document_id"] == "doc-1"


def test_chat_endpoint_returns_answer_and_citations():
    from botadvisor.app.main import create_app

    client = TestClient(create_app(chat_service=FakeChatService(), health_service=FakeHealthService()))

    with client.stream(
        "POST",
        "/chat",
        json={"message": "protein intake?", "platform": "filesystem", "top_k": 1},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [line for line in response.iter_lines() if line]

    assert len(lines) >= 1
    assert '"done":true' in lines[-1]
    assert '"chunk_id":"chunk-1"' in lines[-1]
