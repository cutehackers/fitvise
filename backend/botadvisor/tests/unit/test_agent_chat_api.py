from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient

from botadvisor.app.agent.schemas import AgentTurnResult
from botadvisor.app.tools.contracts import ToolExecutionResult


def test_chat_service_uses_agent_runtime_when_configured():
    from botadvisor.app.chat.schemas import ChatRequest
    from botadvisor.app.chat.service import RetrievalChatService

    agent_service = Mock()
    agent_service.run_turn.return_value = AgentTurnResult(
        answer="Tool-based answer.",
        used_tool_name="retrieval",
        tool_result=ToolExecutionResult(
            tool_name="retrieval",
            payload={
                "results": [
                    {
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "content": "retrieved content",
                        "similarity_score": 0.88,
                        "metadata": {"source_url": "file:///tmp/sample.txt"},
                    }
                ]
            },
        ),
    )

    service = RetrievalChatService(retrieval_service=Mock(), agent_service=agent_service)
    response = service.chat(ChatRequest(message="protein intake?", platform="filesystem", top_k=1))

    assert response.answer == "Tool-based answer."
    assert response.total_sources == 1
    assert response.sources[0].chunk_id == "chunk-1"


def test_chat_endpoint_keeps_ndjson_contract_with_agent_runtime():
    from botadvisor.app.chat.service import RetrievalChatService
    from botadvisor.app.main import create_app

    class FakeHealthService:
        async def get_status(self):
            from botadvisor.app.chat.schemas import HealthResponse

            return HealthResponse(
                status="healthy",
                service="botadvisor-api",
                retrieval_available=True,
                langfuse_enabled=False,
                checks={
                    "retrieval": {"status": "healthy"},
                    "vector_store": {"status": "healthy"},
                    "llm_path": {"status": "healthy"},
                },
            )

    agent_service = Mock()
    agent_service.run_turn.return_value = AgentTurnResult(
        answer="Tool-based answer.",
        used_tool_name="retrieval",
        tool_result=ToolExecutionResult(
            tool_name="retrieval",
            payload={
                "results": [
                    {
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "content": "retrieved content",
                        "similarity_score": 0.88,
                        "metadata": {"source_url": "file:///tmp/sample.txt"},
                    }
                ]
            },
        ),
    )

    chat_service = RetrievalChatService(retrieval_service=Mock(), agent_service=agent_service)
    client = TestClient(create_app(chat_service=chat_service, health_service=FakeHealthService()))

    with client.stream(
        "POST",
        "/chat",
        json={"message": "protein intake?", "platform": "filesystem", "top_k": 1},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [line for line in response.iter_lines() if line]

    assert '"done":true' in lines[-1]
    assert '"chunk_id":"chunk-1"' in lines[-1]
    assert "Tool-based answer." in lines[-1]
