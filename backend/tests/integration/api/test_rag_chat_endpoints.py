"""Integration tests for RAG chat endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.schemas.chat import ChatMessage, ChatRequest


class TestRagChatEndpoints:
    """Integration tests for /chat-rag endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_rag_use_case(self):
        """Create mock RAG use case."""
        mock = MagicMock()
        return mock

    def test_chat_rag_success(self, client, mock_rag_use_case):
        """Test successful RAG chat request."""
        # Mock RAG response
        async def mock_stream():
            chunks = ["Exercise ", "is ", "important."]
            for chunk in chunks:
                yield chunk

        mock_docs = [
            Document(
                page_content="Exercise improves health",
                metadata={
                    "chunk_id": "1",
                    "document_id": "doc_1",
                    "similarity_score": 0.92,
                },
            )
        ]

        mock_rag_use_case.execute_rag_stream = AsyncMock(
            return_value=(mock_stream(), mock_docs)
        )

        # Patch dependency
        with patch(
            "app.api.v1.fitvise.chat.get_rag_use_case", return_value=mock_rag_use_case
        ):
            response = client.post(
                "/api/v1/fitvise/chat-rag",
                json={
                    "message": {"role": "user", "content": "What are exercise benefits?"},
                    "session_id": "test-session-123",
                },
            )

        # Verify response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        # Parse streaming response
        lines = response.text.strip().split("\n")
        assert len(lines) > 0

    def test_chat_rag_empty_message_content(self, client):
        """Test RAG chat with empty message content."""
        response = client.post(
            "/api/v1/fitvise/chat-rag",
            json={
                "message": {"role": "user", "content": ""},
                "session_id": "test-session",
            },
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"]["message"].lower()

    def test_chat_rag_whitespace_message(self, client):
        """Test RAG chat with whitespace-only content."""
        response = client.post(
            "/api/v1/fitvise/chat-rag",
            json={
                "message": {"role": "user", "content": "   \n\t  "},
                "session_id": "test-session",
            },
        )

        assert response.status_code == 400

    def test_chat_rag_missing_message(self, client):
        """Test RAG chat with missing message field."""
        response = client.post(
            "/api/v1/fitvise/chat-rag",
            json={"session_id": "test-session"},
        )

        assert response.status_code == 422  # Validation error

    def test_chat_rag_missing_session_id(self, client):
        """Test RAG chat with missing session_id."""
        response = client.post(
            "/api/v1/fitvise/chat-rag",
            json={"message": {"role": "user", "content": "test"}},
        )

        assert response.status_code == 422  # Validation error

    def test_chat_rag_streaming_response_format(self, client, mock_rag_use_case):
        """Test that streaming response follows expected format."""

        async def mock_stream():
            yield "Chunk 1"
            yield "Chunk 2"

        mock_docs = [
            Document(
                page_content="Source content",
                metadata={
                    "chunk_id": "1",
                    "document_id": "doc_1",
                    "similarity_score": 0.88,
                    "source": "test.pdf",
                },
            )
        ]

        mock_rag_use_case.execute_rag_stream = AsyncMock(
            return_value=(mock_stream(), mock_docs)
        )

        with patch(
            "app.api.v1.fitvise.chat.get_rag_use_case", return_value=mock_rag_use_case
        ):
            response = client.post(
                "/api/v1/fitvise/chat-rag",
                json={
                    "message": {"role": "user", "content": "test query"},
                    "session_id": "test-session",
                },
            )

        # Parse response lines
        lines = response.text.strip().split("\n")
        import json

        # Last line should have sources
        final_response = json.loads(lines[-1])
        assert final_response["done"] is True
        assert "sources" in final_response
        assert len(final_response["sources"]) == 1

        # Earlier lines should be chunks
        for line in lines[:-1]:
            chunk_response = json.loads(line)
            assert chunk_response["done"] is False
            assert chunk_response.get("sources") is None

    def test_chat_rag_source_citation_format(self, client, mock_rag_use_case):
        """Test that source citations have correct format."""

        async def mock_stream():
            yield "Response"

        mock_docs = [
            Document(
                page_content="Source content with metadata",
                metadata={
                    "chunk_id": "chunk_abc",
                    "document_id": "doc_xyz",
                    "similarity_score": 0.95,
                    "source": "fitness_guide.pdf",
                    "page": 42,
                },
            )
        ]

        mock_rag_use_case.execute_rag_stream = AsyncMock(
            return_value=(mock_stream(), mock_docs)
        )

        with patch(
            "app.api.v1.fitvise.chat.get_rag_use_case", return_value=mock_rag_use_case
        ):
            response = client.post(
                "/api/v1/fitvise/chat-rag",
                json={
                    "message": {"role": "user", "content": "test"},
                    "session_id": "test",
                },
            )

        # Parse final response with sources
        import json

        lines = response.text.strip().split("\n")
        final_response = json.loads(lines[-1])

        # Verify source citation structure
        source = final_response["sources"][0]
        assert source["index"] == 1
        assert source["content"] == "Source content with metadata"
        assert source["similarity_score"] == 0.95
        assert source["document_id"] == "doc_xyz"
        assert source["chunk_id"] == "chunk_abc"
        assert "source" in source["metadata"]


class TestHealthEndpoints:
    """Integration tests for LLM health endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_llm_health_endpoint(self, client):
        """Test /health/llm endpoint."""
        # Mock health monitor
        mock_monitor = MagicMock()
        mock_monitor.check_health = AsyncMock(
            return_value={
                "status": "healthy",
                "model": "llama3.2:3b",
                "response_time_ms": 123.45,
                "avg_response_time_ms": 150.0,
                "p95_response_time_ms": 200.0,
                "success_rate": 98.5,
                "error": None,
                "last_check": "2025-11-13T10:00:00",
            }
        )

        with patch(
            "app.api.v1.fitvise.chat.get_llm_health_monitor",
            return_value=mock_monitor,
        ):
            response = client.get("/api/v1/fitvise/health/llm")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"] == "llama3.2:3b"
        assert "response_time_ms" in data
        assert "avg_response_time_ms" in data
        assert "p95_response_time_ms" in data
        assert "success_rate" in data

    def test_llm_health_unhealthy(self, client):
        """Test /health/llm when service is unhealthy."""
        mock_monitor = MagicMock()
        mock_monitor.check_health = AsyncMock(
            return_value={
                "status": "unhealthy",
                "model": "llama3.2:3b",
                "response_time_ms": 5000.0,
                "avg_response_time_ms": 4500.0,
                "p95_response_time_ms": 6000.0,
                "success_rate": 45.0,
                "error": "Connection timeout",
                "last_check": "2025-11-13T10:00:00",
            }
        )

        with patch(
            "app.api.v1.fitvise.chat.get_llm_health_monitor",
            return_value=mock_monitor,
        ):
            response = client.get("/api/v1/fitvise/health/llm")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["error"] == "Connection timeout"
        assert data["success_rate"] == 45.0

    def test_llm_metrics_endpoint(self, client):
        """Test /health/llm/metrics endpoint."""
        mock_monitor = MagicMock()
        mock_monitor.get_metrics = AsyncMock(
            return_value={
                "avg_response_time_ms": 150.0,
                "p95_response_time_ms": 250.0,
                "success_rate": 98.5,
                "total_checks": 100,
                "error_count": 2,
                "success_count": 98,
                "last_check": "2025-11-13T10:00:00",
                "last_error": None,
            }
        )

        with patch(
            "app.api.v1.fitvise.chat.get_llm_health_monitor",
            return_value=mock_monitor,
        ):
            response = client.get("/api/v1/fitvise/health/llm/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] == 100
        assert data["success_count"] == 98
        assert data["error_count"] == 2
        assert data["success_rate"] == 98.5
        assert "avg_response_time_ms" in data
        assert "p95_response_time_ms" in data

    def test_llm_metrics_no_checks(self, client):
        """Test metrics endpoint with no prior checks."""
        mock_monitor = MagicMock()
        mock_monitor.get_metrics = AsyncMock(
            return_value={
                "avg_response_time_ms": 0.0,
                "p95_response_time_ms": 0.0,
                "success_rate": 0.0,
                "total_checks": 0,
                "error_count": 0,
                "success_count": 0,
                "last_check": None,
                "last_error": None,
            }
        )

        with patch(
            "app.api.v1.fitvise.chat.get_llm_health_monitor",
            return_value=mock_monitor,
        ):
            response = client.get("/api/v1/fitvise/health/llm/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] == 0
        assert data["success_rate"] == 0.0

    def test_llm_health_exception_handling(self, client):
        """Test health endpoint exception handling."""
        mock_monitor = MagicMock()
        mock_monitor.check_health = AsyncMock(
            side_effect=Exception("Health check failed")
        )

        with patch(
            "app.api.v1.fitvise.chat.get_llm_health_monitor",
            return_value=mock_monitor,
        ):
            response = client.get("/api/v1/fitvise/health/llm")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data

    def test_llm_metrics_exception_handling(self, client):
        """Test metrics endpoint exception handling."""
        mock_monitor = MagicMock()
        mock_monitor.get_metrics = AsyncMock(
            side_effect=Exception("Metrics retrieval failed")
        )

        with patch(
            "app.api.v1.fitvise.chat.get_llm_health_monitor",
            return_value=mock_monitor,
        ):
            response = client.get("/api/v1/fitvise/health/llm/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["total_checks"] == 0


class TestExistingChatEndpoint:
    """Verify existing /chat endpoint still works."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_regular_chat_still_works(self, client):
        """Test that regular /chat endpoint is not affected."""
        # Mock LLM service
        with patch("app.api.v1.fitvise.chat.llm_service") as mock_llm_service:

            async def mock_chat_stream(request):
                from app.schemas.chat import ChatResponse, ChatMessage

                yield ChatResponse(
                    model="llama3.2:3b",
                    created_at="2025-11-13T10:00:00",
                    message=ChatMessage(role="assistant", content="Test response"),
                    done=True,
                )

            mock_llm_service.chat = mock_chat_stream

            response = client.post(
                "/api/v1/fitvise/chat",
                json={
                    "message": {"role": "user", "content": "test"},
                    "session_id": "test",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"
