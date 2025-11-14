"""Unit tests for OllamaService wrapping LlmService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.infrastructure.external_services.ml_services.llm_services.ollama_service import (
    OllamaService,
)
from app.infrastructure.external_services.ml_services.llm_services.base_llm_service import (
    LlmResponse,
    LlmHealthStatus,
)


class TestOllamaService:
    """Test OllamaService wrapper functionality."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LlmService."""
        mock_service = MagicMock()
        mock_service.llm = MagicMock()
        mock_service.llm.model = "llama3.2:3b"
        return mock_service

    @pytest.fixture
    def ollama_service(self, mock_llm_service):
        """Create OllamaService with mocked LlmService."""
        return OllamaService(mock_llm_service)

    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_service, mock_llm_service):
        """Test successful non-streaming generation."""
        # Mock response
        mock_response = AIMessage(
            content="Test response",
            response_metadata={
                "prompt_eval_count": 10,
                "eval_count": 20,
                "total_duration": 1000000,
            },
        )
        mock_llm_service.llm.ainvoke = AsyncMock(return_value=mock_response)

        # Execute
        result = await ollama_service.generate(prompt="Test prompt", temperature=0.7)

        # Verify
        assert isinstance(result, LlmResponse)
        assert result.content == "Test response"
        assert result.model == "llama3.2:3b"
        assert result.tokens_usage["prompt_tokens"] == 10
        assert result.tokens_usage["completion_tokens"] == 20
        assert result.tokens_usage["total_tokens"] == 30

        # Verify LLM was called correctly
        mock_llm_service.llm.ainvoke.assert_called_once()
        call_args = mock_llm_service.llm.ainvoke.call_args
        assert call_args[1]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, ollama_service, mock_llm_service):
        """Test generation with max_tokens parameter."""
        mock_response = AIMessage(
            content="Response",
            response_metadata={"prompt_eval_count": 5, "eval_count": 10},
        )
        mock_llm_service.llm.ainvoke = AsyncMock(return_value=mock_response)

        # Execute with max_tokens
        await ollama_service.generate(prompt="Test", max_tokens=100)

        # Verify num_predict parameter is set
        call_args = mock_llm_service.llm.ainvoke.call_args
        assert call_args[1]["num_predict"] == 100

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, ollama_service, mock_llm_service):
        """Test error handling in generation."""
        # Mock error
        mock_llm_service.llm.ainvoke = AsyncMock(
            side_effect=Exception("LLM service error")
        )

        # Verify exception is raised
        with pytest.raises(Exception, match="LLM service error"):
            await ollama_service.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, ollama_service, mock_llm_service):
        """Test successful streaming generation."""

        # Mock streaming response
        async def mock_astream(*args, **kwargs):
            chunks = [
                AIMessage(content="Hello"),
                AIMessage(content=" world"),
                AIMessage(content="!"),
            ]
            for chunk in chunks:
                yield chunk

        mock_llm_service.llm.astream = mock_astream

        # Execute
        chunks = []
        async for chunk in ollama_service.generate_stream(
            prompt="Test", temperature=0.5
        ):
            chunks.append(chunk)

        # Verify
        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_empty_chunks(self, ollama_service, mock_llm_service):
        """Test streaming with empty content chunks."""

        async def mock_astream(*args, **kwargs):
            chunks = [
                AIMessage(content="Valid"),
                AIMessage(content=""),  # Empty chunk
                AIMessage(content="Text"),
            ]
            for chunk in chunks:
                yield chunk

        mock_llm_service.llm.astream = mock_astream

        # Execute
        chunks = []
        async for chunk in ollama_service.generate_stream(prompt="Test"):
            chunks.append(chunk)

        # Verify empty chunks are filtered
        assert chunks == ["Valid", "Text"]

    @pytest.mark.asyncio
    async def test_generate_stream_error(self, ollama_service, mock_llm_service):
        """Test error handling in streaming."""

        async def mock_astream(*args, **kwargs):
            raise Exception("Streaming error")
            yield  # pragma: no cover

        mock_llm_service.llm.astream = mock_astream

        # Verify exception is raised
        with pytest.raises(Exception, match="Streaming error"):
            async for _ in ollama_service.generate_stream(prompt="Test"):
                pass

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, ollama_service, mock_llm_service):
        """Test health check when service is healthy."""
        # Mock successful health check
        mock_response = AIMessage(content="pong")
        mock_llm_service.llm.ainvoke = AsyncMock(return_value=mock_response)

        # Execute
        status = await ollama_service.health_check()

        # Verify
        assert isinstance(status, LlmHealthStatus)
        assert status.is_healthy is True
        assert status.model == "llama3.2:3b"
        assert status.response_time_ms > 0
        assert status.error is None

        # Verify health check used minimal tokens
        call_args = mock_llm_service.llm.ainvoke.call_args
        assert call_args[1]["num_predict"] == 1

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, ollama_service, mock_llm_service):
        """Test health check when service is unhealthy."""
        # Mock health check failure
        mock_llm_service.llm.ainvoke = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        # Execute
        status = await ollama_service.health_check()

        # Verify
        assert isinstance(status, LlmHealthStatus)
        assert status.is_healthy is False
        assert status.model == "llama3.2:3b"
        assert status.response_time_ms > 0
        assert "Connection refused" in status.error

    @pytest.mark.asyncio
    async def test_model_name_preserved(self, ollama_service):
        """Test that model name is correctly preserved from LlmService."""
        assert ollama_service._model == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_additional_kwargs_passed(self, ollama_service, mock_llm_service):
        """Test that additional kwargs are passed to LLM."""
        mock_response = AIMessage(
            content="Response", response_metadata={"eval_count": 10}
        )
        mock_llm_service.llm.ainvoke = AsyncMock(return_value=mock_response)

        # Execute with custom kwargs
        await ollama_service.generate(
            prompt="Test", custom_param="value", stop=["STOP"]
        )

        # Verify custom kwargs were passed
        call_args = mock_llm_service.llm.ainvoke.call_args
        assert call_args[1]["custom_param"] == "value"
        assert call_args[1]["stop"] == ["STOP"]
