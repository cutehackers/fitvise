"""
Unit tests for LLM service functionality.

Tests the LLMService class methods in isolation using mocks.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.application.llm_service import LLMService
from tests.utils.test_helpers import generate_data, create_mocks, assert_helpers


class TestLLMService:
    """Test the LLMService class."""

    @pytest.fixture
    def llm_service(self, test_settings):
        """Create an LLMService instance for testing."""
        return LLMService(
            base_url=test_settings.llm_base_url,
            model=test_settings.llm_model,
            timeout=test_settings.llm_timeout,
        )

    @pytest.mark.asyncio
    async def test_generate_response_success(self, llm_service):
        """Test successful response generation."""
        # Prepare test data
        request_data = generate_data.workout_prompt_data()
        mock_response_data = generate_data.workout_response_data()

        # Mock the HTTP client
        with patch.object(llm_service, "_client") as mock_client:
            mock_response = create_mocks.mock_http_response(
                status_code=200,
                json_data={
                    "response": mock_response_data["response"],
                    "usage": {"total_tokens": mock_response_data["tokens_used"]},
                    "model": mock_response_data["model"],
                },
            )
            mock_client.post.return_value = mock_response

            # Execute the method
            result = await llm_service.generate_response(
                prompt=request_data["prompt"],
                context=request_data.get("context"),
                max_tokens=request_data.get("max_tokens"),
                temperature=request_data.get("temperature"),
            )

            # Verify the result
            assert_helpers.assert_valid_workout_response(result)
            assert result["success"] is True
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert result["tokens_used"] > 0
            assert result["response_time"] > 0

    @pytest.mark.asyncio
    async def test_generate_response_http_error(self, llm_service):
        """Test response generation with HTTP error."""
        request_data = generate_data.workout_prompt_data()

        with patch.object(llm_service, "_client") as mock_client:
            # Mock HTTP error
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=MagicMock(status_code=500)
            )

            result = await llm_service.generate_response(prompt=request_data["prompt"])

            # Verify error handling
            assert result["success"] is False
            assert "error" in result
            assert result["tokens_used"] == 0
            assert result["response_time"] > 0

    @pytest.mark.asyncio
    async def test_generate_response_timeout(self, llm_service):
        """Test response generation with timeout."""
        request_data = generate_data.workout_prompt_data()

        with patch.object(llm_service, "_client") as mock_client:
            # Mock timeout
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

            result = await llm_service.generate_response(prompt=request_data["prompt"])

            # Verify timeout handling
            assert result["success"] is False
            assert "timeout" in result.get("error", "").lower()
            assert result["tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_generate_response_invalid_json(self, llm_service):
        """Test response generation with invalid JSON response."""
        request_data = generate_data.workout_prompt_data()

        with patch.object(llm_service, "_client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.text = "Invalid JSON response"
            mock_client.post.return_value = mock_response

            result = await llm_service.generate_response(prompt=request_data["prompt"])

            # Verify JSON error handling
            assert result["success"] is False
            assert "json" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service):
        """Test successful health check."""
        mock_health_data = generate_data.health_response_data()

        with patch.object(llm_service, "_client") as mock_client:
            mock_response = create_mocks.mock_http_response(
                status_code=200,
                json_data={"status": "ok", "model": mock_health_data["model"]},
            )
            mock_client.get.return_value = mock_response

            result = await llm_service.health_check()

            # Verify health check result
            assert_helpers.assert_valid_health_response(result)
            assert result["status"] == "healthy"
            assert isinstance(result["response_time"], (int, float))

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service):
        """Test health check with service failure."""
        with patch.object(llm_service, "_client") as mock_client:
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")

            result = await llm_service.health_check()

            # Verify failure handling
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert isinstance(result["response_time"], (int, float))

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, llm_service):
        """Test health check with timeout."""
        with patch.object(llm_service, "_client") as mock_client:
            mock_client.get.side_effect = httpx.TimeoutException("Health check timeout")

            result = await llm_service.health_check()

            # Verify timeout handling
            assert result["status"] == "unhealthy"
            assert "timeout" in result.get("error", "").lower()
            assert result["response_time"] >= llm_service.timeout

    def test_format_request_data(self, llm_service):
        """Test request data formatting."""
        prompt = "Test workout prompt"
        context = "Test context"
        max_tokens = 500
        temperature = 0.7

        formatted_data = llm_service._format_request_data(
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Verify formatted data structure
        assert isinstance(formatted_data, dict)
        assert "prompt" in formatted_data
        assert "model" in formatted_data
        assert formatted_data["model"] == llm_service.model

        # Verify prompt includes context
        full_prompt = formatted_data["prompt"]
        assert prompt in full_prompt
        if context:
            assert context in full_prompt

    def test_format_request_data_defaults(self, llm_service):
        """Test request data formatting with default values."""
        prompt = "Test prompt"

        formatted_data = llm_service._format_request_data(prompt=prompt)

        # Verify defaults are applied
        assert formatted_data["prompt"] == prompt
        assert formatted_data["model"] == llm_service.model
        # Should use service defaults for max_tokens and temperature

    @pytest.mark.asyncio
    async def test_response_parsing_success(self, llm_service):
        """Test successful response parsing."""
        mock_response = create_mocks.mock_http_response(
            status_code=200,
            json_data={
                "response": "Test workout response",
                "usage": {"total_tokens": 150},
                "model": "test-model",
            },
        )

        start_time = 1000.0
        end_time = 1001.5

        result = llm_service._parse_response(mock_response, start_time, end_time)

        # Verify parsed response
        assert result["success"] is True
        assert result["response"] == "Test workout response"
        assert result["tokens_used"] == 150
        assert result["model"] == "test-model"
        assert result["response_time"] == 1.5

    def test_response_parsing_missing_fields(self, llm_service):
        """Test response parsing with missing fields."""
        mock_response = create_mocks.mock_http_response(
            status_code=200,
            json_data={"response": "Test response"},  # Missing usage and model
        )

        start_time = 1000.0
        end_time = 1001.0

        result = llm_service._parse_response(mock_response, start_time, end_time)

        # Should handle missing fields gracefully
        assert result["success"] is True
        assert result["response"] == "Test response"
        assert result["tokens_used"] == 0  # Default
        assert "model" in result  # Should have some default

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, llm_service):
        """Test handling multiple concurrent requests."""
        import asyncio

        request_data = generate_data.workout_prompt_data()
        mock_response_data = generate_data.workout_response_data()

        with patch.object(llm_service, "_client") as mock_client:
            mock_response = create_mocks.mock_http_response(
                status_code=200,
                json_data={
                    "response": mock_response_data["response"],
                    "usage": {"total_tokens": mock_response_data["tokens_used"]},
                    "model": mock_response_data["model"],
                },
            )
            mock_client.post.return_value = mock_response

            # Execute multiple concurrent requests
            tasks = [llm_service.generate_response(prompt=f"{request_data['prompt']} #{i}") for i in range(5)]

            results = await asyncio.gather(*tasks)

            # Verify all requests succeeded
            assert len(results) == 5
            for result in results:
                assert result["success"] is True
                assert_helpers.assert_valid_workout_response(result)
