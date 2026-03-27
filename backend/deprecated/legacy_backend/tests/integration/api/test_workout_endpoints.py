"""
Integration tests for workout API endpoints.

Tests the complete request/response cycle including:
- Request validation
- Service integration
- Response formatting
- Error handling
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.sample_data import SAMPLE_WORKOUT_PROMPTS, SAMPLE_WORKOUT_RESPONSES
from tests.utils.test_helpers import assert_helpers, generate_data


class TestWorkoutEndpoints:
    """Test workout API endpoints integration."""

    def test_workout_prompt_endpoint_success(self, test_client: TestClient, mock_llm_service):
        """Test successful workout prompt request."""
        # Setup mock response
        mock_response = SAMPLE_WORKOUT_RESPONSES[0].copy()
        mock_llm_service.generate_response.return_value = mock_response

        # Make request
        request_data = SAMPLE_WORKOUT_PROMPTS[0]
        response = test_client.post("/api/v1/workout/prompt", json=request_data)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()

        assert_helpers.assert_valid_workout_response(response_data)
        assert response_data["success"] is True
        assert len(response_data["response"]) > 0

        # Verify service was called correctly
        mock_llm_service.generate_response.assert_called_once()
        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs["prompt"] == request_data["prompt"]
        assert call_args.kwargs["context"] == request_data["context"]
        assert call_args.kwargs["max_tokens"] == request_data["max_tokens"]
        assert call_args.kwargs["temperature"] == request_data["temperature"]

    def test_workout_prompt_endpoint_minimal_request(self, test_client: TestClient, mock_llm_service):
        """Test workout prompt with minimal required data."""
        mock_response = generate_data.workout_response_data()
        mock_llm_service.generate_response.return_value = mock_response

        # Request with only required field
        request_data = {"prompt": "Simple workout request"}
        response = test_client.post("/api/v1/workout/prompt", json=request_data)

        assert response.status_code == 200
        response_data = response.json()
        assert_helpers.assert_valid_workout_response(response_data)

        # Verify service called with defaults
        mock_llm_service.generate_response.assert_called_once()
        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs["prompt"] == request_data["prompt"]
        assert call_args.kwargs["context"] is None

    def test_workout_prompt_endpoint_validation_error(self, test_client: TestClient):
        """Test workout prompt with invalid request data."""
        # Missing required field
        invalid_requests = [
            {},  # Empty request
            {"context": "Only context"},  # Missing prompt
            {"prompt": ""},  # Empty prompt
            {"prompt": "Valid prompt", "max_tokens": -1},  # Invalid max_tokens
            {"prompt": "Valid prompt", "temperature": 2.0},  # Invalid temperature
        ]

        for invalid_request in invalid_requests:
            response = test_client.post("/api/v1/workout/prompt", json=invalid_request)
            assert response.status_code == 422, f"Failed for request: {invalid_request}"

            response_data = response.json()
            assert "detail" in response_data

    def test_workout_prompt_endpoint_service_error(self, test_client: TestClient, mock_llm_service):
        """Test workout prompt when LLM service fails."""
        # Mock service failure
        mock_llm_service.generate_response.return_value = {
            "success": False,
            "error": "LLM service unavailable",
            "tokens_used": 0,
            "response_time": 0.1,
            "model": "test-model",
        }

        request_data = SAMPLE_WORKOUT_PROMPTS[0]
        response = test_client.post("/api/v1/workout/prompt", json=request_data)

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        response_data = response.json()
        assert "detail" in response_data
        assert "unavailable" in response_data["detail"].lower()

    def test_workout_prompt_endpoint_timeout(self, test_client: TestClient, mock_llm_service):
        """Test workout prompt with service timeout."""
        # Mock timeout response
        mock_llm_service.generate_response.return_value = {
            "success": False,
            "error": "Request timeout",
            "tokens_used": 0,
            "response_time": 30.0,
            "model": "test-model",
        }

        request_data = SAMPLE_WORKOUT_PROMPTS[0]
        response = test_client.post("/api/v1/workout/prompt", json=request_data)

        assert response.status_code == 408  # Request Timeout
        response_data = response.json()
        assert "timeout" in response_data["detail"].lower()

    def test_health_endpoint_success(self, test_client: TestClient, mock_llm_service):
        """Test successful health check."""
        # Mock healthy response
        mock_llm_service.health_check.return_value = {
            "status": "healthy",
            "model": "llama2",
            "response_time": 0.15,
        }

        response = test_client.get("/api/v1/workout/health")

        assert response.status_code == 200
        response_data = response.json()

        assert_helpers.assert_valid_health_response(response_data)
        assert response_data["status"] == "healthy"
        mock_llm_service.health_check.assert_called_once()

    def test_health_endpoint_unhealthy(self, test_client: TestClient, mock_llm_service):
        """Test health check when service is unhealthy."""
        # Mock unhealthy response
        mock_llm_service.health_check.return_value = {
            "status": "unhealthy",
            "error": "Connection refused",
            "model": "unknown",
            "response_time": 5.0,
        }

        response = test_client.get("/api/v1/workout/health")

        assert response.status_code == 503  # Service Unavailable
        response_data = response.json()
        assert response_data["status"] == "unhealthy"
        assert "error" in response_data

    def test_workout_prompt_request_response_headers(self, test_client: TestClient, mock_llm_service):
        """Test that proper headers are set in responses."""
        mock_response = generate_data.workout_response_data()
        mock_llm_service.generate_response.return_value = mock_response

        request_data = {"prompt": "Test workout"}
        response = test_client.post("/api/v1/workout/prompt", json=request_data)

        assert response.status_code == 200

        # Check content type
        assert response.headers["content-type"] == "application/json"

        # Check CORS headers (if configured)
        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] in [
                "*",
                "http://localhost:3000",
            ]

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/v1/workout/nonexistent",
            "/api/v2/workout/prompt",  # Wrong version
            "/workout/prompt",  # Missing API prefix
        ],
    )
    def test_nonexistent_endpoints(self, test_client: TestClient, endpoint):
        """Test that nonexistent endpoints return 404."""
        response = test_client.get(endpoint)
        assert response.status_code == 404

    def test_method_not_allowed(self, test_client: TestClient):
        """Test that wrong HTTP methods return 405."""
        # POST endpoint called with GET
        response = test_client.get("/api/v1/workout/prompt")
        assert response.status_code == 405

        # GET endpoint called with POST
        response = test_client.post("/api/v1/workout/health")
        assert response.status_code == 405

    def test_large_request_handling(self, test_client: TestClient, mock_llm_service):
        """Test handling of large requests."""
        mock_response = generate_data.workout_response_data()
        mock_llm_service.generate_response.return_value = mock_response

        # Create large request
        large_prompt = "Create a workout " * 1000  # Very long prompt
        large_context = "Context information " * 500  # Very long context

        request_data = {
            "prompt": large_prompt,
            "context": large_context,
            "max_tokens": 1000,
        }

        response = test_client.post("/api/v1/workout/prompt", json=request_data)

        # Should handle large requests (or return appropriate error)
        assert response.status_code in [
            200,
            413,
            422,
        ]  # Success, Payload Too Large, or Validation Error

        if response.status_code == 200:
            response_data = response.json()
            assert_helpers.assert_valid_workout_response(response_data)

    def test_concurrent_requests(self, test_client: TestClient, mock_llm_service):
        """Test handling multiple concurrent requests."""
        import threading
        import time

        mock_response = generate_data.workout_response_data()
        mock_llm_service.generate_response.return_value = mock_response

        results = []
        errors = []

        def make_request():
            try:
                request_data = {"prompt": f"Workout request {threading.current_thread().ident}"}
                response = test_client.post("/api/v1/workout/prompt", json=request_data)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        end_time = time.time()

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(status == 200 for status in results)

        # Should complete reasonably quickly
        assert end_time - start_time < 10.0  # 10 seconds max
