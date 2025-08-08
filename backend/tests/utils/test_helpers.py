"""
Test helper functions and utilities.

Provides common functionality needed across multiple test files:
- Data generators
- Assertion helpers
- Mock creators
- Test environment utilities
"""

import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string of specified length."""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def random_email() -> str:
        """Generate a random email address."""
        username = TestDataGenerator.random_string(8).lower()
        domain = random.choice(["example.com", "test.org", "demo.net"])
        return f"{username}@{domain}"

    @staticmethod
    def workout_prompt_data(
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate workout prompt request data."""
        return {
            "prompt": prompt or f"Create a workout for {TestDataGenerator.random_string(5)}",
            "context": context or f"User context: {TestDataGenerator.random_string(20)}",
            "max_tokens": max_tokens or random.randint(100, 1000),
            "temperature": temperature or round(random.uniform(0.1, 1.0), 1),
        }

    @staticmethod
    def workout_response_data(
        response: Optional[str] = None,
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None,
        model: Optional[str] = None,
        success: bool = True,
    ) -> Dict[str, Any]:
        """Generate workout response data."""
        return {
            "response": response or f"Generated workout: {TestDataGenerator.random_string(100)}",
            "tokens_used": tokens_used or random.randint(50, 500),
            "response_time": response_time or round(random.uniform(0.1, 5.0), 2),
            "model": model or f"test-model-{TestDataGenerator.random_string(3)}",
            "success": success,
        }

    @staticmethod
    def health_response_data(
        status: str = "healthy",
        model: Optional[str] = None,
        response_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate health check response data."""
        return {
            "status": status,
            "model": model or f"health-model-{TestDataGenerator.random_string(3)}",
            "response_time": response_time or round(random.uniform(0.01, 0.5), 3),
        }


class MockCreators:
    """Create mock objects for testing."""

    @staticmethod
    def mock_llm_service(
        generate_response_return: Optional[Dict] = None,
        health_check_return: Optional[Dict] = None,
        side_effect: Optional[Exception] = None,
    ) -> AsyncMock:
        """Create a mock LLM service."""
        mock_service = AsyncMock()

        if side_effect:
            mock_service.generate_response.side_effect = side_effect
            mock_service.health_check.side_effect = side_effect
        else:
            mock_service.generate_response.return_value = (
                generate_response_return or TestDataGenerator.workout_response_data()
            )
            mock_service.health_check.return_value = health_check_return or TestDataGenerator.health_response_data()

        return mock_service

    @staticmethod
    def mock_http_response(
        status_code: int = 200,
        json_data: Optional[Dict] = None,
        text: Optional[str] = None,
    ) -> MagicMock:
        """Create a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        mock_response.text = text or json.dumps(json_data or {})
        return mock_response


class AssertionHelpers:
    """Helper functions for test assertions."""

    @staticmethod
    def assert_valid_workout_response(response_data: Dict[str, Any]):
        """Assert that response data is a valid workout response."""
        required_fields = [
            "response",
            "tokens_used",
            "response_time",
            "model",
            "success",
        ]

        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"

        assert isinstance(response_data["response"], str), "Response must be a string"
        assert len(response_data["response"]) > 0, "Response cannot be empty"
        assert isinstance(response_data["tokens_used"], int), "Tokens used must be an integer"
        assert response_data["tokens_used"] > 0, "Tokens used must be positive"
        assert isinstance(response_data["response_time"], (int, float)), "Response time must be numeric"
        assert response_data["response_time"] > 0, "Response time must be positive"
        assert isinstance(response_data["model"], str), "Model must be a string"
        assert isinstance(response_data["success"], bool), "Success must be a boolean"

    @staticmethod
    def assert_valid_health_response(response_data: Dict[str, Any]):
        """Assert that response data is a valid health response."""
        required_fields = ["status", "model", "response_time"]

        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"

        assert response_data["status"] in [
            "healthy",
            "unhealthy",
        ], "Status must be 'healthy' or 'unhealthy'"
        assert isinstance(response_data["model"], str), "Model must be a string"
        assert isinstance(response_data["response_time"], (int, float)), "Response time must be numeric"
        assert response_data["response_time"] >= 0, "Response time must be non-negative"

    @staticmethod
    def assert_api_error_response(response_data: Dict[str, Any], expected_status: int = None):
        """Assert that response data is a valid API error response."""
        required_fields = ["detail", "status_code"]

        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"

        assert isinstance(response_data["detail"], str), "Detail must be a string"
        assert isinstance(response_data["status_code"], int), "Status code must be an integer"

        if expected_status:
            assert (
                response_data["status_code"] == expected_status
            ), f"Expected status {expected_status}, got {response_data['status_code']}"


class TestEnvironment:
    """Utilities for test environment setup and teardown."""

    @staticmethod
    def setup_test_env_vars(
        env_overrides: Dict[str, str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Set up test environment variables.
        Returns original values for restoration.
        """
        import os

        default_test_env = {
            "ENVIRONMENT": "test",
            "DEBUG": "true",
            "DATABASE_URL": "sqlite:///./test.db",
            "LLM_BASE_URL": "http://localhost:11434",
            "LLM_MODEL": "test-model",
            "SECRET_KEY": "test-secret-key-minimum-32-chars",
            "LOG_LEVEL": "DEBUG",
        }

        if env_overrides:
            default_test_env.update(env_overrides)

        original_values = {}
        for key, value in default_test_env.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        return original_values

    @staticmethod
    def restore_env_vars(original_values: Dict[str, Optional[str]]):
        """Restore original environment variables."""
        import os

        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @staticmethod
    def create_temp_file(content: str, suffix: str = ".txt") -> str:
        """Create a temporary file with given content."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name

    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up a temporary file."""
        from pathlib import Path

        temp_file = Path(file_path)
        if temp_file.exists():
            temp_file.unlink()


# Convenience exports
generate_data = TestDataGenerator()
create_mocks = MockCreators()
assert_helpers = AssertionHelpers()
test_env = TestEnvironment()
