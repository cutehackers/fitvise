"""
Global test configuration and fixtures for pytest.

This file contains:
- Global pytest fixtures
- Test client setup
- Database fixtures
- Mock configurations
- Environment setup for testing
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from app.core.settings import Settings

# Try to import app, but allow tests to run without it
# (e.g., for isolated unit tests like table_serialization)
try:
    from app.main import app
except ImportError:
    app = None


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Create test-specific settings.
    Override environment variables for testing.
    """
    # Set test environment variables
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///./test.db",
        "LLM_BASE_URL": "http://localhost:11434",
        "LLM_MODEL": "llama2",
        "SECRET_KEY": "test-secret-key-min-32-chars-long",
        "LOG_LEVEL": "DEBUG",
        "WEAVIATE_URL": "http://localhost:8080",
        "WEAVIATE_API_KEY": "test-api-key",
    }

    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    settings = Settings()

    yield settings

    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def test_client(test_settings: Settings) -> Generator[TestClient, None, None]:
    """
    Create a test client for the FastAPI application.
    Uses the test settings configuration.
    """
    if app is None:
        pytest.skip("FastAPI app not available (missing dependencies)")
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
async def async_test_client(
    test_settings: Settings,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Create an async test client for the FastAPI application.
    Useful for testing async endpoints and streaming responses.
    """
    if app is None:
        pytest.skip("FastAPI app not available (missing dependencies)")
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
def mock_llm_service():
    """
    Mock the LLM service for unit tests.
    Prevents external API calls during testing.
    """
    from unittest.mock import AsyncMock, patch

    with patch("app.application.llm_service.LLMService") as mock_service:
        mock_instance = AsyncMock()
        mock_service.return_value = mock_instance

        # Default mock responses
        mock_instance.generate_response.return_value = {
            "response": "Mocked workout response",
            "tokens_used": 100,
            "response_time": 0.5,
            "model": "mocked-model",
        }

        mock_instance.health_check.return_value = {
            "status": "healthy",
            "model": "mocked-model",
            "response_time": 0.1,
        }

        yield mock_instance


@pytest.fixture(scope="function")
def sample_workout_request():
    """Sample workout request data for testing."""
    return {
        "prompt": "Create a 30-minute cardio workout for beginners",
        "context": "User is a beginner with no equipment",
        "max_tokens": 500,
        "temperature": 0.7,
    }


@pytest.fixture(scope="function")
def sample_workout_response():
    """Sample workout response data for testing."""
    return {
        "response": "Here's a 30-minute beginner cardio workout:\n\n1. Warm-up (5 minutes)\n2. Jumping jacks (3 sets)\n3. High knees (3 sets)\n4. Cool-down (5 minutes)",
        "tokens_used": 150,
        "response_time": 1.2,
        "model": "llama2",
        "success": True,
    }


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """
    Automatically cleanup test files after each test.
    Runs after every test function.
    """
    yield  # Run the test

    # Cleanup test database
    test_db_path = Path("test.db")
    if test_db_path.exists():
        test_db_path.unlink()

    # Cleanup any test uploads
    test_upload_dir = Path("test_uploads")
    if test_upload_dir.exists():
        import shutil

        shutil.rmtree(test_upload_dir)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests (deselect with '-m \"not unit\"')")
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (deselect with '-m \"not e2e\"')",
    )
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "external: marks tests that require external services")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Mark external service tests
        if "external" in item.name or "llm" in item.name.lower():
            item.add_marker(pytest.mark.external)
