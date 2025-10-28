"""API tests for embeddings endpoints (Task 2.2.1).

Tests cover:
- HTTP status codes and error handling
- Request/response validation
- Authentication and authorization (if implemented)
- API contract compliance
- OpenAPI schema validation
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)


class TestEmbeddingsHealthEndpoint:
    """Test /api/v1/embeddings/health endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service, \
             patch("app.api.v1.embeddings.router.get_embedding_repository") as mock_repo:

            # Mock healthy services
            service = AsyncMock()
            service.is_loaded = True
            service.model_name = "test-model"

            async def mock_health():
                return True

            service.health_check = mock_health
            mock_service.return_value = service

            repo = AsyncMock()
            repo.health_check = mock_health
            mock_repo.return_value = repo

            response = client.get("/api/v1/embeddings/health")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "embedding_service" in data
            assert "weaviate" in data


class TestEmbedQueryEndpoint:
    """Test POST /api/v1/embeddings/embed/query endpoint."""

    def test_embed_query_success(self, client):
        """Test successful query embedding."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service, \
             patch("app.api.v1.embeddings.router.get_embedding_repository"), \
             patch("app.api.v1.embeddings.router.get_embedding_domain_service"):

            # Mock embedding service
            service = AsyncMock()

            async def mock_initialize():
                pass

            async def mock_embed_query(query, use_cache=True):
                from app.domain.value_objects.embedding_vector import EmbeddingVector
                return EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))

            async def mock_cache_stats():
                return {"cache_size": 0, "cache_hits": 0}

            service.initialize = mock_initialize
            service.embed_query = mock_embed_query
            service.get_cache_stats = mock_cache_stats
            mock_service.return_value = service

            payload = {
                "query": "What exercises for lower back pain?",
                "use_cache": True,
                "store_embedding": False,
            }

            response = client.post("/api/v1/embeddings/embed/query", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "query_id" in data
            assert data["vector_dimension"] == 384

    def test_embed_query_empty_text(self, client):
        """Test embedding empty query returns validation error."""
        payload = {
            "query": "",
        }

        response = client.post("/api/v1/embeddings/embed/query", json=payload)

        # FastAPI validation should catch this
        assert response.status_code == 422

    def test_embed_query_invalid_request(self, client):
        """Test invalid request structure."""
        payload = {
            "invalid_field": "value",
        }

        response = client.post("/api/v1/embeddings/embed/query", json=payload)

        assert response.status_code == 422  # Validation error


class TestEmbedChunksEndpoint:
    """Test POST /api/v1/embeddings/embed/chunks endpoint."""

    def test_embed_chunks_success(self, client):
        """Test successful chunk embedding."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service, \
             patch("app.api.v1.embeddings.router.get_embedding_repository"), \
             patch("app.api.v1.embeddings.router.get_embedding_domain_service"):

            # Mock services
            service = AsyncMock()

            async def mock_initialize():
                pass

            async def mock_embed_batch(texts, batch_size=32, show_progress=True):
                from app.domain.value_objects.embedding_vector import EmbeddingVector
                return [
                    EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))
                    for _ in texts
                ]

            service.initialize = mock_initialize
            service.embed_batch = mock_embed_batch
            mock_service.return_value = service

            chunk_id = str(uuid4())
            doc_id = str(uuid4())

            payload = {
                "chunks": [
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "text": "Exercise text content",
                        "sequence": 0,
                    }
                ],
                "batch_size": 32,
                "store_embeddings": False,
            }

            response = client.post("/api/v1/embeddings/embed/chunks", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_chunks"] == 1
            assert data["embedded_count"] == 1

    def test_embed_chunks_empty_list(self, client):
        """Test embedding empty chunks list returns validation error."""
        payload = {
            "chunks": [],
        }

        response = client.post("/api/v1/embeddings/embed/chunks", json=payload)

        assert response.status_code == 422  # Validation error


class TestBatchEmbedEndpoint:
    """Test POST /api/v1/embeddings/embed/batch endpoint."""

    def test_batch_embed_with_texts(self, client):
        """Test batch embedding with text list."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service, \
             patch("app.api.v1.embeddings.router.get_embedding_repository"), \
             patch("app.api.v1.embeddings.router.get_embedding_domain_service"):

            # Mock services
            service = AsyncMock()

            async def mock_initialize():
                pass

            async def mock_embed_batch(texts, batch_size=32, show_progress=True):
                from app.domain.value_objects.embedding_vector import EmbeddingVector
                return [
                    EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))
                    for _ in texts
                ]

            service.initialize = mock_initialize
            service.embed_batch = mock_embed_batch
            mock_service.return_value = service

            payload = {
                "texts": ["Text 1", "Text 2", "Text 3"],
                "batch_size": 32,
                "store_embeddings": False,
            }

            response = client.post("/api/v1/embeddings/embed/batch", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_items"] == 3

    def test_batch_embed_missing_input(self, client):
        """Test batch embedding without texts or text_items."""
        payload = {
            "batch_size": 32,
        }

        response = client.post("/api/v1/embeddings/embed/batch", json=payload)

        assert response.status_code == 422  # Validation error


class TestSearchEndpoint:
    """Test POST /api/v1/embeddings/search endpoint."""

    def test_search_by_query(self, client):
        """Test search by query text."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service, \
             patch("app.api.v1.embeddings.router.get_embedding_repository") as mock_repo, \
             patch("app.api.v1.embeddings.router.get_embedding_domain_service"):

            # Mock services
            service = AsyncMock()

            async def mock_initialize():
                pass

            async def mock_embed_query(query, use_cache=True):
                from app.domain.value_objects.embedding_vector import EmbeddingVector
                return EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))

            async def mock_cache_stats():
                return {"cache_size": 0}

            service.initialize = mock_initialize
            service.embed_query = mock_embed_query
            service.get_cache_stats = mock_cache_stats
            mock_service.return_value = service

            # Mock repository
            repo = AsyncMock()

            async def mock_similarity_search(query_vector, k, filters=None, min_similarity=0.0):
                return []  # Empty results

            repo.similarity_search = mock_similarity_search
            mock_repo.return_value = repo

            payload = {
                "query": "Lower back exercises",
                "k": 10,
                "min_similarity": 0.7,
            }

            response = client.post("/api/v1/embeddings/search", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "total_results" in data
            assert "results" in data

    def test_search_missing_query_input(self, client):
        """Test search without query or query_vector."""
        payload = {
            "k": 10,
        }

        response = client.post("/api/v1/embeddings/search", json=payload)

        assert response.status_code == 422  # Validation error

    def test_search_invalid_k_value(self, client):
        """Test search with invalid k value."""
        payload = {
            "query": "test",
            "k": 0,  # Invalid: must be >= 1
        }

        response = client.post("/api/v1/embeddings/search", json=payload)

        assert response.status_code == 422  # Validation error


class TestSetupEndpoint:
    """Test POST /api/v1/embeddings/setup endpoint."""

    def test_setup_infrastructure(self, client):
        """Test setup infrastructure endpoint."""
        payload = {
            "vector_dimension": 384,
            "recreate_schema": False,
        }

        # Note: This would require mocking the entire setup use case
        # For now, just verify endpoint exists and accepts requests
        response = client.post("/api/v1/embeddings/setup", json=payload)

        # May fail if actual services aren't running, but should not be 404
        assert response.status_code != 404


class TestAPIValidation:
    """Test API request/response validation."""

    def test_embed_query_response_schema(self, client):
        """Test that response matches expected schema."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service, \
             patch("app.api.v1.embeddings.router.get_embedding_repository"), \
             patch("app.api.v1.embeddings.router.get_embedding_domain_service"):

            service = AsyncMock()

            async def mock_initialize():
                pass

            async def mock_embed_query(query, use_cache=True):
                from app.domain.value_objects.embedding_vector import EmbeddingVector
                return EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))

            async def mock_cache_stats():
                return {"cache_size": 0, "cache_hits": 0}

            service.initialize = mock_initialize
            service.embed_query = mock_embed_query
            service.get_cache_stats = mock_cache_stats
            mock_service.return_value = service

            payload = {"query": "Test"}

            response = client.post("/api/v1/embeddings/embed/query", json=payload)

            assert response.status_code == 200
            data = response.json()

            # Verify all required fields are present
            required_fields = [
                "success",
                "query_id",
                "vector_dimension",
                "processing_time_ms",
                "cache_hit",
                "stored",
            ]

            for field in required_fields:
                assert field in data, f"Missing required field: {field}"


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_internal_server_error_handling(self, client):
        """Test handling of internal server errors."""
        with patch("app.api.v1.embeddings.router.get_embedding_service") as mock_service:
            # Mock service to raise exception
            service = AsyncMock()

            async def mock_error():
                raise Exception("Simulated error")

            service.initialize = mock_error
            mock_service.return_value = service

            payload = {"query": "Test"}

            response = client.post("/api/v1/embeddings/embed/query", json=payload)

            # Should handle gracefully, returning 500
            assert response.status_code == 500


class TestAPIDocumentation:
    """Test API documentation and OpenAPI schema."""

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/embeddings/embed/query" in schema["paths"]

    def test_docs_endpoint_available(self, client):
        """Test that /docs endpoint is available (in non-production)."""
        response = client.get("/docs")

        # Should be available in development
        assert response.status_code in [200, 307]  # 307 = redirect
