"""Embeddings API endpoints with Dependency Injection.

This file demonstrates the updated API router using the new DI system.
Compare this with the original router.py to see the transformation.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

# NEW: Import DI container and type hints
from app.di import container
from app.di.container import (
    EmbeddingServiceProvider,
    WeaviateClientProvider,
    DocumentRepositoryProvider,
)

# Import schemas (unchanged)
from app.api.v1.embeddings.schemas import (
    EmbedChunksRequest,
    EmbedChunksResponse,
    SearchRequest,
    SearchResponse,
    HealthCheckResponse,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# ============================================================================  
# NEW: Dependency Injection - Simplified, Type-Safe, Testable
# ============================================================================

# Before: Manual service creation with complex initialization
# async def get_embedding_service():
#     config = EmbeddingModelConfig.for_realtime()
#     service = SentenceTransformerService(config)
#     await service.initialize()
#     return service

# After: Simple, type-safe dependency injection
@router.post(
    "/embed/query",
    response_model=Dict[str, Any],
    summary="Embed query with DI",
    description="Generate embedding for user query using DI-managed service",
)
async def embed_query_with_di(
    query: str,
    # Dependencies injected automatically by DI container
    embedding_service: EmbeddingServiceProvider = Depends(
        container.external.sentence_transformer_service.provider
    ),
    settings: Settings = Depends(container.settings.provider),
):
    """Embed query endpoint demonstrating DI benefits.
    
    Benefits shown:
    1. No manual service instantiation
    2. Automatic initialization
    3. Type safety with IDE support
    4. Easy testing with dependency override
    5. Configuration injection
    """
    try:
        # Service is fully initialized and ready to use
        embedding = await embedding_service.embed_text(query)
        
        return {
            "query": query,
            "embedding": embedding,
            "model": embedding_service.model_name,
            "dimension": len(embedding),
            "environment": settings.environment,
            "di_managed": True,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}",
        )

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search embeddings with DI",
    description="Perform similarity search using DI-managed repositories",
)
async def search_embeddings_with_di(
    request: SearchRequest,
    # Multiple dependencies injected automatically
    embedding_service: EmbeddingServiceProvider = Depends(
        container.external.sentence_transformer_service.provider
    ),
    weaviate_client: WeaviateClientProvider = Depends(
        container.external.weaviate_client.provider
    ),
    embedding_repository: DocumentRepositoryProvider = Depends(
        container.repositories.embedding_repository.provider
    ),
):
    """Search endpoint demonstrating multiple dependency injection.
    
    Benefits shown:
    1. Multiple dependencies automatically injected
    2. Automatic client connection management
    3. Type-safe repository access
    4. Easy testing with selective mocking
    """
    try:
        # Query embedding (DI-managed service)
        query_embedding = await embedding_service.embed_text(request.query)
        
        # Similarity search (DI-managed repository)
        results = await embedding_repository.search_similar(
            query_vector=query_embedding,
            k=request.k,
            min_similarity=request.min_similarity,
        )
        
        return SearchResponse(
            success=True,
            query=request.query,
            total_results=len(results),
            results=[
                {
                    "document_id": result.document_id,
                    "similarity_score": result.score,
                    "content": result.content[:100] + "..." if len(result.content) > 100 else result.content,
                }
                for result in results
            ],
            processing_time_ms=0,  # Would be calculated in real implementation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )

@router.get(
    "/health/di",
    response_model=Dict[str, Any],
    summary="DI-based health check",
    description="Comprehensive health check using DI-managed services",
)
async def health_check_with_di(
    # Health check dependencies injected automatically
    weaviate_health: bool = Depends(
        container.external.weaviate_health_check.provider
    ),
    embedding_health: bool = Depends(
        container.external.embedding_service_health_check.provider
    ),
    services_health: dict = Depends(
        container.services.services_health.provider
    ),
    repositories_health: dict = Depends(
        container.repositories.repositories_health.provider
    ),
):
    """Health check endpoint using DI-managed health providers.
    
    Benefits shown:
    1. Comprehensive health monitoring
    2. Automatic dependency health checks
    3. Centralized health check logic
    4. Easy to extend with new services
    """
    return {
        "status": "healthy" if all([
            weaviate_health,
            embedding_health,
            services_health["overall"],
            repositories_health["overall"],
        ]) else "degraded",
        "checks": {
            "weaviate": {"healthy": weaviate_health},
            "embedding_service": {"healthy": embedding_health},
            "application_services": services_health,
            "repositories": repositories_health,
        },
        "di_system": {
            "status": "active",
            "providers_loaded": len(container.providers),
        },
    }

@router.get(
    "/config/di",
    summary="DI configuration",
    description="Show current DI configuration for debugging",
)
async def show_di_config(
    # Configuration injected automatically
    settings: Settings = Depends(container.settings.provider),
    embedding_config: EmbeddingModelConfig = Depends(
        container.config.realtime_embedding_config.provider
    ),
    weaviate_config: WeaviateConfig = Depends(
        container.config.weaviate_config.provider
    ),
):
    """Show DI configuration for debugging purposes.
    
    Benefits shown:
    1. Easy configuration inspection
    2. Environment-specific configs
    3. Type-safe configuration access
    4. Centralized configuration management
    """
    return {
        "environment": settings.environment,
        "debug_mode": settings.debug,
        "di_configuration": {
            "embedding_model": embedding_config.model_name,
            "embedding_device": embedding_config.device,
            "weaviate_url": weaviate_config.url,
            "weaviate_host": weaviate_config.host,
            "weaviate_port": weaviate_config.port,
        },
        "di_benefits": [
            "Automatic service initialization",
            "Type-safe dependency injection",
            "Easy testing with mocks",
            "Centralized configuration",
            "Lifecycle management",
        ],
    }

# ============================================================================
# Testing Examples (would be in separate test files)
# ============================================================================

"""
Example tests showing how easy testing becomes with DI:

def test_embed_query_with_di():
    from app.di.testing import create_test_container, TestOverrides
    from app.di import container
    
    # Create test container with mocked dependencies
    test_container = create_test_container()
    
    # Override just the embedding service
    TestOverrides.with_mock_embedding_service(
        test_container, 
        mock_embedding=[0.9, 0.8, 0.7]
    )
    
    with container.override(test_container):
        # Test with mocked embedding service
        response = client.post("/embeddings/embed/query", json={"query": "test"})
        assert response.status_code == 200
        assert response.json()["embedding"] == [0.9, 0.8, 0.7]

def test_health_check_with_service_failure():
    from app.di.testing import create_test_container
    from app.di import container
    
    test_container = create_test_container()
    
    # Override health check to simulate failure
    test_container.external.embedding_service_health_check.override(
        providers.Factory(lambda: AsyncMock(return_value=False))
    )
    
    with container.override(test_container):
        response = client.get("/embeddings/health/di")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert not response.json()["checks"]["embedding_service"]["healthy"]

Benefits demonstrated:
1. Easy mocking of individual dependencies
2. No complex patching required
3. Test isolation automatically managed
4. Clear test intent and behavior
"""
