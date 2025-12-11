"""FitVise Main Application.

This is the main entry point for the FitVise FastAPI application.
It sets up the application with dependency injection, middleware, routing,
and other core configuration.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.di import bootstrap
from app.di.container import container

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown of the application with proper resource management.
    This replaces the old manual initialization pattern with DI-managed lifecycle.
    """
    try:
        # Startup
        logger.info("🚀 Starting FitVise application...")
        
        # Initialize container resources (DI-managed)
        await container.init_resources()
        
        # Initialize critical services through DI
        await container.external.init_weaviate_client()
        await container.external.init_sentence_transformer()
        await container.external.init_ollama_service()
        
        # Log successful startup
        settings = container.settings()
        logger.info(f"✅ Application startup complete")
        logger.info(f"🌍 Environment: {settings.environment}")
        logger.info(f"🤖 LLM Model: {settings.llm_model}")
        logger.info(f"🔢 Embedding Model: {settings.sentence_transformer_model}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🔄 Shutting down FitVise application...")
        
        try:
            # Shutdown container resources (DI-managed)
            await container.shutdown_resources()
            logger.info("✅ Application shutdown complete")
        except Exception as e:
            logger.error(f"❌ Application shutdown failed: {e}")


# Create FastAPI application using DI bootstrap
app = bootstrap.create_fastapi_app(container)


# Legacy compatibility: maintain old manual pattern during transition
# This can be removed once migration is complete

# Old settings import (deprecated - use container.settings() instead)
try:
    from app.core.settings import get_settings
    _legacy_settings = get_settings()
except ImportError:
    _legacy_settings = None

# Old manual initialization (deprecated - use DI-managed lifecycle instead)
async def legacy_init_resources():
    """Legacy resource initialization (deprecated).
    
    This function is maintained for backward compatibility during migration.
    New code should use the DI-managed lifecycle instead.
    """
    logger.warning("🔄 Using legacy initialization - consider migrating to DI pattern")
    
    # Initialize Weaviate client
    try:
        weaviate_client = container.weaviate_client()
        if not weaviate_client.is_connected:
            await weaviate_client.connect()
        logger.info("✅ Weaviate client connected")
    except Exception as e:
        logger.error(f"❌ Weaviate initialization failed: {e}")
    
    # Initialize embedding service
    try:
        embedding_service = container.sentence_transformer_service()
        await embedding_service.initialize()
        logger.info("✅ Embedding service initialized")
    except Exception as e:
        logger.error(f"❌ Embedding service initialization failed: {e}")
    
    # Initialize LLM service
    try:
        llm_service = container.llm_service()
        await llm_service.initialize()
        logger.info("✅ LLM service initialized")
    except Exception as e:
        logger.error(f"❌ LLM service initialization failed: {e}")


# Add legacy health check endpoint (deprecated - use new /health endpoint)
@app.get("/legacy/health", tags=["legacy"])
async def legacy_health_check():
    """Legacy health check endpoint (deprecated).
    
    This endpoint is maintained for backward compatibility during migration.
    New code should use the new /health endpoint with comprehensive checks.
    """
    logger.warning("🔄 Using legacy health check - consider migrating to DI health endpoint")
    
    try:
        # Use DI-managed services for health checks
        weaviate_healthy = await container.external.weaviate_health_check()
        embedding_healthy = await container.external.embedding_service_health_check()
        ollama_healthy = await container.external.ollama_health_check()
        
        overall_healthy = weaviate_healthy and embedding_healthy and ollama_healthy
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "services": {
                "weaviate": weaviate_healthy,
                "embedding_service": embedding_healthy,
                "ollama": ollama_healthy,
            },
            "message": "Legacy health check - consider migrating to /health endpoint",
        }
        
    except Exception as e:
        logger.error(f"Legacy health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Legacy health check failed",
        }


# Migration helper endpoint
@app.get("/migration/status", tags=["migration"])
async def migration_status():
    """Check migration status from legacy patterns to DI.
    
    This endpoint helps track progress during the migration from
    manual dependency management to the DI system.
    """
    settings = container.settings()
    
    return {
        "di_system": {
            "status": "active",
            "container_initialized": True,
            "providers_count": len(container.providers),
        },
        "legacy_system": {
            "status": "deprecated",
            "global_settings_available": _legacy_settings is not None,
            "legacy_endpoints": ["/legacy/health"],
        },
        "environment": settings.environment,
        "recommendations": [
            "Replace manual service instantiation with DI providers",
            "Use container.settings() instead of get_settings()",
            "Replace legacy endpoints with new DI-managed endpoints",
            "Update tests to use test container from app.di.testing",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with legacy uvicorn pattern for development
    settings = container.settings()
    
    print("🌪️  Starting Fitvise backend server...")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🌐 API will be available at: http://{settings.api_host}:{settings.api_port}")
    print(f"📜 Interactive docs at: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔄 Auto-reload: {'enabled' if settings.debug else 'disabled'}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print(f"📊 DI System: Active")
    print("-" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
