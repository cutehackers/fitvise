"""Application Bootstrap.

This module provides utilities for bootstrapping the FitVise application
with proper dependency injection setup and lifecycle management.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.di import container
from app.di.container import FitviseContainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown of the application with proper resource management.
    
    Args:
        app: FastAPI application instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Startup
        logger.info("🚀 Starting FitVise application...")
        
        # Initialize container resources
        await container.init_resources()
        
        # Initialize critical services
        await container.external.init_weaviate_client()
        await container.external.init_sentence_transformer()
        await container.external.init_ollama_service()
        
        logger.info("✅ Application startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🔄 Shutting down FitVise application...")
        
        try:
            # Shutdown container resources
            await container.shutdown_resources()
            logger.info("✅ Application shutdown complete")
        except Exception as e:
            logger.error(f"❌ Application shutdown failed: {e}")


def create_fastapi_app(
    container_instance: Optional[FitviseContainer] = None,
    debug: Optional[bool] = None,
) -> FastAPI:
    """Create and configure FastAPI application with dependency injection.
    
    Args:
        container_instance: Optional DI container instance (defaults to global container)
        debug: Optional debug mode override
        
    Returns:
        Configured FastAPI application
    """
    # Use provided container or global container
    di_container = container_instance or container
    
    # Get settings
    settings = di_container.config.settings()
    
    # Override debug if provided
    if debug is not None:
        settings.debug = debug
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    # Wire dependency injection for API modules
    from app.api.v1.embeddings import router as embeddings_module
    from app.api.v1.rag import ingestion as rag_ingestion_module
    from app.api.v1.rag import search as rag_search_module
    from app.api.v1.rag import data_sources as rag_data_sources_module
    from app.api.v1.rag import storage as rag_storage_module

    di_container.wire(
        embeddings_module,
        rag_ingestion_module,
        rag_search_module,
        rag_data_sources_module,
        rag_storage_module,
    )
    
    # Configure CORS
    configure_cors(app, settings)
    
    # Configure middleware
    configure_middleware(app, settings)
    
    # Configure routes
    configure_routes(app, di_container)
    
    # Configure exception handlers
    configure_exception_handlers(app)
    
    return app


def configure_cors(app: FastAPI, settings) -> None:
    """Configure CORS middleware.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods_list,
        allow_headers=settings.cors_allow_headers_list,
    )


def configure_middleware(app: FastAPI, settings) -> None:
    """Configure application middleware.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # Add request logging middleware
    if settings.debug:
        from fastapi.middleware import Middleware
        from fastapi.middleware.base import BaseHTTPMiddleware
        
        class RequestLoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                logger = logging.getLogger("fastapi.request")
                logger.info(f"📥 {request.method} {request.url.path}")
                response = await call_next(request)
                logger.info(f"📤 {response.status_code} {request.method} {request.url.path}")
                return response
        
        app.add_middleware(RequestLoggingMiddleware)
    
    # Add other middleware as needed
    # Example: rate limiting, authentication, etc.


def configure_routes(app: FastAPI, container: FitviseContainer) -> None:
    """Configure application routes.
    
    Args:
        app: FastAPI application instance
        container: DI container instance
    """
    # Import and include routers
    from app.api.v1.embeddings.router import router as embeddings_router
    from app.api.v1.rag.ingestion import router as ingestion_router
    from app.api.v1.rag.search import router as search_router
    from app.api.v1.rag.data_sources import router as data_sources_router
    from app.api.v1.rag.storage import router as storage_router
    
    # Include API v1 routes
    app.include_router(
        embeddings_router,
        prefix=container.config.settings().api_v1_prefix,
    )
    
    app.include_router(
        ingestion_router,
        prefix=container.config.settings().api_v1_prefix,
    )
    
    app.include_router(
        search_router,
        prefix=container.config.settings().api_v1_prefix,
    )
    
    app.include_router(
        data_sources_router,
        prefix=container.config.settings().api_v1_prefix,
    )
    
    app.include_router(
        storage_router,
        prefix=container.config.settings().api_v1_prefix,
    )
    
    # Add health check endpoint
    configure_health_routes(app, container)


def configure_health_routes(app: FastAPI, container: FitviseContainer) -> None:
    """Configure health check routes.
    
    Args:
        app: FastAPI application instance
        container: DI container instance
    """
    from fastapi import Depends
    
    @app.get("/health", tags=["health"])
    async def health_check(
        services_health: dict = Depends(container.services.services_health_check.provider),
        external_health: dict = Depends(container.external.external_services_health_check.provider),
        repositories_health: dict = Depends(container.repositories.repositories_health_check.provider),
    ):
        """Comprehensive health check endpoint."""
        return {
            "status": "healthy" if all([
                services_health["overall"],
                external_health["overall"],
                repositories_health["overall"],
            ]) else "degraded",
            "services": services_health,
            "external": external_health,
            "repositories": repositories_health,
        }


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    from fastapi import Request, HTTPException
    from fastapi.responses import JSONResponse
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": "http_exception",
                    "status_code": exc.status_code,
                }
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_server_error",
                    "status_code": 500,
                }
            },
        )


async def bootstrap_app(
    container_instance: Optional[FitviseContainer] = None,
    debug: Optional[bool] = None,
) -> FastAPI:
    """Bootstrap the application with full initialization.
    
    This function creates and fully initializes the FastAPI application
    with all dependencies and resources.
    
    Args:
        container_instance: Optional DI container instance
        debug: Optional debug mode override
        
    Returns:
        Fully initialized FastAPI application
    """
    # Create application
    app = create_fastapi_app(container_instance, debug)
    
    # Run additional initialization if needed
    # This is where you can add startup tasks like database migrations,
    # cache warming, etc.
    
    return app


def create_test_app(
    container_instance: Optional[FitviseContainer] = None,
) -> FastAPI:
    """Create FastAPI application for testing.
    
    Creates a test-specific application with appropriate test configuration.
    
    Args:
        container_instance: Optional test container instance
        
    Returns:
        FastAPI application configured for testing
    """
    from app.di.testing import create_test_container
    
    # Create test container if not provided
    test_container = container_instance or create_test_container()
    
    # Create app with test container
    app = create_fastapi_app(test_container, debug=True)
    
    # Add test-specific routes
    @app.get("/test/reset", tags=["test"])
    async def reset_test_state():
        """Reset test state endpoint."""
        # Reset test data, clear caches, etc.
        return {"status": "reset"}
    
    return app


# Environment-specific bootstrap functions
async def create_production_app() -> FastAPI:
    """Create application for production environment."""
    return await bootstrap_app(debug=False)


async def create_development_app() -> FastAPI:
    """Create application for development environment."""
    return await bootstrap_app(debug=True)


async def create_staging_app() -> FastAPI:
    """Create application for staging environment."""
    return await bootstrap_app(debug=False)
