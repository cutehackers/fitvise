"""FitVise Application Runner.

This is the main entry point for running the FitVise backend application.
It provides a clean bootstrap interface with proper environment detection
and dependency injection setup.
"""

import asyncio
import logging
from pathlib import Path

import uvicorn

from app.di import bootstrap, container

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_app_for_environment():
    """Create application based on environment configuration.
    
    Returns:
        FastAPI application configured for current environment
    """
    settings = container.settings()
    
    if settings.environment == "production":
        return asyncio.run(bootstrap.create_production_app())
    elif settings.environment == "staging":
        return asyncio.run(bootstrap.create_staging_app())
    else:
        # Default to development
        return asyncio.run(bootstrap.create_development_app())


def print_startup_info():
    """Print application startup information."""
    settings = container.settings()
    
    print("🌪️  Starting Fitvise backend server...")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🌐 API will be available at: http://{settings.api_host}:{settings.api_port}")
    print(f"📜 Interactive docs at: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔄 Auto-reload: {'enabled' if settings.debug else 'disabled'}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print(f"🔢 Embedding Model: {settings.sentence_transformer_model}")
    print(f"📊 DI System: Active with {len(container.providers)} providers")
    print(f"💾 Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else 'SQLite'}")
    print(f"🔍 Weaviate: {settings.weaviate_url}")
    print("-" * 60)


def run_with_legacy_config():
    """Run application with legacy configuration support.
    
    This function provides backward compatibility during migration.
    """
    import sys
    
    # Check for legacy configuration
    if len(sys.argv) > 1:
        if sys.argv[1] == "--legacy":
            logger.warning("🔄 Running with legacy configuration")
            
            # Use old pattern for legacy mode
            from app.core.settings import settings as legacy_settings
            
            uvicorn.run(
                "app.main:app",
                host=legacy_settings.api_host,
                port=legacy_settings.api_port,
                reload=legacy_settings.debug,
                log_level=legacy_settings.log_level.lower(),
                access_log=True,
            )
            return
        
        elif sys.argv[1] == "--test":
            logger.info("🧪 Running in test mode")
            
            # Create test application
            app = asyncio.run(bootstrap.create_test_app())
            
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=8001,
                reload=True,
                log_level="debug",
                access_log=True,
            )
            return
        
        elif sys.argv[1] == "--help":
            print_help()
            return
    
    # Default: run with DI system
    run_with_di()


def print_help():
    """Print help information."""
    print("FitVise Backend Runner")
    print("Usage: python run.py [OPTIONS]")
    print("")
    print("Options:")
    print("  --legacy    Run with legacy configuration (deprecated)")
    print("  --test      Run in test mode with test container")
    print("  --help      Show this help message")
    print("")
    print("Environment Variables:")
    print("  ENVIRONMENT=local|staging|production    Set environment")
    print("  DEBUG=true|false                        Enable debug mode")
    print("  API_HOST=hostname                       Set API host")
    print("  API_PORT=port                          Set API port")
    print("")
    print("Examples:")
    print("  python run.py                    # Run with DI system")
    print("  python run.py --legacy           # Run with legacy pattern")
    print("  python run.py --test             # Run in test mode")
    print("  ENVIRONMENT=production python run.py  # Run in production")


def run_with_di():
    """Run application with DI system."""
    settings = container.settings()
    
    print_startup_info()
    
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
        # Add uvicorn-specific optimizations
        loop="uvloop" if settings.environment == "production" else "asyncio",
        http="httptools",
        ws="websockets",
        # Limit concurrent connections for production
        limit_concurrency=100 if settings.environment == "production" else None,
        timeout_keep_alive=30 if settings.environment == "production" else 5,
    )


def run_with_docker():
    """Run application optimized for Docker environment."""
    import os
    
    # Docker-specific configuration
    settings = container.settings()
    
    # Override settings for Docker
    settings.api_host = os.getenv("API_HOST", "0.0.0.0")
    settings.api_port = int(os.getenv("API_PORT", "8000"))
    settings.debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print_startup_info()
    print("🐳 Docker mode: Optimized for container environment")
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Never reload in Docker
        log_level=settings.log_level.lower(),
        access_log=True,
        # Docker optimizations
        workers=1,  # Single worker for async app
        limit_max_requests=1000,  # Restart worker after 1000 requests
        limit_max_requests_jitter=50,  # Add randomness to restart timing
    )


def main():
    """Main entry point."""
    import os
    
    # Check if running in Docker
    if os.getenv("DOCKER_ENV") == "true":
        run_with_docker()
    else:
        run_with_legacy_config()


if __name__ == "__main__":
    main()
