"""FitVise Dependency Injection System.

This module provides a unified dependency injection system for the FitVise application.
It replaces scattered dependency patterns with a centralized, testable, and maintainable
container using the dependency-injector library.

## Usage

### Basic Usage
```python
from app.di import container, bootstrap

# Bootstrap the application
app = bootstrap.create_fastapi_app()

# Get services directly
settings = container.settings()
llm_service = container.llm_service()

# In FastAPI endpoints
@router.get("/health")
async def health_check(
    llm_service: LLMService = Depends(container.llm_service.provider)
):
    return await llm_service.health_check()
```

### Testing Usage
```python
from app.di.testing import create_test_container
from app.di import bootstrap

def test_endpoint():
    # Create test container with mocked dependencies
    test_container = create_test_container()
    
    # Override the main container
    with container.override(test_container):
        app = bootstrap.create_fastapi_app()
        # Test with mocked dependencies
```

### Configuration
The container automatically loads configuration from environment variables
and provides typed access to all settings.

## Architecture

The DI system consists of:
- **Container**: Central dependency container with all providers
- **Providers**: Individual dependency providers (config, services, repositories)
- **Bootstrap**: Application startup and configuration utilities
- **Testing**: Test-specific container with mock providers

## Migration

This system provides a gradual migration path from existing dependency patterns.
See the migration guide for detailed steps.
"""

from app.di.container import FitviseContainer
from app.di.bootstrap import bootstrap_app, create_fastapi_app

# Create global container instance
container = FitviseContainer()

# Public API
__all__ = [
    "FitviseContainer",
    "container", 
    "bootstrap_app",
    "create_fastapi_app",
]
