"""Main FitVise DI Container.

This module defines the main dependency injection container that provides
access to all application services, repositories, and configuration.
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from app.di.providers.config import ConfigProviders
from app.di.providers.external import ExternalServiceProviders
from app.di.providers.repositories import RepositoryProviders
from app.di.providers.services import ServiceProviders


class FitviseContainer(containers.DeclarativeContainer):
    """Main dependency injection container for FitVise application.

    This container provides centralized dependency management for all
    application components including configuration, services, repositories,
    and external integrations.

    ## Architecture

    The container is organized into logical provider groups:
    - **config**: Configuration and settings providers
    - **repositories**: Data access layer providers
    - **services**: Business logic and application service providers
    - **external**: External service integration providers

    ## Usage

    ```python
    from app.di import container

    # Direct access to services
    settings = container.settings()
    llm_service = container.llm_service()
    
    # FastAPI dependency injection
    @router.get("/health")
    async def health_check(
        llm_service: LLMService = Depends(container.llm_service.provider)
    ):
        return await llm_service.health_check()
    ```

    ## Lifecycle

    All providers are configured with appropriate scopes:
    - **singleton**: Created once per container lifecycle
    - **transient**: Created on each request
    - **scoped**: Created once per web request (when applicable)

    ## Testing

    For testing, override providers with test implementations:
    ```python
    from app.di.testing import create_test_container
    
    test_container = create_test_container()
    with container.override(test_container):
        # Test with mocked dependencies
    ```
    """

    # Configuration providers
    config = providers.Container(ConfigProviders)
    
    # External service providers (depends on config)
    external = providers.Container(
        ExternalServiceProviders,
        config=config,
    )
    
    # Repository providers (depends on config and external services)
    repositories = providers.Container(
        RepositoryProviders,
        config=config,
        external_services=external,
    )
    
    # Service providers (depends on repositories and external services)
    services = providers.Container(
        ServiceProviders,
        config=config,
        repositories=repositories,
        external_services=external,
    )

    # Convenience shortcuts for commonly used services
    settings = config.settings
    llm_service = services.llm_service
    embedding_service = services.embedding_service
    weaviate_client = external.weaviate_client
    document_repository = repositories.document_repository

    def __init__(self):
        """Initialize the container."""
        super().__init__()
        self._wired_modules = set()

    def wire(self, *modules):
        """Wire dependency injection to specified modules.

        Args:
            *modules: Modules to wire with dependency injection
        """
        from dependency_injector.wiring import wire

        for module in modules:
            if module not in self._wired_modules:
                wire(self, module)
                self._wired_modules.add(module)

    def unwire(self, *modules):
        """Unwire dependency injection from specified modules.

        Args:
            *modules: Modules to unwire
        """
        from dependency_injector.wiring import unwire

        for module in modules:
            if module in self._wired_modules:
                unwire(module)
                self._wired_modules.remove(module)

    async def init_resources(self):
        """Initialize all container resources.

        This method should be called during application startup to ensure
        all resources are properly initialized.
        """
        # Initialize resource providers
        if hasattr(self.external, 'init_resources'):
            await self.external.init_resources()
        if hasattr(self.services, 'init_resources'):
            await self.services.init_resources()
        # Initialize critical external services
        if hasattr(self.external, "init_weaviate_client"):
            await self.external.init_weaviate_client()
        if hasattr(self.external, "init_sentence_transformer"):
            await self.external.init_sentence_transformer()
        if hasattr(self.external, "init_ollama_service"):
            await self.external.init_ollama_service()

    async def shutdown_resources(self):
        """Shutdown all container resources.

        This method should be called during application shutdown to ensure
        clean resource cleanup.
        """
        # Shutdown resource providers in reverse order
        if hasattr(self.services, 'shutdown_resources'):
            await self.services.shutdown_resources()
        if hasattr(self.external, 'shutdown_resources'):
            await self.external.shutdown_resources()


# Type hints for dependency injection
SettingsProvider = Provide["config.settings"]
LLMServiceProvider = Provide["services.llm_service"]
EmbeddingServiceProvider = Provide["services.embedding_service"]
WeaviateClientProvider = Provide["external.weaviate_client"]
DocumentRepositoryProvider = Provide["repositories.document_repository"]
