"""FastAPI dependency injection for repositories.

Provides dependency injection functions for use in FastAPI endpoints
and application services.
"""
from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.database.database import get_async_session
from app.infrastructure.persistence.repositories.container import RepositoryContainer


async def get_repository_container(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[RepositoryContainer, None]:
    """Get repository container for request lifecycle.

    Container is created per-request with session attached.
    All repositories share the same session within the request.

    Args:
        settings: Application settings (auto-injected)
        session: Database session (auto-injected if database mode)

    Yields:
        RepositoryContainer instance

    Example:
        ```python
        @app.get("/documents")
        async def list_documents(
            container: RepositoryContainer = Depends(get_repository_container)
        ):
            documents = await container.document_repository.find_all()
            return documents
        ```
    """
    # Get session for database repositories
    async for db_session in get_async_session():
        container = RepositoryContainer(settings, db_session)
        try:
            yield container
        finally:
            # Cleanup if needed
            pass


async def get_document_repository(
    container: RepositoryContainer = Depends(get_repository_container),
) -> DocumentRepository:
    """FastAPI dependency for document repository.

    Automatically selects repository type based on DATABASE_URL configuration.
    Uses database repository if async driver is configured, otherwise in-memory.

    Args:
        container: Repository container (auto-injected)

    Returns:
        DocumentRepository instance

    Example:
        ```python
        @app.get("/documents")
        async def list_documents(
            repo: DocumentRepository = Depends(get_document_repository)
        ):
            documents = await repo.find_all()
            return documents
        ```
    """
    return container.document_repository


def get_data_source_repository(
    container: RepositoryContainer = Depends(get_repository_container),
) -> DataSourceRepository:
    """FastAPI dependency for data source repository.

    Args:
        container: Repository container (auto-injected)

    Returns:
        DataSourceRepository instance (currently always in-memory)

    Example:
        ```python
        @app.get("/sources")
        async def list_sources(
            repo: DataSourceRepository = Depends(get_data_source_repository)
        ):
            sources = await repo.find_all()
            return sources
        ```
    """
    return container.data_source_repository


async def create_repository_container_for_pipeline(
    settings: Settings | None = None,
) -> RepositoryContainer:
    """Create repository container for RAG pipeline execution.

    This function creates a properly configured repository container for use
    in the RAG pipeline workflow, handling session management automatically.

    Args:
        settings: Optional settings instance (uses default if not provided)

    Returns:
        RepositoryContainer with all repositories ready to use

    Example:
        ```python
        container = await create_repository_container_for_pipeline()

        # Access repositories
        documents = await container.document_repository.find_all()
        sources = await container.data_source_repository.find_all()

        # Use in pipeline tasks
        task = RagIngestionTask(
            document_repository=container.document_repository,
            data_source_repository=container.data_source_repository,
        )
        await task.execute(spec)
        ```

    Note:
        For database repositories, the session is managed internally.
        The container will work correctly within async contexts.
    """
    if settings is None:
        settings = get_settings()

    # Create session for database mode
    async for session in get_async_session():
        return RepositoryContainer(settings, session)

    # Fallback for in-memory mode (no session needed)
    return RepositoryContainer(settings)


async def get_repository_container_with_external_services(
    settings: Settings = Depends(get_settings),
) -> tuple[RepositoryContainer, "ExternalServicesContainer"]:
    """Get both RepositoryContainer and ExternalServicesContainer with proper relationship.

    Creates RepositoryContainer first, then ExternalServicesContainer, and links them
    together to eliminate circular dependencies.

    Args:
        settings: Application settings (auto-injected)

    Yields:
        Tuple of (RepositoryContainer, ExternalServicesContainer) with proper relationship

    Example:
        ```python
        @app.get("/documents")
        async def list_documents(
            containers: tuple[RepositoryContainer, ExternalServicesContainer] = Depends(get_repository_container_with_external_services)
        ):
            repo_container, external_container = containers
            documents = await repo_container.document_repository.find_all()
            embeddings = await repo_container.embedding_repository.find_all()
            return {"documents": documents, "embeddings": embeddings}
        ```
    """
    # Import here to avoid circular dependency
    from app.infrastructure.external_services.external_services_container import (
        ExternalServicesContainer,
    )

    # Get session for database repositories
    async for db_session in get_async_session():
        # Create repository container first
        repo_container = RepositoryContainer(settings, db_session)

        # Create external services container
        external_container = ExternalServicesContainer(settings)

        # Link them: RepositoryContainer needs ExternalServicesContainer for embedding repository
        repo_container.external_services = external_container

        yield repo_container, external_container
        return

    # Fallback for in-memory mode (no session needed)
    repo_container = RepositoryContainer(settings)
    external_container = ExternalServicesContainer(settings)
    repo_container.external_services = external_container
    yield repo_container, external_container


# Backward compatibility function - deprecated
async def create_repository_bundle_for_pipeline(
    settings: Settings | None = None,
) -> tuple[DocumentRepository, DataSourceRepository]:
    """DEPRECATED: Use create_repository_container_for_pipeline() instead.

    Create repository bundle for RAG pipeline execution.

    Args:
        settings: Optional settings instance

    Returns:
        Tuple of (document_repository, data_source_repository)

    Note:
        This function is deprecated and will be removed in a future version.
        Use create_repository_container_for_pipeline() instead, which returns
        a RepositoryContainer that provides cleaner access to repositories.
    """
    import warnings

    warnings.warn(
        "create_repository_bundle_for_pipeline() is deprecated. "
        "Use create_repository_container_for_pipeline() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    container = await create_repository_container_for_pipeline(settings)
    return container.document_repository, container.data_source_repository




