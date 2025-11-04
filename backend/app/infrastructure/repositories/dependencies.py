"""FastAPI dependency injection for repositories.

Provides dependency injection functions for use in FastAPI endpoints
and application services.
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.infrastructure.database.database import get_async_session
from app.infrastructure.repositories.factory import RepositoryFactory

# Global settings instance
_settings = Settings()


async def get_document_repository() -> AsyncGenerator[DocumentRepository, None]:
    """FastAPI dependency for document repository.

    Automatically selects repository type based on DATABASE_URL configuration.
    Uses database repository if async driver is configured, otherwise in-memory.

    Yields:
        DocumentRepository instance (async context manager)

    Example:
        ```python
        @app.get("/documents")
        async def list_documents(
            repo: DocumentRepository = Depends(get_document_repository)
        ):
            documents = await repo.list()
            return documents
        ```
    """
    repository_type = RepositoryFactory.get_repository_type_from_settings(_settings)

    if repository_type == "database":
        # Create database repository with async session
        async for session in get_async_session():
            yield RepositoryFactory.create_document_repository(
                repository_type="database",
                session=session,
            )
    else:
        # Create in-memory repository (no session needed)
        yield RepositoryFactory.create_document_repository(repository_type="memory")


def get_data_source_repository() -> DataSourceRepository:
    """FastAPI dependency for data source repository.

    Returns:
        DataSourceRepository instance (currently always in-memory)

    Example:
        ```python
        @app.get("/sources")
        def list_sources(
            repo: DataSourceRepository = Depends(get_data_source_repository)
        ):
            sources = repo.find_all()
            return sources
        ```
    """
    return RepositoryFactory.create_data_source_repository(repository_type="memory")


async def create_repository_bundle_for_pipeline(
    settings: Settings | None = None,
) -> tuple[DocumentRepository, DataSourceRepository]:
    """Create repository bundle for RAG pipeline execution.

    This function creates properly configured repository instances for use
    in the RAG pipeline workflow, handling session management automatically.

    Args:
        settings: Optional settings instance (uses global if not provided)

    Returns:
        Tuple of (document_repository, data_source_repository)

    Example:
        ```python
        doc_repo, src_repo = await create_repository_bundle_for_pipeline()
        workflow = RAGWorkflow(
            repositories=RepositoryBundle(
                document_repository=doc_repo,
                data_source_repository=src_repo,
            )
        )
        ```

    Note:
        For database repositories, the session is managed internally.
        The repository will work correctly within async contexts.
    """
    if settings is None:
        settings = _settings

    repository_type = RepositoryFactory.get_repository_type_from_settings(settings)

    # Data source repository (always in-memory for now)
    data_source_repo = RepositoryFactory.create_data_source_repository(
        repository_type="memory"
    )

    # Document repository based on configuration
    if repository_type == "database":
        # For pipeline usage, we need a repository with a long-lived session
        # The pipeline will manage the session lifecycle
        from app.infrastructure.database.database import async_session_maker

        session = async_session_maker()
        document_repo = RepositoryFactory.create_document_repository(
            repository_type="database",
            session=session,
        )
    else:
        document_repo = RepositoryFactory.create_document_repository(
            repository_type="memory"
        )

    return document_repo, data_source_repo
