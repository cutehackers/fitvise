"""Repository Providers.

This module provides repository providers for the DI container.
It handles database repositories, vector store repositories, and other data access components.
"""

from dependency_injector import containers, providers
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.repositories.search_repository import SearchRepository
from app.infrastructure.persistence.repositories.async_document_repository import (
    AsyncDocumentRepository,
)
from app.infrastructure.persistence.repositories.async_data_source_repository import (
    AsyncDataSourceRepository,
)
from app.infrastructure.persistence.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)
from app.infrastructure.persistence.repositories.weaviate_search_repository import (
    WeaviateSearchRepository,
)


class RepositoryProviders(containers.DeclarativeContainer):
    """Repository providers for the FitVise application.

    Provides access to data repositories including SQL database repositories
    and vector store repositories with proper session management.
    """

    # Configuration dependencies
    config = providers.Dependency()
    external_services = providers.Dependency()

    # Database session factory
    session_factory = providers.Singleton(
        async_sessionmaker,
        config.settings.database_url,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def _session_resource(session_factory: async_sessionmaker) -> AsyncSession:
        async with session_factory() as session:
            yield session

    # Current database session (scoped to request)
    session = providers.Resource(
        _session_resource,
        session_factory=session_factory,
    )

    # Individual repository providers
    document_repository = providers.Factory(
        AsyncDocumentRepository,
        session=session,
    )

    data_source_repository = providers.Factory(
        AsyncDataSourceRepository,
        session=session,
    )

    weaviate_embedding_repository = providers.Factory(
        WeaviateEmbeddingRepository,
        weaviate_client=external_services.weaviate_client,
    )

    search_repository = providers.Factory(
        WeaviateSearchRepository,
        weaviate_client=external_services.weaviate_client,
        embedding_repository=weaviate_embedding_repository,
        embedding_service=external_services.sentence_transformer_service,
    )

    document_repo_interface = providers.Factory(lambda repo: repo, repo=document_repository)
    data_source_repo_interface = providers.Factory(lambda repo: repo, repo=data_source_repository)
    embedding_repo_interface = providers.Factory(lambda repo: repo, repo=weaviate_embedding_repository)
    search_repo_interface = providers.Factory(lambda repo: repo, repo=search_repository)

    def _repository_bundle(
        document_repo: DocumentRepository,
        data_source_repo: DataSourceRepository,
        embedding_repo: EmbeddingRepository,
    ) -> dict:
        return {
            "document_repository": document_repo,
            "data_source_repository": data_source_repo,
            "embedding_repository": embedding_repo,
        }

    repository_bundle = providers.Factory(
        _repository_bundle,
        document_repo=document_repo_interface,
        data_source_repo=data_source_repo_interface,
        embedding_repo=embedding_repo_interface,
    )

    # Database session management
    async def _transaction_session(session_factory: async_sessionmaker) -> AsyncSession:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    transaction_session = providers.Resource(_transaction_session, session_factory=session_factory)

    # Health check providers
    async def _database_health_check(session: AsyncSession) -> bool:
        try:
            await session.execute("SELECT 1")
            return True
        except Exception:
            return False

    database_health_check = providers.Factory(_database_health_check, session=session)

    async def _repositories_health_check(database_healthy: bool, weaviate_healthy: bool) -> dict:
        return {
            "overall": database_healthy and weaviate_healthy,
            "database": database_healthy,
            "weaviate": weaviate_healthy,
        }

    repositories_health_check = providers.Factory(
        _repositories_health_check,
        database_healthy=database_health_check,
        weaviate_healthy=external_services.weaviate_health_check,
    )

    # Repository initialization providers
    async def _init_weaviate_repository(
        repo: WeaviateEmbeddingRepository,
        weaviate_client,
    ) -> WeaviateEmbeddingRepository:
        await weaviate_client
        return repo

    init_weaviate_repository = providers.Resource(
        _init_weaviate_repository,
        repo=weaviate_embedding_repository,
        weaviate_client=external_services.init_weaviate_client,
    )

    # Environment-specific repository providers
    def _active_embedding_repository(
        weaviate_repo: WeaviateEmbeddingRepository,
        database_url: str,
    ) -> EmbeddingRepository:
        return weaviate_repo

    active_embedding_repository = providers.Factory(
        _active_embedding_repository,
        weaviate_repo=weaviate_embedding_repository,
        database_url=config.database_url,
    )
