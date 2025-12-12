"""Repository dependencies backed by the DI container.

These helpers expose repository instances and health checks for FastAPI
endpoints and background tasks without constructing legacy containers.
"""

from typing import AsyncGenerator

from app.di import container
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.repositories.search_repository import SearchRepository


async def get_document_repository() -> DocumentRepository:
    """Return document repository from the DI container."""
    return container.repositories.document_repository()


async def get_data_source_repository() -> DataSourceRepository:
    """Return data source repository from the DI container."""
    return container.repositories.data_source_repository()


async def get_embedding_repository() -> EmbeddingRepository:
    """Return embedding repository from the DI container."""
    return container.repositories.weaviate_embedding_repository()


async def get_search_repository() -> SearchRepository:
    """Return search repository from the DI container."""
    return container.repositories.search_repository()


async def get_transaction_session() -> AsyncGenerator:
    """Yield a transactional session managed by the DI container."""
    async with container.repositories.transaction_session() as session:
        yield session


async def repositories_health() -> dict:
    """Return repository health status using DI providers."""
    return await container.repositories.repositories_health_check()
