"""Repository factory for dependency injection.

Provides factory functions to create repository instances based on
configuration, enabling easy switching between in-memory and persistent storage.
"""
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.infrastructure.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.repositories.in_memory_data_source_repository import (
    InMemoryDataSourceRepository,
)
from app.infrastructure.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)


RepositoryType = Literal["memory", "database"]


class RepositoryFactory:
    """Factory for creating repository instances based on configuration."""

    @staticmethod
    def create_document_repository(
        repository_type: RepositoryType = "memory",
        session: AsyncSession | None = None,
    ) -> DocumentRepository:
        """Create a document repository instance.

        Args:
            repository_type: Type of repository to create ('memory' or 'database')
            session: AsyncSession for database repositories (required if type='database')

        Returns:
            DocumentRepository instance

        Raises:
            ValueError: If database type selected but no session provided
        """
        if repository_type == "memory":
            return InMemoryDocumentRepository()
        elif repository_type == "database":
            if session is None:
                raise ValueError(
                    "AsyncSession is required for database repository type"
                )
            return SQLAlchemyDocumentRepository(session)
        else:
            raise ValueError(
                f"Unknown repository type: {repository_type}. "
                f"Must be 'memory' or 'database'"
            )

    @staticmethod
    def create_data_source_repository(
        repository_type: RepositoryType = "memory",
    ) -> DataSourceRepository:
        """Create a data source repository instance.

        Args:
            repository_type: Type of repository to create ('memory' only for now)

        Returns:
            DataSourceRepository instance

        Note:
            Currently only in-memory implementation is available.
            Database implementation coming soon.
        """
        if repository_type == "memory":
            return InMemoryDataSourceRepository()
        else:
            raise ValueError(
                f"Unknown repository type: {repository_type}. "
                f"Only 'memory' is currently supported for DataSourceRepository"
            )

    @staticmethod
    def get_repository_type_from_settings(settings: Settings) -> RepositoryType:
        """Determine repository type based on database URL configuration.

        Args:
            settings: Application settings

        Returns:
            'database' if DATABASE_URL is configured for PostgreSQL/SQLite with async driver,
            'memory' otherwise
        """
        db_url = settings.database_url.lower()

        # Check for async-compatible database URLs
        if any(
            driver in db_url
            for driver in ["aiosqlite", "asyncpg", "aiomysql", "asyncmy"]
        ):
            return "database"

        return "memory"
