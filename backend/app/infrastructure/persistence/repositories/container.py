"""Repository container for dependency injection.

Provides unified repository access for both FastAPI endpoints and standalone scripts.
The container manages repository lifecycle and configuration-based instantiation.
"""
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.infrastructure.persistence.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)
from app.infrastructure.persistence.repositories.in_memory_data_source_repository import (
    InMemoryDataSourceRepository,
)
from app.infrastructure.persistence.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.persistence.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)


class RepositoryContainer:
    """Container for repository instances.

    This container provides:
    - Unified repository access for FastAPI and scripts
    - Configuration-driven implementation selection
    - Lazy initialization of repositories
    - Session lifecycle management
    - External services integration for embedding repositories

    Usage in FastAPI:
        container = RepositoryContainer(settings, session)
        repo = container.document_repository

    Usage in scripts:
        settings = Settings()
        container = RepositoryContainer(settings)
        repo = container.document_repository  # Gets in-memory if no session
    """

    def __init__(
        self,
        settings: Settings,
        session: Optional[AsyncSession] = None,
        external_services: Optional["ExternalServicesContainer"] = None,
    ):
        """Initialize container with configuration.

        Args:
            settings: Application settings
            session: Optional database session for database repositories
            external_services: Optional external services container for embedding repositories
        """
        self.settings = settings
        self.session = session
        self.external_services = external_services
        self._use_database = self._should_use_database()

        # Lazy-initialized repositories
        self._document_repository: Optional[DocumentRepository] = None
        self._data_source_repository: Optional[DataSourceRepository] = None
        self._embedding_repository: Optional[EmbeddingRepository] = None

    def _should_use_database(self) -> bool:
        """Check if database repositories should be used based on settings.

        Returns:
            True if database URL is configured with async driver
        """
        db_url = self.settings.database_url.lower()
        async_drivers = ["aiosqlite", "asyncpg", "aiomysql", "asyncmy"]
        return any(driver in db_url for driver in async_drivers)

    @property
    def document_repository(self) -> DocumentRepository:
        """Get document repository instance.

        Returns repository based on configuration:
        - Database repository if session provided and DB configured
        - In-memory repository otherwise

        Returns:
            DocumentRepository implementation

        Raises:
            ValueError: If database mode but no session provided
        """
        if self._document_repository is None:
            if self._use_database:
                if self.session is None:
                    raise ValueError(
                        "Database session required for database repository. "
                        "Pass session to RepositoryContainer constructor."
                    )
                self._document_repository = SQLAlchemyDocumentRepository(self.session)
            else:
                self._document_repository = InMemoryDocumentRepository()

        return self._document_repository

    @property
    def data_source_repository(self) -> DataSourceRepository:
        """Get data source repository instance.

        Currently returns in-memory implementation.
        Database implementation will be added when available.

        Returns:
            DataSourceRepository implementation
        """
        if self._data_source_repository is None:
            # Future: Add database implementation
            # if self._use_database and self.session:
            #     self._data_source_repository = SQLAlchemyDataSourceRepository(
            #         self.session
            #     )
            self._data_source_repository = InMemoryDataSourceRepository()

        return self._data_source_repository

    @property
    def embedding_repository(self) -> EmbeddingRepository:
        """Get embedding repository instance using external services.

        Returns a WeaviateEmbeddingRepository using the external services container.
        Requires external_services to be provided during initialization.

        Returns:
            WeaviateEmbeddingRepository instance

        Raises:
            ValueError: If external_services is not provided
            ExternalServicesError: If repository initialization fails
        """
        if self._embedding_repository is None:
            if self.external_services is None:
                raise ValueError(
                    "ExternalServicesContainer required for embedding repository. "
                    "Pass external_services to RepositoryContainer constructor."
                )

            try:
                # Create WeaviateEmbeddingRepository using external services
                weaviate_client = self.external_services.weaviate_client
                self._embedding_repository = WeaviateEmbeddingRepository(weaviate_client)
            except Exception as exc:
                # Import here to avoid circular dependency
                from app.infrastructure.external_services.external_services_container import (
                    ExternalServicesError,
                )
                raise ExternalServicesError(
                    f"Failed to initialize embedding repository: {str(exc)}"
                ) from exc

        return self._embedding_repository
