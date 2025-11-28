"""Tests for RepositoryContainer.

Tests the repository container's ability to provide repositories
based on configuration and manage their lifecycle.
"""
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings
from app.infrastructure.persistence.repositories.container import RepositoryContainer
from app.infrastructure.persistence.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.persistence.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from app.infrastructure.persistence.repositories.in_memory_data_source_repository import (
    InMemoryDataSourceRepository,
)


@pytest.fixture
def memory_settings():
    """Settings configured for in-memory repositories."""
    settings = Mock(spec=Settings)
    settings.database_url = "sqlite:///:memory:"
    return settings


@pytest.fixture
def database_settings():
    """Settings configured for database repositories."""
    settings = Mock(spec=Settings)
    settings.database_url = "postgresql+asyncpg://localhost/test"
    return settings


@pytest.fixture
def async_session():
    """Mock async database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


class TestRepositoryContainerInitialization:
    """Test container initialization and configuration detection."""

    def test_container_initializes_with_settings(self, memory_settings):
        """Test container initializes with settings."""
        container = RepositoryContainer(memory_settings)

        assert container.settings == memory_settings
        assert container.session is None
        assert container._use_database is False

    def test_container_initializes_with_session(self, database_settings, async_session):
        """Test container initializes with session."""
        container = RepositoryContainer(database_settings, async_session)

        assert container.settings == database_settings
        assert container.session == async_session
        assert container._use_database is True

    def test_container_detects_memory_mode(self, memory_settings):
        """Test container correctly detects in-memory mode."""
        container = RepositoryContainer(memory_settings)

        assert container._use_database is False

    def test_container_detects_database_mode_asyncpg(self, database_settings):
        """Test container detects database mode with asyncpg driver."""
        container = RepositoryContainer(database_settings)

        assert container._use_database is True

    def test_container_detects_database_mode_aiosqlite(self):
        """Test container detects database mode with aiosqlite driver."""
        settings = Mock(spec=Settings)
        settings.database_url = "sqlite+aiosqlite:///test.db"

        container = RepositoryContainer(settings)

        assert container._use_database is True

    def test_container_detects_database_mode_aiomysql(self):
        """Test container detects database mode with aiomysql driver."""
        settings = Mock(spec=Settings)
        settings.database_url = "mysql+aiomysql://localhost/test"

        container = RepositoryContainer(settings)

        assert container._use_database is True


class TestDocumentRepositoryProperty:
    """Test document repository property and lazy initialization."""

    def test_lazy_initialization(self, memory_settings):
        """Test repositories are lazily initialized."""
        container = RepositoryContainer(memory_settings)

        assert container._document_repository is None

        # First access initializes
        repo = container.document_repository
        assert container._document_repository is not None

        # Second access returns same instance
        repo2 = container.document_repository
        assert repo is repo2

    def test_memory_repository_creation(self, memory_settings):
        """Test container creates in-memory document repository."""
        container = RepositoryContainer(memory_settings)

        repo = container.document_repository

        assert isinstance(repo, InMemoryDocumentRepository)

    def test_database_repository_creation(self, database_settings, async_session):
        """Test container creates database document repository."""
        container = RepositoryContainer(database_settings, async_session)

        repo = container.document_repository

        assert isinstance(repo, SQLAlchemyDocumentRepository)
        assert repo._session == async_session

    def test_database_mode_requires_session(self, database_settings):
        """Test database mode raises error without session."""
        container = RepositoryContainer(database_settings)

        with pytest.raises(ValueError, match="Database session required"):
            _ = container.document_repository

    def test_memory_mode_no_session_required(self, memory_settings):
        """Test memory mode works without session."""
        container = RepositoryContainer(memory_settings)  # No session!

        repo = container.document_repository

        assert isinstance(repo, InMemoryDocumentRepository)


class TestDataSourceRepositoryProperty:
    """Test data source repository property."""

    def test_lazy_initialization(self, memory_settings):
        """Test data source repository is lazily initialized."""
        container = RepositoryContainer(memory_settings)

        assert container._data_source_repository is None

        # First access initializes
        repo = container.data_source_repository
        assert container._data_source_repository is not None

        # Second access returns same instance
        repo2 = container.data_source_repository
        assert repo is repo2

    def test_always_returns_in_memory_repository(self, memory_settings):
        """Test data source repository always returns in-memory implementation."""
        container = RepositoryContainer(memory_settings)

        repo = container.data_source_repository

        assert isinstance(repo, InMemoryDataSourceRepository)

    def test_database_mode_not_yet_implemented(self, database_settings, async_session):
        """Test database mode for data source repo is not yet implemented."""
        container = RepositoryContainer(database_settings, async_session)

        # Currently returns in-memory even in database mode
        repo = container.data_source_repository

        assert isinstance(repo, InMemoryDataSourceRepository)


class TestContainerUsagePatterns:
    """Test common container usage patterns."""

    def test_container_for_fastapi_endpoint(self, database_settings, async_session):
        """Test container usage in FastAPI endpoint pattern."""
        # Simulates FastAPI dependency injection
        container = RepositoryContainer(database_settings, async_session)

        # Endpoint accesses multiple repositories
        doc_repo = container.document_repository
        src_repo = container.data_source_repository

        assert doc_repo is not None
        assert src_repo is not None

    def test_container_for_script(self, memory_settings):
        """Test container usage in standalone script pattern."""
        # Scripts can use container without session for in-memory mode
        container = RepositoryContainer(memory_settings)

        doc_repo = container.document_repository
        src_repo = container.data_source_repository

        assert isinstance(doc_repo, InMemoryDocumentRepository)
        assert isinstance(src_repo, InMemoryDataSourceRepository)

    def test_container_for_pipeline(self, database_settings, async_session):
        """Test container usage in pipeline pattern."""
        # Pipeline creates container once and passes to all phases
        container = RepositoryContainer(database_settings, async_session)

        # Multiple phases access repositories
        phase1_repo = container.document_repository
        phase2_repo = container.document_repository

        # Same instance shared across phases
        assert phase1_repo is phase2_repo

    def test_container_repository_caching(self, database_settings, async_session):
        """Test container caches repository instances."""
        container = RepositoryContainer(database_settings, async_session)

        # Access repository multiple times
        repo1 = container.document_repository
        repo2 = container.document_repository
        repo3 = container.document_repository

        # All should be the same instance
        assert repo1 is repo2
        assert repo2 is repo3


class TestContainerWithDifferentDrivers:
    """Test container behavior with different database drivers."""

    @pytest.mark.parametrize(
        "database_url,should_use_database",
        [
            ("sqlite+aiosqlite:///test.db", True),
            ("postgresql+asyncpg://localhost/test", True),
            ("mysql+aiomysql://localhost/test", True),
            ("mysql+asyncmy://localhost/test", True),
            ("sqlite:///:memory:", False),
            ("postgresql://localhost/test", False),  # No async driver
            ("mysql://localhost/test", False),  # No async driver
        ],
    )
    def test_driver_detection(self, database_url, should_use_database):
        """Test container correctly detects async drivers."""
        settings = Mock(spec=Settings)
        settings.database_url = database_url

        container = RepositoryContainer(settings)

        assert container._use_database == should_use_database
