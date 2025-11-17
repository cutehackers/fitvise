"""Persistence layer repository implementations."""

from app.infrastructure.persistence.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.persistence.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from app.infrastructure.persistence.repositories.container import RepositoryContainer
from app.infrastructure.persistence.repositories.dependencies import (
    get_repository_container,
    get_document_repository,
    get_data_source_repository,
    create_repository_container_for_pipeline,
    create_repository_bundle_for_pipeline,  # Deprecated but kept for backward compatibility
)

# Backward-compatible aliases
PostgresDocumentRepository = SQLAlchemyDocumentRepository  # Works with both PostgreSQL and SQLite
SQLiteDocumentRepository = SQLAlchemyDocumentRepository  # Same implementation, different name

__all__ = [
    # Repository implementations
    "InMemoryDocumentRepository",
    "SQLAlchemyDocumentRepository",
    "PostgresDocumentRepository",  # Alias for backward compatibility
    "SQLiteDocumentRepository",  # Alias for clarity
    # Container and dependencies (NEW - Recommended)
    "RepositoryContainer",
    "get_repository_container",
    "get_document_repository",
    "get_data_source_repository",
    "create_repository_container_for_pipeline",
    # Deprecated but kept for backward compatibility
    "create_repository_bundle_for_pipeline",
]
