"""Infrastructure repository implementations."""

from app.infrastructure.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from app.infrastructure.repositories.factory import RepositoryFactory, RepositoryType
from app.infrastructure.repositories.dependencies import (
    get_document_repository,
    get_data_source_repository,
    create_repository_bundle_for_pipeline,
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
    # Factory and dependencies
    "RepositoryFactory",
    "RepositoryType",
    "get_document_repository",
    "get_data_source_repository",
    "create_repository_bundle_for_pipeline",
]
