"""Persistence layer repository implementations."""

from app.infrastructure.persistence.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.persistence.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from app.di.containers.container import AppContainer

# Backward-compatible aliases
PostgresDocumentRepository = SQLAlchemyDocumentRepository  # Works with both PostgreSQL and SQLite
SQLiteDocumentRepository = SQLAlchemyDocumentRepository  # Same implementation, different name

__all__ = [
    # Repository implementations
    "InMemoryDocumentRepository",
    "SQLAlchemyDocumentRepository",
    "PostgresDocumentRepository",  # Alias for backward compatibility
    "SQLiteDocumentRepository",  # Alias for clarity
]
