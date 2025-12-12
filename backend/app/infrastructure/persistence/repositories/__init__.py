"""Persistence layer repository implementations."""

from app.infrastructure.persistence.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.persistence.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from app.infrastructure.persistence.repositories.dependencies import (
    get_document_repository,
    get_data_source_repository,
    get_embedding_repository,
    get_search_repository,
    get_transaction_session,
    repositories_health,
)

# Aliases
PostgresDocumentRepository = SQLAlchemyDocumentRepository
SQLiteDocumentRepository = SQLAlchemyDocumentRepository  # Same implementation, different name

__all__ = [
    # Repository implementations
    "InMemoryDocumentRepository",
    "SQLAlchemyDocumentRepository",
    "PostgresDocumentRepository",
    "SQLiteDocumentRepository",  # Alias for clarity
    # Dependency helpers (DI-first)
    "get_document_repository",
    "get_data_source_repository",
    "get_embedding_repository",
    "get_search_repository",
    "get_transaction_session",
    "repositories_health",
]
