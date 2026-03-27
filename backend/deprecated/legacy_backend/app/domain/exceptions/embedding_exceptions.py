"""Embedding-related exceptions (Epic 2.2).

This module defines exceptions for embedding generation, storage,
and retrieval operations.
"""

from typing import Optional
from uuid import UUID


class EmbeddingException(Exception):
    """Base exception for embedding operations."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        """Initialize embedding exception.

        Args:
            message: Exception message
            details: Additional error details
        """
        self.message = message
        self.details = details
        super().__init__(message)


class EmbeddingGenerationError(EmbeddingException):
    """Exception raised when embedding generation fails."""

    def __init__(
        self,
        message: str = "Embedding generation failed",
        model_name: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize embedding generation error.

        Args:
            message: Error message
            model_name: Name of the embedding model
            details: Additional error details
        """
        self.model_name = model_name
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.model_name:
            parts.append(f"model={self.model_name}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)


class EmbeddingStorageError(EmbeddingException):
    """Exception raised when embedding storage operations fail."""

    def __init__(
        self,
        message: str = "Embedding storage operation failed",
        operation: Optional[str] = None,
        embedding_id: Optional[UUID] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize embedding storage error.

        Args:
            message: Error message
            operation: Storage operation that failed
            embedding_id: ID of embedding involved
            details: Additional error details
        """
        self.operation = operation
        self.embedding_id = embedding_id
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.embedding_id:
            parts.append(f"embedding_id={self.embedding_id}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)


class ModelLoadError(EmbeddingException):
    """Exception raised when embedding model loading fails."""

    def __init__(
        self,
        message: str = "Failed to load embedding model",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize model load error.

        Args:
            message: Error message
            model_name: Name of the model
            model_path: Path to model files
            details: Additional error details
        """
        self.model_name = model_name
        self.model_path = model_path
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.model_name:
            parts.append(f"model={self.model_name}")
        if self.model_path:
            parts.append(f"path={self.model_path}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)


class DimensionMismatchError(EmbeddingException):
    """Exception raised when embedding dimensions don't match."""

    def __init__(
        self,
        message: str = "Embedding dimension mismatch",
        expected: Optional[int] = None,
        actual: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize dimension mismatch error.

        Args:
            message: Error message
            expected: Expected dimension
            actual: Actual dimension
            details: Additional error details
        """
        self.expected = expected
        self.actual = actual
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.expected is not None and self.actual is not None:
            parts.append(f"expected={self.expected}, got={self.actual}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)


class VectorSearchError(EmbeddingException):
    """Exception raised when vector similarity search fails."""

    def __init__(
        self,
        message: str = "Vector search operation failed",
        query_dimension: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize vector search error.

        Args:
            message: Error message
            query_dimension: Dimension of query vector
            details: Additional error details
        """
        self.query_dimension = query_dimension
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.query_dimension:
            parts.append(f"query_dim={self.query_dimension}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)


class IngestionPipelineError(EmbeddingException):
    """Exception raised when ingestion pipeline operations fail."""

    def __init__(
        self,
        message: str = "Ingestion pipeline operation failed",
        stage: Optional[str] = None,
        document_count: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize ingestion pipeline error.

        Args:
            message: Error message
            stage: Pipeline stage where error occurred (chunking, deduplication, embedding, storage)
            document_count: Number of documents being processed
            details: Additional error details
        """
        self.stage = stage
        self.document_count = document_count
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.stage:
            parts.append(f"stage={self.stage}")
        if self.document_count is not None:
            parts.append(f"documents={self.document_count}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)


class DeduplicationError(EmbeddingException):
    """Exception raised during content deduplication operations."""

    def __init__(
        self,
        message: str = "Content deduplication failed",
        chunks_processed: Optional[int] = None,
        duplicates_found: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize deduplication error.

        Args:
            message: Error message
            chunks_processed: Number of chunks processed before failure
            duplicates_found: Number of duplicates found before failure
            details: Additional error details
        """
        self.chunks_processed = chunks_processed
        self.duplicates_found = duplicates_found
        super().__init__(message, details)

    def __str__(self) -> str:
        """String representation of error."""
        parts = [self.message]
        if self.chunks_processed is not None:
            parts.append(f"chunks_processed={self.chunks_processed}")
        if self.duplicates_found is not None:
            parts.append(f"duplicates_found={self.duplicates_found}")
        if self.details:
            parts.append(f"details: {self.details}")
        return ", ".join(parts)
