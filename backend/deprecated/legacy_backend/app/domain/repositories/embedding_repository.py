"""Embedding repository interface (Epic 2.2).

This module defines the abstract repository interface for embedding
persistence and retrieval operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.entities.embedding import Embedding
from app.domain.value_objects.embedding_vector import EmbeddingVector


class EmbeddingRepository(ABC):
    """Abstract repository for embedding persistence (Task 2.2.1).

    Defines the contract for storing, retrieving, and searching embeddings
    in a vector database with support for batch operations and similarity search.

    Examples:
        >>> repository = WeaviateEmbeddingRepository()
        >>> await repository.save(embedding)
        >>> found = await repository.find_by_id(embedding.id)
        >>> found.id == embedding.id
        True

        >>> # Batch operations
        >>> await repository.batch_save(embeddings)

        >>> # Similarity search
        >>> results = await repository.similarity_search(
        ...     query_vector=query_embedding.vector,
        ...     k=10,
        ...     filters={"doc_type": "pdf"}
        ... )
    """

    @abstractmethod
    async def save(self, embedding: Embedding) -> None:
        """Save a single embedding.

        Args:
            embedding: Embedding to save

        Raises:
            EmbeddingStorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def batch_save(
        self,
        embeddings: Sequence[Embedding],
        batch_size: int = 100,
    ) -> int:
        """Save multiple embeddings in batches.

        Args:
            embeddings: Sequence of embeddings to save
            batch_size: Number of embeddings per batch

        Returns:
            Number of embeddings successfully saved

        Raises:
            EmbeddingStorageError: If batch save operation fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, embedding_id: UUID) -> Optional[Embedding]:
        """Find embedding by ID.

        Args:
            embedding_id: Embedding identifier

        Returns:
            Embedding if found, None otherwise

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def find_by_chunk_id(self, chunk_id: UUID) -> Optional[Embedding]:
        """Find embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Embedding if found, None otherwise

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def find_by_chunk_ids(
        self, chunk_ids: Sequence[UUID]
    ) -> List[Embedding]:
        """Find embeddings by multiple chunk IDs.

        Args:
            chunk_ids: Sequence of chunk identifiers

        Returns:
            List of embeddings found

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def find_by_document_id(self, document_id: UUID) -> List[Embedding]:
        """Find all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of embeddings for the document

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def find_by_query_text(self, query_text: str) -> Optional[Embedding]:
        """Find embedding by query text for caching.

        Args:
            query_text: Query text to search for

        Returns:
            Embedding if found, None otherwise

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: EmbeddingVector,
        k: int = 10,
        filters: Optional[Dict[str, any]] = None,
        min_similarity: float = 0.0,
    ) -> List[tuple[Embedding, float]]:
        """Search for similar embeddings using vector similarity.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of (embedding, similarity_score) tuples, sorted by similarity

        Raises:
            EmbeddingStorageError: If search operation fails
        """
        pass

    @abstractmethod
    async def exists(self, embedding_id: UUID) -> bool:
        """Check if embedding exists.

        Args:
            embedding_id: Embedding identifier

        Returns:
            True if embedding exists

        Raises:
            EmbeddingStorageError: If check operation fails
        """
        pass

    @abstractmethod
    async def delete(self, embedding_id: UUID) -> bool:
        """Delete embedding by ID.

        Args:
            embedding_id: Embedding identifier

        Returns:
            True if embedding was deleted, False if not found

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def delete_by_chunk_id(self, chunk_id: UUID) -> int:
        """Delete embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of embeddings deleted

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of embeddings deleted

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Count total number of embeddings.

        Returns:
            Total embedding count

        Raises:
            EmbeddingStorageError: If count operation fails
        """
        pass

    @abstractmethod
    async def count_by_model(self, model_name: str) -> int:
        """Count embeddings by model name.

        Args:
            model_name: Embedding model name

        Returns:
            Number of embeddings for the model

        Raises:
            EmbeddingStorageError: If count operation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check repository health and connectivity.

        Returns:
            True if repository is healthy and accessible

        Raises:
            EmbeddingStorageError: If health check fails
        """
        pass
