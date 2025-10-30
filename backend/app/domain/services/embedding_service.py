"""Embedding domain service (Epic 2.2).

This module defines the domain service coordinating embedding generation,
storage, and retrieval operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.entities.embedding import Embedding
from app.domain.entities.chunk import Chunk
from app.domain.exceptions.embedding_exceptions import (
    DimensionMismatchError,
    EmbeddingGenerationError,
    EmbeddingStorageError,
)
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.value_objects.embedding_vector import EmbeddingVector


class EmbeddingService:
    """Domain service for embedding operations (Task 2.2.1).

    Coordinates between embedding generation, validation, and storage,
    implementing business logic for batch vs real-time processing.

    Examples:
        >>> service = EmbeddingService(repository)
        >>> # Store embedding
        >>> await service.store_chunk_embedding(
        ...     chunk=chunk,
        ...     vector=vector,
        ...     model_name="Alibaba-NLP/gte-multilingual-base"
        ... )

        >>> # Batch storage
        >>> await service.store_chunk_embeddings_batch(
        ...     chunks=chunks,
        ...     vectors=vectors,
        ...     model_name="Alibaba-NLP/gte-multilingual-base"
        ... )

        >>> # Search similar
        >>> results = await service.find_similar(
        ...     query_vector=query_vector,
        ...     k=10
        ... )
    """

    def __init__(self, repository: EmbeddingRepository) -> None:
        """Initialize embedding service.

        Args:
            repository: Embedding repository for persistence
        """
        self._repository = repository

    async def store_chunk_embedding(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
        model_name: str,
        model_version: str = "1.0",
        metadata: Optional[Dict[str, any]] = None,
    ) -> Embedding:
        """Store embedding for a chunk.

        Args:
            chunk: Source chunk
            vector: Embedding vector
            model_name: Name of embedding model
            model_version: Model version
            metadata: Additional metadata

        Returns:
            Stored embedding

        Raises:
            EmbeddingStorageError: If storage fails
        """
        embedding = Embedding.for_chunk(
            vector=vector,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            model_name=model_name,
            model_version=model_version,
            metadata=metadata,
        )

        try:
            await self._repository.save(embedding)
            return embedding
        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to store chunk embedding",
                operation="save",
                embedding_id=embedding.id,
                details=str(e),
            ) from e

    async def store_chunk_embeddings_batch(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[EmbeddingVector],
        model_name: str,
        model_version: str = "1.0",
        batch_size: int = 100,
    ) -> List[Embedding]:
        """Store embeddings for multiple chunks in batches.

        Args:
            chunks: Source chunks
            vectors: Embedding vectors (same order as chunks)
            model_name: Name of embedding model
            model_version: Model version
            batch_size: Number of embeddings per batch

        Returns:
            List of stored embeddings

        Raises:
            ValueError: If chunks and vectors have different lengths
            EmbeddingStorageError: If batch storage fails
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks != {len(vectors)} vectors"
            )

        embeddings = [
            Embedding.for_chunk(
                vector=vector,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                model_name=model_name,
                model_version=model_version,
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        try:
            await self._repository.batch_save(embeddings, batch_size=batch_size)
            return embeddings
        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to batch store embeddings",
                operation="batch_save",
                details=str(e),
            ) from e

    async def find_by_chunk(self, chunk_id: UUID) -> Optional[Embedding]:
        """Find embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Embedding if found, None otherwise
        """
        return await self._repository.find_by_chunk_id(chunk_id)

    async def find_by_document(self, document_id: UUID) -> List[Embedding]:
        """Find all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of embeddings for the document
        """
        return await self._repository.find_by_document_id(document_id)

    async def find_similar(
        self,
        query_vector: EmbeddingVector,
        k: int = 10,
        filters: Optional[Dict[str, any]] = None,
        min_similarity: float = 0.0,
    ) -> List[tuple[Embedding, float]]:
        """Find similar embeddings using vector search.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of (embedding, similarity_score) tuples
        """
        return await self._repository.similarity_search(
            query_vector=query_vector,
            k=k,
            filters=filters,
            min_similarity=min_similarity,
        )

    async def validate_dimension_compatibility(
        self,
        embedding: Embedding,
        expected_dimension: int,
    ) -> None:
        """Validate embedding dimension matches expected.

        Args:
            embedding: Embedding to validate
            expected_dimension: Expected vector dimension

        Raises:
            DimensionMismatchError: If dimensions don't match
        """
        if embedding.dimension != expected_dimension:
            raise DimensionMismatchError(
                expected=expected_dimension,
                actual=embedding.dimension,
                details=f"Embedding {embedding.id} has incompatible dimension",
            )

    async def delete_by_chunk(self, chunk_id: UUID) -> int:
        """Delete embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of embeddings deleted
        """
        return await self._repository.delete_by_chunk_id(chunk_id)

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of embeddings deleted
        """
        return await self._repository.delete_by_document_id(document_id)

    async def health_check(self) -> bool:
        """Check embedding service health.

        Returns:
            True if service is healthy
        """
        return await self._repository.health_check()
