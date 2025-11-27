"""Embedding result value object.

This module defines the EmbeddingResult value object that encapsulates
the result of embedding generation with metadata and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from app.domain.value_objects.embedding_vector import EmbeddingVector


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of embedding generation operation.

    Encapsulates the generated embedding vector along with metadata
    about the generation process, performance metrics, and caching information.

    Attributes:
        id: Unique identifier for this embedding result
        query: Original text that was embedded
        vector: Generated embedding vector
        model_name: Name of the embedding model used
        model_version: Version of the embedding model
        processing_time_ms: Time taken to generate embedding in milliseconds
        cache_hit: Whether the result was retrieved from cache
        stored: Whether the embedding was stored persistently
        embedding_id: ID of stored embedding (if stored)
        vector_dimension: Dimension of the embedding vector
        metadata: Additional metadata about the embedding
    """

    id: UUID
    query: str
    vector: EmbeddingVector
    model_name: str
    model_version: str
    processing_time_ms: float
    cache_hit: bool
    stored: bool
    embedding_id: Optional[UUID]
    vector_dimension: int
    metadata: Dict[str, Any]

    @classmethod
    def create(
        cls,
        query: str,
        vector: EmbeddingVector,
        model_name: str,
        model_version: str = "1.0",
        processing_time_ms: float = 0.0,
        cache_hit: bool = False,
        stored: bool = False,
        embedding_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Create a new embedding result.

        Args:
            query: Original text that was embedded
            vector: Generated embedding vector
            model_name: Name of the embedding model used
            model_version: Version of the embedding model
            processing_time_ms: Time taken to generate embedding
            cache_hit: Whether result was retrieved from cache
            stored: Whether embedding was stored persistently
            embedding_id: ID of stored embedding
            metadata: Additional metadata

        Returns:
            New EmbeddingResult instance
        """
        return cls(
            id=uuid4(),
            query=query,
            vector=vector,
            model_name=model_name,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
            cache_hit=cache_hit,
            stored=stored,
            embedding_id=embedding_id,
            vector_dimension=vector.dimension,
            metadata=metadata or {},
        )

    @classmethod
    def from_cache_hit(
        cls,
        query: str,
        vector: EmbeddingVector,
        model_name: str,
        embedding_id: UUID,
        model_version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Create embedding result for cache hit.

        Args:
            query: Original text that was embedded
            vector: Retrieved embedding vector from cache
            model_name: Name of the embedding model used
            embedding_id: ID of stored embedding
            model_version: Version of the embedding model
            metadata: Additional metadata

        Returns:
            EmbeddingResult marked as cache hit
        """
        return cls(
            id=uuid4(),
            query=query,
            vector=vector,
            model_name=model_name,
            model_version=model_version,
            processing_time_ms=0.0,  # Cache hits are effectively instant
            cache_hit=True,
            stored=True,  # Cached embeddings are stored
            embedding_id=embedding_id,
            vector_dimension=vector.dimension,
            metadata=metadata or {},
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of embedding result
        """
        return {
            "id": str(self.id),
            "query": self.query,
            "vector": self.vector.as_dict(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
            "cache_hit": self.cache_hit,
            "stored": self.stored,
            "embedding_id": str(self.embedding_id) if self.embedding_id else None,
            "vector_dimension": self.vector_dimension,
            "metadata": self.metadata,
        }

    def with_embedding_id(self, embedding_id: UUID) -> EmbeddingResult:
        """Create new result with updated embedding ID.

        Args:
            embedding_id: New embedding ID

        Returns:
            New EmbeddingResult with updated ID
        """
        return EmbeddingResult(
            id=self.id,
            query=self.query,
            vector=self.vector,
            model_name=self.model_name,
            model_version=self.model_version,
            processing_time_ms=self.processing_time_ms,
            cache_hit=self.cache_hit,
            stored=True,
            embedding_id=embedding_id,
            vector_dimension=self.vector_dimension,
            metadata=self.metadata,
        )