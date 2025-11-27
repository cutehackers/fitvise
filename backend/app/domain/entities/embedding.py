"""Embedding entity for domain model (Epic 2.2).

This module defines the Embedding entity representing a vector embedding
with associated metadata, tracking, and relationships to source content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.domain.value_objects.embedding_vector import EmbeddingVector


@dataclass
class Embedding:
    """Embedding entity representing a vector embedding (Task 2.2.1).

    An embedding captures the semantic representation of text or other content
    as a dense vector, with metadata tracking its source, model, and creation.

    Attributes:
        id: Unique embedding identifier
        vector: The embedding vector (immutable value object)
        model_name: Name of the model that generated this embedding
        model_version: Version of the embedding model
        dimension: Vector dimensionality
        chunk_id: Optional reference to source chunk
        query_id: Optional reference to source query
        document_id: Optional reference to source document
        metadata: Additional metadata (source type, context, etc.)
        created_at: Timestamp when embedding was created
        updated_at: Timestamp when embedding was last updated

    Examples:
        >>> import numpy as np
        >>> vector = EmbeddingVector(np.random.rand(768))
        >>> embedding = Embedding(
        ...     vector=vector,
        ...     model_name="Alibaba-NLP/gte-multilingual-base",
        ...     chunk_id=uuid4()
        ... )
        >>> embedding.dimension
        768
        >>> embedding.is_normalized()
        True

        >>> # Create embedding for query
        >>> query_embedding = Embedding.for_query(
        ...     vector=vector,
        ...     query_id=uuid4(),
        ...     model_name="Alibaba-NLP/gte-multilingual-base"
        ... )
        >>> query_embedding.source_type
        'query'
    """

    id: UUID = field(default_factory=uuid4)
    vector: EmbeddingVector = field(default=None)  # type: ignore
    model_name: str = ""
    model_version: str = "1.0"
    dimension: int = 0
    chunk_id: Optional[UUID] = None
    query_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        # Create a copy of metadata to ensure independence
        if self.metadata is not None:
            self.metadata = dict(self.metadata)
        # Auto-sync dimension with vector length for consistency
        if self.vector is not None and self.dimension != len(self.vector):
            self.dimension = len(self.vector)
        self.validate()

    @classmethod
    def for_chunk(
        cls,
        vector: EmbeddingVector,
        chunk_id: UUID,
        model_name: str,
        document_id: Optional[UUID] = None,
        model_version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Embedding:
        """Create embedding for a document chunk.

        Args:
            vector: Embedding vector
            chunk_id: Source chunk identifier
            model_name: Name of embedding model
            document_id: Optional source document identifier
            model_version: Model version string
            metadata: Additional metadata

        Returns:
            Embedding instance for chunk
        """
        meta = metadata or {}
        meta["source_type"] = "chunk"

        return cls(
            vector=vector,
            model_name=model_name,
            model_version=model_version,
            chunk_id=chunk_id,
            document_id=document_id,
            metadata=meta,
        )

    @classmethod
    def for_query(
        cls,
        vector: EmbeddingVector,
        query_id: Optional[UUID] = None,
        model_name: str = "",
        model_version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Embedding:
        """Create embedding for a user query.

        Args:
            vector: Embedding vector
            query_id: Optional query identifier
            model_name: Name of embedding model
            model_version: Model version string
            metadata: Additional metadata

        Returns:
            Embedding instance for query
        """
        meta = metadata or {}
        meta["source_type"] = "query"

        return cls(
            vector=vector,
            model_name=model_name,
            model_version=model_version,
            query_id=query_id,
            metadata=meta,
        )

    @classmethod
    def for_query_text(
        cls,
        vector: EmbeddingVector,
        query_text: str,
        model_name: str,
        model_version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Embedding:
        """Create embedding for a user query with stored text for caching.

        Args:
            vector: Embedding vector
            query_text: Query text to store for caching
            model_name: Name of embedding model
            model_version: Model version string
            metadata: Additional metadata

        Returns:
            Embedding instance for query with cached text
        """
        meta = metadata or {}
        meta["source_type"] = "query_text"
        meta["query_text"] = query_text

        return cls(
            vector=vector,
            model_name=model_name,
            model_version=model_version,
            metadata=meta,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Embedding:
        """Create embedding from dictionary representation.

        Args:
            data: Dictionary with embedding data

        Returns:
            Embedding instance
        """
        # Reconstruct EmbeddingVector from array
        vector_data = data.get("vector")
        vector = (
            EmbeddingVector.from_list(vector_data)
            if vector_data is not None
            else None
        )

        # Parse UUIDs
        id_str = data.get("id")
        chunk_id_str = data.get("chunk_id")
        query_id_str = data.get("query_id")
        document_id_str = data.get("document_id")

        # Parse timestamps
        created_str = data.get("created_at")
        updated_str = data.get("updated_at")

        return cls(
            id=UUID(id_str) if id_str else uuid4(),
            vector=vector,  # type: ignore
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", "1.0"),
            dimension=data.get("dimension", 0),
            chunk_id=UUID(chunk_id_str) if chunk_id_str else None,
            query_id=UUID(query_id_str) if query_id_str else None,
            document_id=UUID(document_id_str) if document_id_str else None,
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(created_str)
                if created_str
                else datetime.utcnow()
            ),
            updated_at=(
                datetime.fromisoformat(updated_str)
                if updated_str
                else datetime.utcnow()
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert embedding to dictionary representation.

        Returns:
            Dictionary with all embedding data
        """
        return {
            "id": str(self.id),
            "vector": self.vector.to_list() if self.vector else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "chunk_id": str(self.chunk_id) if self.chunk_id else None,
            "query_id": str(self.query_id) if self.query_id else None,
            "document_id": str(self.document_id) if self.document_id else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def validate(self) -> None:
        """Validate embedding entity.

        Raises:
            ValueError: If embedding is invalid
        """
        if self.vector is None:
            raise ValueError("vector cannot be None")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if self.dimension != len(self.vector):
            raise ValueError(f"dimension mismatch: {self.dimension} != {len(self.vector)}")
        if self.dimension < 1:
            raise ValueError(f"dimension must be ≥1, got {self.dimension}")
        # Optional: embedding can reference chunk_id, query_id, or neither (for generic embeddings)

        # Validate vector itself
        self.vector.validate()

    def similarity_to(self, other: Embedding) -> float:
        """Calculate similarity to another embedding.

        Args:
            other: Other embedding to compare

        Returns:
            Similarity score (cosine similarity)

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if self.dimension != other.dimension:
            raise ValueError(
                f"dimension mismatch: {self.dimension} != {other.dimension}"
            )

        return self.vector.cosine_similarity(other.vector)

    def is_normalized(self) -> bool:
        """Check if embedding vector is normalized.

        Returns:
            True if vector is L2-normalized
        """
        return self.vector.is_normalized()

    def normalize(self) -> None:
        """Normalize the embedding vector in-place."""
        self.vector = self.vector.normalize()
        self.updated_at = datetime.utcnow()

    @property
    def source_type(self) -> str:
        """Get source type from metadata.

        Returns:
            Source type string ("chunk", "query", etc.)
        """
        return self.metadata.get("source_type", "unknown")

    @property
    def source_id(self) -> Optional[UUID]:
        """Get primary source identifier.

        Returns:
            Chunk ID if available, otherwise query ID
        """
        return self.chunk_id or self.query_id

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Embedding):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Calculate hash based on ID."""
        return hash(self.id)

    def __repr__(self) -> str:
        """String representation of embedding."""
        return (
            f"Embedding(id={self.id}, model={self.model_name}, "
            f"dim={self.dimension}, source={self.source_type})"
        )
