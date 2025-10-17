"""Chunk entity representing semantic slices of a document (Task 2.1.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import UUID

from app.domain.value_objects.chunk_metadata import ChunkMetadata


@dataclass
class Chunk:
    """Domain entity for a single semantic chunk."""

    chunk_id: str
    document_id: UUID
    text: str
    metadata: ChunkMetadata
    embedding_vector_id: Optional[str] = None
    score: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("chunk text cannot be empty")

    @property
    def length(self) -> int:
        """Return number of characters."""
        return len(self.text)

    def with_score(self, score: float) -> "Chunk":
        """Return a copy with an updated similarity score."""
        return Chunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            text=self.text,
            metadata=self.metadata,
            embedding_vector_id=self.embedding_vector_id,
            score=score,
            attributes=self.attributes.copy(),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serialize chunk for persistence or transport."""
        payload = {
            "id": self.chunk_id,
            "document_id": str(self.document_id),
            "text": self.text,
            "metadata": self.metadata.as_dict(),
        }
        if self.embedding_vector_id:
            payload["embedding_vector_id"] = self.embedding_vector_id
        if self.score is not None:
            payload["score"] = self.score
        if self.attributes:
            payload["attributes"] = self.attributes
        payload["length"] = self.length
        return payload
