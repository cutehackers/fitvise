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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary representation."""
        # Handle field name mapping from serialized data
        chunk_data = data.copy()

        # Map "id" to "chunk_id" if needed (as used in as_dict serialization)
        if "id" in chunk_data and "chunk_id" not in chunk_data:
            chunk_data["chunk_id"] = chunk_data.pop("id")

        # Convert document_id back to UUID if it's a string
        if isinstance(chunk_data.get("document_id"), str):
            chunk_data["document_id"] = UUID(chunk_data["document_id"])

        # Reconstruct ChunkMetadata from its dictionary representation
        metadata_dict = chunk_data.pop("metadata", {})
        metadata = ChunkMetadata.from_dict(metadata_dict) if metadata_dict else ChunkMetadata.from_dict({
            "sequence": 0,
            "start": 0,
            "end": len(chunk_data.get("text", ""))
        })

        # Handle default values and optional fields
        chunk_id = chunk_data.get("chunk_id")
        if not chunk_id:
            chunk_id = str(UUID(bytes=bytes(16)))  # Generate a random UUID

        return cls(
            chunk_id=chunk_id,
            document_id=chunk_data["document_id"],
            text=chunk_data["text"],
            metadata=metadata,
            embedding_vector_id=chunk_data.get("embedding_vector_id"),
            score=chunk_data.get("score"),
            attributes=chunk_data.get("attributes", {})
        )
