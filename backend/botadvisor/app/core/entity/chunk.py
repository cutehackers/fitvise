"""
Chunk Entity

Text chunk with metadata for retrieval and citation.
"""

from dataclasses import dataclass
from typing import Any

from .document_metadata import DocumentMetadata

@dataclass(frozen=True)
class Chunk:
    """
    Text chunk with metadata for retrieval and citation.

    Represents a unit of text returned by the retriever with
    sufficient metadata for citation and context.
    """

    chunk_id: str
    content: str
    metadata: DocumentMetadata
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
        }
