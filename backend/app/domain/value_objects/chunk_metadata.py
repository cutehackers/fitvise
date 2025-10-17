"""Chunk metadata value object for semantic retrieval (Task 2.1.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChunkMetadata:
    """Immutable view of contextual information for a single chunk."""

    sequence: int
    start: int
    end: int
    token_count: Optional[int] = None
    section: Optional[str] = None
    heading_path: Optional[List[str]] = None
    page_number: Optional[int] = None
    source_type: Optional[str] = None
    similarity_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("chunk sequence must be non-negative")
        if self.start < 0 or self.end < 0:
            raise ValueError("chunk character offsets must be non-negative")
        if self.end < self.start:
            raise ValueError("chunk end offset cannot be smaller than start offset")

    @property
    def length(self) -> int:
        """Return the number of characters contained in the chunk."""
        return self.end - self.start

    def as_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary for persistence or APIs."""
        payload: Dict[str, Any] = {
            "sequence": self.sequence,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }
        if self.token_count is not None:
            payload["token_count"] = self.token_count
        if self.section:
            payload["section"] = self.section
        if self.heading_path:
            payload["heading_path"] = self.heading_path
        if self.page_number is not None:
            payload["page_number"] = self.page_number
        if self.source_type:
            payload["source_type"] = self.source_type
        if self.similarity_score is not None:
            payload["similarity_score"] = self.similarity_score
        if self.extra:
            payload["extra"] = self.extra
        return payload.copy()
