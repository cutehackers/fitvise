"""
Document Entity

Minimal document identity model used across the RAG pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

@dataclass(frozen=True)
class Document:
    """
    Minimal document identity model used across the RAG pipeline.

    This model supports:
    - Deduplication via checksum during ingestion
    - Citation metadata enrichment in retriever results
    - Traceability across storage, retrieval, and API responses

    Attributes:
        id: Unique identifier (UUID v4) for this document record
        source_id: Platform-native identifier (e.g., file path, URL, GDrive ID)
        platform: Source platform type (filesystem, web, gdrive, etc.)
        source_url: Original source URL or path for citation
        checksum: SHA-256 hash of raw content for deduplication
        size_bytes: Content size in bytes
        mime_type: MIME type of the source content
        created_at: UTC timestamp when document was ingested
    """

    id: str
    source_id: str
    platform: str
    source_url: str
    checksum: str
    size_bytes: int
    mime_type: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        source_id: str,
        platform: str,
        source_url: str,
        content: bytes,
        mime_type: str,
        id: str | None = None,
    ) -> Document:
        """
        Create a new Document instance with computed checksum.

        Args:
            source_id: Platform-native identifier
            platform: Source platform type
            source_url: Original source URL or path
            content: Raw content bytes
            mime_type: MIME type of content
            id: Optional custom ID (defaults to UUID v4)

        Returns:
            A new Document instance with SHA-256 checksum computed from content
        """
        checksum = compute_checksum(content)
        return cls(
            id=id or str(uuid4()),
            source_id=source_id,
            platform=platform,
            source_url=source_url,
            checksum=checksum,
            size_bytes=len(content),
            mime_type=mime_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "platform": self.platform,
            "source_url": self.source_url,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary for deserialization."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            platform=data["platform"],
            source_url=data["source_url"],
            checksum=data["checksum"],
            size_bytes=data["size_bytes"],
            mime_type=data["mime_type"],
            created_at=created_at or datetime.now(timezone.utc),
        )

def compute_checksum(content: bytes, algorithm: str = "sha256") -> str:
    """
    Compute checksum of content for deduplication.

    Args:
        content: Content bytes to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hexadecimal checksum string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return hasher.hexdigest()

def compute_file_checksum(path: str, algorithm: str = "sha256") -> str:
    """
    Compute checksum of file for deduplication.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hexadecimal checksum string
    """
    from pathlib import Path
    path_obj = Path(path)
    hasher = hashlib.new(algorithm)
    with path_obj.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
