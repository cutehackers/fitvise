"""
Document Metadata Entity

Lightweight metadata wrapper for documents in retriever results.
"""

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DocumentMetadata:
    """
    Lightweight metadata wrapper for documents in retriever results.

    Used by retrieval layer to attach citation information to chunks.
    """

    doc_id: str
    source_id: str
    platform: str
    source_url: str
    page: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "platform": self.platform,
            "source_url": self.source_url,
            "page": self.page,
            "section": self.section,
        }
