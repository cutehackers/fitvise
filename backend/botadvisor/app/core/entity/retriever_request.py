"""
Retriever Request Entity

Request model for retriever operations.
"""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrieverRequest:
    """
    Request model for retriever operations.

    Attributes:
        query: Search query text
        platform: Optional platform filter (filesystem, web, gdrive, etc.)
        filters: Optional additional metadata filters
        top_k: Maximum number of results to return
    """

    query: str
    platform: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
