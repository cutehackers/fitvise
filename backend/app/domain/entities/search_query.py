"""Search query entity for semantic search requests (Task 2.4.1).

This module defines the SearchQuery entity that represents user search requests
with filters, pagination, and relevance configuration for semantic search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.domain.value_objects.search_filters import SearchFilters


@dataclass
class SearchQuery:
    """Domain entity representing a semantic search query.

    Encapsulates user search intent including the query text, result limits,
    filtering criteria, and relevance configuration.

    Attributes:
        query_id: Unique identifier for the search query
        text: The search query text
        filters: Search filters for document type, date range, etc.
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity threshold (0.0-1.0)
        include_metadata: Whether to include full document metadata
        session_id: Optional session identifier for conversation context
        user_id: Optional user identifier for personalization
        created_at: Timestamp when query was created
        metadata: Additional query metadata
    """

    query_id: UUID
    text: str
    filters: SearchFilters
    top_k: int = 10
    min_similarity: float = 0.0
    include_metadata: bool = True
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    created_at: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate search query after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Search query text cannot be empty")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")

        # Set created_at if not provided
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.utcnow().isoformat()

    @classmethod
    def create(
        cls,
        text: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = 10,
        min_similarity: float = 0.0,
        include_metadata: bool = True,
        session_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchQuery:
        """Factory method to create a new SearchQuery.

        Args:
            text: Search query text
            filters: Optional search filters
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            include_metadata: Whether to include document metadata
            session_id: Optional session identifier
            user_id: Optional user identifier
            metadata: Additional query metadata

        Returns:
            New SearchQuery instance
        """
        return SearchQuery(
            query_id=uuid4(),
            text=text.strip(),
            filters=filters or SearchFilters(),
            top_k=top_k,
            min_similarity=min_similarity,
            include_metadata=include_metadata,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )

    def with_filters(self, filters: SearchFilters) -> SearchQuery:
        """Return a copy with updated filters.

        Args:
            filters: New search filters

        Returns:
            New SearchQuery with updated filters
        """
        return SearchQuery(
            query_id=self.query_id,
            text=self.text,
            filters=filters,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            include_metadata=self.include_metadata,
            session_id=self.session_id,
            user_id=self.user_id,
            created_at=self.created_at,
            metadata=self.metadata.copy(),
        )

    def with_top_k(self, top_k: int) -> SearchQuery:
        """Return a copy with updated top_k.

        Args:
            top_k: New maximum number of results

        Returns:
            New SearchQuery with updated top_k
        """
        return SearchQuery(
            query_id=self.query_id,
            text=self.text,
            filters=self.filters,
            top_k=top_k,
            min_similarity=self.min_similarity,
            include_metadata=self.include_metadata,
            session_id=self.session_id,
            user_id=self.user_id,
            created_at=self.created_at,
            metadata=self.metadata.copy(),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert SearchQuery to dictionary.

        Returns:
            Dictionary representation of the search query
        """
        return {
            "query_id": str(self.query_id),
            "text": self.text,
            "filters": self.filters.as_dict(),
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "include_metadata": self.include_metadata,
            "session_id": str(self.session_id) if self.session_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation of search query."""
        return f"SearchQuery(id={self.query_id}, text='{self.text[:50]}...', top_k={self.top_k})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SearchQuery(query_id={self.query_id}, text='{self.text}', "
            f"top_k={self.top_k}, min_similarity={self.min_similarity}, "
            f"filters={self.filters})"
        )