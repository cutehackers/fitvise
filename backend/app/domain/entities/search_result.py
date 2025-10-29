"""Search result entity for semantic search responses (Task 2.4.1).

This module defines the SearchResult entity that represents individual search results
with relevance scores, document metadata, and chunk information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.chunk import Chunk
from app.domain.value_objects.similarity_score import SimilarityScore


@dataclass
class SearchResult:
    """Domain entity representing a single search result.

    Encapsulates a retrieved chunk with its relevance score, document context,
    and additional metadata for the search response.

    Attributes:
        result_id: Unique identifier for this search result
        chunk_id: ID of the matched chunk
        document_id: ID of the source document
        content: The actual text content of the chunk
        similarity_score: Similarity score between query and chunk
        rank: Result ranking in the search results (1-based)
        document_metadata: Metadata about the source document
        chunk_metadata: Metadata about the specific chunk
        highlight_text: Highlighted snippet showing relevance
        context_before: Text before the matched content
        context_after: Text after the matched content
        created_at: Timestamp when result was generated
        metadata: Additional result metadata
    """

    result_id: UUID
    chunk_id: UUID
    document_id: UUID
    content: str
    similarity_score: SimilarityScore
    rank: int
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    highlight_text: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    created_at: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate search result after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Search result content cannot be empty")

        if self.rank <= 0:
            raise ValueError("Rank must be greater than 0")

        # Set created_at if not provided
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.utcnow().isoformat()

    @classmethod
    def from_chunk_with_score(
        cls,
        chunk: Chunk,
        similarity_score: SimilarityScore,
        rank: int,
        document_metadata: Optional[Dict[str, Any]] = None,
        highlight_text: Optional[str] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> SearchResult:
        """Create SearchResult from Chunk with similarity score.

        Args:
            chunk: Source chunk entity
            similarity_score: Similarity score object
            rank: Result ranking
            document_metadata: Additional document metadata
            highlight_text: Highlighted text snippet
            context_before: Text before the main content
            context_after: Text after the main content

        Returns:
            New SearchResult instance
        """
        return SearchResult(
            result_id=UUID(),
            chunk_id=UUID(chunk.chunk_id) if isinstance(chunk.chunk_id, str) else chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.text,
            similarity_score=similarity_score,
            rank=rank,
            document_metadata=document_metadata or {},
            chunk_metadata=chunk.metadata.as_dict() if hasattr(chunk.metadata, 'as_dict') else {},
            highlight_text=highlight_text,
            context_before=context_before,
            context_after=context_after,
        )

    @classmethod
    def create(
        cls,
        chunk_id: UUID,
        document_id: UUID,
        content: str,
        similarity_score: SimilarityScore,
        rank: int,
        document_metadata: Optional[Dict[str, Any]] = None,
        chunk_metadata: Optional[Dict[str, Any]] = None,
        highlight_text: Optional[str] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> SearchResult:
        """Factory method to create a new SearchResult.

        Args:
            chunk_id: ID of the matched chunk
            document_id: ID of the source document
            content: Text content of the chunk
            similarity_score: Similarity score object
            rank: Result ranking
            document_metadata: Document metadata
            chunk_metadata: Chunk metadata
            highlight_text: Highlighted text snippet
            context_before: Text before the main content
            context_after: Text after the main content

        Returns:
            New SearchResult instance
        """
        return SearchResult(
            result_id=UUID(),
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            similarity_score=similarity_score,
            rank=rank,
            document_metadata=document_metadata or {},
            chunk_metadata=chunk_metadata or {},
            highlight_text=highlight_text,
            context_before=context_before,
            context_after=context_after,
        )

    def with_highlight(
        self,
        highlight_text: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> SearchResult:
        """Return a copy with updated highlighting.

        Args:
            highlight_text: Highlighted text snippet
            context_before: Text before the main content
            context_after: Text after the main content

        Returns:
            New SearchResult with updated highlighting
        """
        return SearchResult(
            result_id=self.result_id,
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            content=self.content,
            similarity_score=self.similarity_score,
            rank=self.rank,
            document_metadata=self.document_metadata.copy(),
            chunk_metadata=self.chunk_metadata.copy(),
            highlight_text=highlight_text,
            context_before=context_before,
            context_after=context_after,
            created_at=self.created_at,
            metadata=self.metadata.copy(),
        )

    def is_above_threshold(self, threshold: float) -> bool:
        """Check if result meets minimum similarity threshold.

        Args:
            threshold: Minimum similarity threshold

        Returns:
            True if similarity score is above threshold
        """
        return self.similarity_score.score >= threshold

    def get_excerpt(self, max_length: int = 200) -> str:
        """Get a concise excerpt of the content.

        Args:
            max_length: Maximum length of the excerpt

        Returns:
            Truncated content with ellipsis if needed
        """
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length - 3] + "..."

    def as_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to dictionary.

        Returns:
            Dictionary representation of the search result
        """
        return {
            "result_id": str(self.result_id),
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content,
            "similarity_score": self.similarity_score.as_dict(),
            "rank": self.rank,
            "document_metadata": self.document_metadata,
            "chunk_metadata": self.chunk_metadata,
            "highlight_text": self.highlight_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def as_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary with essential fields only.

        Returns:
            Concise dictionary representation for API responses
        """
        return {
            "result_id": str(self.result_id),
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.get_excerpt(),
            "similarity_score": self.similarity_score.score,
            "rank": self.rank,
            "doc_type": self.document_metadata.get("doc_type", "unknown"),
            "highlight_text": self.highlight_text,
        }

    def __str__(self) -> str:
        """String representation of search result."""
        return (
            f"SearchResult(id={self.result_id}, rank={self.rank}, "
            f"score={self.similarity_score.score:.3f}, "
            f"content='{self.get_excerpt()}')"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SearchResult(result_id={self.result_id}, chunk_id={self.chunk_id}, "
            f"document_id={self.document_id}, similarity_score={self.similarity_score}, "
            f"rank={self.rank}, content_length={len(self.content)})"
        )