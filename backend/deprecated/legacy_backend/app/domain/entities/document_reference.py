"""Document reference entity for chat context (Phase 3 refactoring).

This module defines the DocumentReference entity that represents a document
reference within a chat context, wrapping SearchResult for chat-specific usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import UUID

from app.domain.entities.search_result import SearchResult


@dataclass
class DocumentReference:
    """Domain entity representing a document reference in chat context.

    A lightweight wrapper around SearchResult specifically designed for chat
    conversations, providing document context and citation information.

    Attributes:
        reference_id: Unique identifier for this reference
        search_result: The underlying search result
        relevance_reason: Why this document is relevant to the query
        citation_text: Formatted citation for chat responses
        is_used: Whether this reference was used in generating the response
        context_tokens: Estimated token count for context usage
        metadata: Additional reference metadata
    """

    reference_id: UUID
    search_result: SearchResult
    relevance_reason: Optional[str] = None
    citation_text: Optional[str] = None
    is_used: bool = False
    context_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate document reference after initialization."""
        if not self.search_result:
            raise ValueError("Search result cannot be None")

        # Generate citation text if not provided
        if not self.citation_text:
            self.citation_text = self._generate_citation()

        # Estimate context tokens if not provided
        if self.context_tokens == 0:
            self.context_tokens = self._estimate_tokens()

    def _generate_citation(self) -> str:
        """Generate citation text for this reference.

        Returns:
            Formatted citation string
        """
        doc_metadata = self.search_result.document_metadata
        doc_type = doc_metadata.get("doc_type", "document")
        title = doc_metadata.get("title", f"Document {self.search_result.document_id}")

        return f"[{doc_type}:{title}]"

    def _estimate_tokens(self) -> int:
        """Estimate token count for context usage.

        Returns:
            Estimated token count (rough approximation: 1 token ≈ 4 characters)
        """
        content_length = len(self.search_result.content)
        return max(1, content_length // 4)

    @classmethod
    def from_search_result(
        cls,
        search_result: SearchResult,
        relevance_reason: Optional[str] = None,
        citation_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentReference:
        """Create DocumentReference from SearchResult.

        Args:
            search_result: The underlying search result
            relevance_reason: Why this document is relevant
            citation_text: Custom citation text
            metadata: Additional reference metadata

        Returns:
            New DocumentReference instance
        """
        return DocumentReference(
            reference_id=UUID(),
            search_result=search_result,
            relevance_reason=relevance_reason,
            citation_text=citation_text,
            metadata=metadata or {},
        )

    def mark_as_used(self) -> None:
        """Mark this reference as used in response generation."""
        self.is_used = True

    def mark_as_unused(self) -> None:
        """Mark this reference as unused in response generation."""
        self.is_used = False

    def get_content_excerpt(self, max_length: int = 300) -> str:
        """Get content excerpt for context.

        Args:
            max_length: Maximum length of the excerpt

        Returns:
            Content excerpt with appropriate length
        """
        return self.search_result.get_excerpt(max_length)

    def get_document_type(self) -> str:
        """Get the document type.

        Returns:
            Document type string
        """
        return self.search_result.document_metadata.get("doc_type", "unknown")

    def get_document_title(self) -> str:
        """Get the document title.

        Returns:
            Document title or fallback
        """
        return (
            self.search_result.document_metadata.get("title")
            or f"Document {self.search_result.document_id}"
        )

    def get_similarity_score(self) -> float:
        """Get the similarity score.

        Returns:
            Similarity score as float
        """
        return self.search_result.similarity_score.score

    def is_highly_relevant(self, threshold: float = 0.8) -> bool:
        """Check if this reference is highly relevant.

        Args:
            threshold: Relevance threshold (default: 0.8)

        Returns:
            True if similarity score is above threshold
        """
        return self.get_similarity_score() >= threshold

    def as_dict(self) -> Dict[str, Any]:
        """Convert DocumentReference to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "reference_id": str(self.reference_id),
            "search_result": self.search_result.as_summary_dict(),
            "relevance_reason": self.relevance_reason,
            "citation_text": self.citation_text,
            "is_used": self.is_used,
            "context_tokens": self.context_tokens,
            "metadata": self.metadata,
        }

    def as_citation_dict(self) -> Dict[str, Any]:
        """Convert to citation-focused dictionary.

        Returns:
            Citation-focused dictionary for responses
        """
        return {
            "citation_text": self.citation_text,
            "document_id": str(self.search_result.document_id),
            "document_title": self.get_document_title(),
            "document_type": self.get_document_type(),
            "similarity_score": self.get_similarity_score(),
            "rank": self.search_result.rank,
        }

    def __str__(self) -> str:
        """String representation of document reference."""
        return (
            f"DocumentReference(id={self.reference_id}, "
            f"score={self.get_similarity_score():.3f}, "
            f"doc='{self.get_document_title()}', used={self.is_used})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"DocumentReference(reference_id={self.reference_id}, "
            f"search_result={self.search_result}, "
            f"relevance_reason='{self.relevance_reason}', "
            f"is_used={self.is_used})"
        )