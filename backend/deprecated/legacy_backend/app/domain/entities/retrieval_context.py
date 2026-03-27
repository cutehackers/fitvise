"""Retrieval context entity for managing chat context windows (Phase 3 refactoring).

This module defines the RetrievalContext entity that manages context windows,
document fitting, and token usage for chat conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.document_reference import DocumentReference


@dataclass
class RetrievalContext:
    """Domain entity representing a retrieval context for chat.

    Manages context windows, document fitting, and token usage to ensure
    optimal context utilization within LLM token limits.

    Attributes:
        context_id: Unique identifier for this context
        query: The original query that generated this context
        document_references: List of document references in the context
        total_tokens: Total token count for the entire context
        max_tokens: Maximum allowed tokens for the context
        context_text: Formatted context text for LLM consumption
        metadata: Additional context metadata
        is_complete: Whether the context fitting process is complete
        truncated_count: Number of documents that couldn't fit
    """

    context_id: UUID
    query: str
    document_references: List[DocumentReference] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 4000  # Default context window limit
    context_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    truncated_count: int = 0

    def __post_init__(self) -> None:
        """Validate retrieval context after initialization."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")

        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be greater than 0")

        # Generate context text if not provided
        if self.context_text is None and self.document_references:
            self._generate_context_text()

    def add_reference(self, reference: DocumentReference) -> bool:
        """Add a document reference to the context.

        Args:
            reference: Document reference to add

        Returns:
            True if reference was added, False if it would exceed token limit
        """
        if self.total_tokens + reference.context_tokens > self.max_tokens:
            self.truncated_count += 1
            return False

        self.document_references.append(reference)
        self.total_tokens += reference.context_tokens
        reference.mark_as_used()
        return True

    def add_references(self, references: List[DocumentReference]) -> int:
        """Add multiple document references, fitting as many as possible.

        Args:
            references: List of document references to add

        Returns:
            Number of references successfully added
        """
        added_count = 0
        # Sort by relevance (highest similarity score first)
        sorted_refs = sorted(
            references,
            key=lambda ref: ref.get_similarity_score(),
            reverse=True,
        )

        for reference in sorted_refs:
            if self.add_reference(reference):
                added_count += 1
            else:
                break

        # Regenerate context text
        self._generate_context_text()
        return added_count

    def remove_reference(self, reference_id: UUID) -> bool:
        """Remove a document reference from the context.

        Args:
            reference_id: ID of reference to remove

        Returns:
            True if reference was removed, False if not found
        """
        for i, ref in enumerate(self.document_references):
            if ref.reference_id == reference_id:
                removed_ref = self.document_references.pop(i)
                self.total_tokens -= removed_ref.context_tokens
                removed_ref.mark_as_unused()
                self._generate_context_text()
                return True
        return False

    def clear_references(self) -> None:
        """Clear all document references."""
        for ref in self.document_references:
            ref.mark_as_unused()
        self.document_references.clear()
        self.total_tokens = 0
        self.truncated_count = 0
        self.context_text = ""

    def fit_to_token_limit(self) -> None:
        """Fit references to token limit by removing least relevant items.

        Removes references until total tokens are within the limit,
        prioritizing higher similarity scores.
        """
        if self.total_tokens <= self.max_tokens:
            return

        # Sort by relevance (lowest similarity first for removal)
        sorted_refs = sorted(
            self.document_references,
            key=lambda ref: ref.get_similarity_score(),
        )

        # Remove references until within token limit
        for ref in sorted_refs:
            if self.total_tokens <= self.max_tokens:
                break

            self.remove_reference(ref.reference_id)
            self.truncated_count += 1

        self.is_complete = True

    def _generate_context_text(self) -> None:
        """Generate formatted context text for LLM consumption."""
        if not self.document_references:
            self.context_text = ""
            return

        context_parts = ["Context from relevant documents:"]

        for ref in self.document_references:
            doc_title = ref.get_document_title()
            doc_type = ref.get_document_type()
            content = ref.get_content_excerpt(500)  # Limit per document

            context_parts.append(
                f"\n[{doc_type}: {doc_title}]\n{content}"
            )

        self.context_text = "\n".join(context_parts)

    def get_token_usage_ratio(self) -> float:
        """Get the ratio of used tokens to maximum tokens.

        Returns:
            Token usage ratio (0.0 to 1.0)
        """
        return self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    def is_near_limit(self, threshold: float = 0.9) -> bool:
        """Check if context is near token limit.

        Args:
            threshold: Usage threshold (default: 0.9)

        Returns:
            True if usage is above threshold
        """
        return self.get_token_usage_ratio() >= threshold

    def has_space_for(self, tokens: int) -> bool:
        """Check if context has space for additional tokens.

        Args:
            tokens: Number of tokens to check

        Returns:
            True if space is available
        """
        return self.total_tokens + tokens <= self.max_tokens

    def get_highly_relevant_references(self, threshold: float = 0.8) -> List[DocumentReference]:
        """Get highly relevant document references.

        Args:
            threshold: Relevance threshold

        Returns:
            List of highly relevant references
        """
        return [
            ref for ref in self.document_references
            if ref.is_highly_relevant(threshold)
        ]

    def get_unused_references(self) -> List[DocumentReference]:
        """Get references that were marked as unused.

        Returns:
            List of unused references
        """
        return [ref for ref in self.document_references if not ref.is_used]

    def get_summary(self) -> Dict[str, Any]:
        """Get context summary statistics.

        Returns:
            Dictionary with context statistics
        """
        if not self.document_references:
            return {
                "total_documents": 0,
                "total_tokens": 0,
                "usage_ratio": 0.0,
                "truncated_count": self.truncated_count,
                "avg_similarity": 0.0,
            }

        similarities = [ref.get_similarity_score() for ref in self.document_references]

        return {
            "total_documents": len(self.document_references),
            "total_tokens": self.total_tokens,
            "usage_ratio": self.get_token_usage_ratio(),
            "truncated_count": self.truncated_count,
            "avg_similarity": sum(similarities) / len(similarities),
            "max_similarity": max(similarities),
            "min_similarity": min(similarities),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Convert RetrievalContext to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "context_id": str(self.context_id),
            "query": self.query,
            "document_references": [ref.as_dict() for ref in self.document_references],
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "context_text": self.context_text,
            "metadata": self.metadata,
            "is_complete": self.is_complete,
            "truncated_count": self.truncated_count,
            "summary": self.get_summary(),
        }

    def __str__(self) -> str:
        """String representation of retrieval context."""
        return (
            f"RetrievalContext(id={self.context_id}, "
            f"docs={len(self.document_references)}, "
            f"tokens={self.total_tokens}/{self.max_tokens}, "
            f"usage={self.get_token_usage_ratio():.1%})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"RetrievalContext(context_id={self.context_id}, "
            f"query='{self.query[:50]}...', "
            f"document_references={len(self.document_references)}, "
            f"total_tokens={self.total_tokens}, "
            f"max_tokens={self.max_tokens}, "
            f"is_complete={self.is_complete})"
        )