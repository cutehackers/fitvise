"""Context service for managing retrieval contexts (Phase 3 refactoring).

This module defines the ContextService domain service that manages context windows,
document fitting, and token utilization for chat conversations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.document_reference import DocumentReference
from app.domain.entities.retrieval_context import RetrievalContext
from app.domain.exceptions.retrieval_exceptions import ContextBuildingError


@dataclass
class ContextBuildingConfig:
    """Configuration for context building operations.

    Attributes:
        default_max_tokens: Default maximum tokens for context
        min_similarity_threshold: Minimum similarity for documents to include
        max_documents_per_context: Maximum documents to consider for context
        context_compression_ratio: Target compression ratio for context
        enable_context_summarization: Whether to enable context summarization
    """

    default_max_tokens: int = 4000
    min_similarity_threshold: float = 0.0
    max_documents_per_context: int = 20
    context_compression_ratio: float = 0.8
    enable_context_summarization: bool = False


class ContextService:
    """Domain service for managing retrieval contexts.

    Provides business logic for context window management, document fitting,
    and token utilization optimization for chat conversations.

    Responsibilities:
    - Build retrieval contexts from document references
    - Optimize context window utilization
    - Manage document prioritization and fitting
    - Handle context compression and summarization
    - Track context statistics and metrics

    Examples:
        >>> service = ContextService()
        >>> context = await service.build_context(
        ...     query="What exercises help with back pain?",
        ...     document_references=references,
        ...     max_tokens=3000
        ... )
        >>> len(context.document_references)
        5
        >>> context.get_token_usage_ratio()
        0.75
    """

    def __init__(self, config: Optional[ContextBuildingConfig] = None) -> None:
        """Initialize context service.

        Args:
            config: Optional context building configuration
        """
        self._config = config or ContextBuildingConfig()

    async def build_context(
        self,
        query: str,
        document_references: List[DocumentReference],
        max_tokens: int,
        context_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalContext:
        """Build a retrieval context from document references.

        Args:
            query: The query that generated the documents
            document_references: List of document references to include
            max_tokens: Maximum tokens allowed in context
            context_id: Optional context ID (will generate if not provided)
            metadata: Optional context metadata

        Returns:
            RetrievalContext with fitted documents

        Raises:
            ContextBuildingError: If context building fails
        """
        try:
            if not query or not query.strip():
                raise ContextBuildingError("Query cannot be empty")

            if not document_references:
                # Return empty context
                return RetrievalContext(
                    context_id=context_id or UUID(),
                    query=query.strip(),
                    max_tokens=max_tokens,
                    metadata=metadata or {},
                    is_complete=True,
                )

            # Filter by similarity threshold
            filtered_references = self._filter_by_similarity(document_references)

            # Sort by relevance (similarity score, rank)
            sorted_references = self._sort_by_relevance(filtered_references)

            # Create context and fit documents
            context = RetrievalContext(
                context_id=context_id or UUID(),
                query=query.strip(),
                max_tokens=max_tokens,
                metadata=metadata or {},
            )

            # Add references within token limit
            added_count = context.add_references(sorted_references)

            # Mark context as complete and fit if necessary
            context.fit_to_token_limit()
            context.is_complete = True

            return context

        except Exception as e:
            if isinstance(e, ContextBuildingError):
                raise
            raise ContextBuildingError(f"Failed to build context: {str(e)}") from e

    def _filter_by_similarity(
        self,
        document_references: List[DocumentReference],
    ) -> List[DocumentReference]:
        """Filter document references by similarity threshold.

        Args:
            document_references: List of document references to filter

        Returns:
            Filtered list of document references
        """
        threshold = self._config.min_similarity_threshold
        if threshold <= 0.0:
            return document_references

        return [
            ref for ref in document_references
            if ref.get_similarity_score() >= threshold
        ]

    def _sort_by_relevance(
        self,
        document_references: List[DocumentReference],
    ) -> List[DocumentReference]:
        """Sort document references by relevance.

        Args:
            document_references: List of document references to sort

        Returns:
            Sorted list of document references
        """
        # Sort by similarity score (descending), then by rank (ascending)
        return sorted(
            document_references,
            key=lambda ref: (-ref.get_similarity_score(), ref.search_result.rank),
        )

    async def optimize_context(
        self,
        context: RetrievalContext,
        optimization_target: str = "tokens",
    ) -> RetrievalContext:
        """Optimize an existing context.

        Args:
            context: The context to optimize
            optimization_target: Optimization goal ("tokens", "quality", "diversity")

        Returns:
            Optimized retrieval context
        """
        try:
            if optimization_target == "tokens":
                return await self._optimize_for_tokens(context)
            elif optimization_target == "quality":
                return await self._optimize_for_quality(context)
            elif optimization_target == "diversity":
                return await self._optimize_for_diversity(context)
            else:
                raise ContextBuildingError(f"Unknown optimization target: {optimization_target}")

        except Exception as e:
            raise ContextBuildingError(f"Failed to optimize context: {str(e)}") from e

    async def _optimize_for_tokens(self, context: RetrievalContext) -> RetrievalContext:
        """Optimize context for token efficiency.

        Args:
            context: Context to optimize

        Returns:
            Token-optimized context
        """
        if not context.is_near_limit():
            return context

        # Remove least relevant documents until within comfortable limit
        target_ratio = self._config.context_compression_ratio
        target_tokens = int(context.max_tokens * target_ratio)

        while context.total_tokens > target_tokens and context.document_references:
            # Find least relevant document
            least_relevant = min(
                context.document_references,
                key=lambda ref: ref.get_similarity_score(),
            )
            context.remove_reference(least_relevant.reference_id)

        context.is_complete = True
        return context

    async def _optimize_for_quality(self, context: RetrievalContext) -> RetrievalContext:
        """Optimize context for quality (highest relevance).

        Args:
            context: Context to optimize

        Returns:
            Quality-optimized context
        """
        # Re-sort by relevance and refit
        sorted_refs = sorted(
            context.document_references,
            key=lambda ref: (-ref.get_similarity_score(), ref.search_result.rank),
        )

        # Clear and re-add in order
        context.clear_references()
        context.add_references(sorted_refs)
        context.fit_to_token_limit()
        context.is_complete = True

        return context

    async def _optimize_for_diversity(self, context: RetrievalContext) -> RetrievalContext:
        """Optimize context for document diversity.

        Args:
            context: Context to optimize

        Returns:
            Diversity-optimized context
        """
        # Group by document type to ensure diversity
        doc_type_groups = {}
        for ref in context.document_references:
            doc_type = ref.get_document_type()
            if doc_type not in doc_type_groups:
                doc_type_groups[doc_type] = []
            doc_type_groups[doc_type].append(ref)

        # Clear and re-add with diversity
        context.clear_references()

        # Add documents round-robin from different types
        max_rounds = max(len(group) for group in doc_type_groups.values()) if doc_type_groups else 0
        for round_num in range(max_rounds):
            for doc_type, references in doc_type_groups.items():
                if round_num < len(references):
                    context.add_reference(references[round_num])

        context.fit_to_token_limit()
        context.is_complete = True
        return context

    def calculate_context_quality_score(self, context: RetrievalContext) -> float:
        """Calculate quality score for a context.

        Args:
            context: Context to score

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not context.document_references:
            return 0.0

        # Factors: avg similarity, document diversity, token utilization
        avg_similarity = sum(ref.get_similarity_score() for ref in context.document_references) / len(context.document_references)

        # Document diversity (unique document types)
        doc_types = set(ref.get_document_type() for ref in context.document_references)
        diversity_score = len(doc_types) / len(context.document_references)

        # Token utilization (prefer moderate utilization, not too high or too low)
        utilization = context.get_token_usage_ratio()
        utilization_score = 1.0 - abs(utilization - 0.7)  # Optimal around 70%

        # Weighted combination
        quality_score = (
            avg_similarity * 0.5 +
            diversity_score * 0.3 +
            utilization_score * 0.2
        )

        return max(0.0, min(1.0, quality_score))

    def get_context_statistics(self, context: RetrievalContext) -> Dict[str, Any]:
        """Get detailed statistics for a context.

        Args:
            context: Context to analyze

        Returns:
            Dictionary with context statistics
        """
        if not context.document_references:
            return {
                "document_count": 0,
                "total_tokens": 0,
                "utilization_ratio": 0.0,
                "quality_score": 0.0,
                "document_types": [],
                "avg_similarity": 0.0,
            }

        similarities = [ref.get_similarity_score() for ref in context.document_references]
        doc_types = [ref.get_document_type() for ref in context.document_references]

        return {
            "document_count": len(context.document_references),
            "total_tokens": context.total_tokens,
            "utilization_ratio": context.get_token_usage_ratio(),
            "quality_score": self.calculate_context_quality_score(context),
            "document_types": list(set(doc_types)),
            "document_type_distribution": {
                doc_type: doc_types.count(doc_type) for doc_type in set(doc_types)
            },
            "avg_similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
            "truncated_count": context.truncated_count,
            "is_complete": context.is_complete,
        }

    async def merge_contexts(
        self,
        contexts: List[RetrievalContext],
        max_tokens: Optional[int] = None,
    ) -> RetrievalContext:
        """Merge multiple contexts into one.

        Args:
            contexts: List of contexts to merge
            max_tokens: Maximum tokens for merged context (uses max of contexts if not provided)

        Returns:
            Merged retrieval context
        """
        if not contexts:
            raise ContextBuildingError("Cannot merge empty context list")

        if len(contexts) == 1:
            return contexts[0]

        # Determine max tokens
        merged_max_tokens = max_tokens or max(ctx.max_tokens for ctx in contexts)

        # Combine all document references
        all_references = []
        for context in contexts:
            all_references.extend(context.document_references)

        # Remove duplicates (same document ID) keeping highest similarity
        unique_references = {}
        for ref in all_references:
            doc_id = ref.search_result.document_id
            if doc_id not in unique_references or ref.get_similarity_score() > unique_references[doc_id].get_similarity_score():
                unique_references[doc_id] = ref

        # Create merged context
        merged_query = f"Merged context from {len(contexts)} queries"
        merged_context = RetrievalContext(
            context_id=UUID(),
            query=merged_query,
            max_tokens=merged_max_tokens,
            metadata={
                "merged_contexts": len(contexts),
                "original_queries": [ctx.query for ctx in contexts],
                "merge_timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Add unique references (sorted by relevance)
        sorted_references = sorted(
            unique_references.values(),
            key=lambda ref: (-ref.get_similarity_score(), ref.search_result.rank),
        )

        merged_context.add_references(sorted_references)
        merged_context.fit_to_token_limit()
        merged_context.is_complete = True

        return merged_context