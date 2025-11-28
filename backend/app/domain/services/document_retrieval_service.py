"""Document retrieval service for coordinated retrieval and context building.

This module defines the DocumentRetrievalService domain service that coordinates
between retrieval operations and context building for document retrieval workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.document_reference import DocumentReference
from app.domain.entities.retrieval_context import RetrievalContext
from app.domain.exceptions.retrieval_exceptions import RetrievalError, ContextBuildingError
from app.domain.services.context_service import ContextService
from app.domain.services.retrieval_service import RetrievalService


class DocumentRetrievalService:
    """Domain service for coordinated document retrieval operations.

    Coordinates between semantic search and context building to provide
    complete document retrieval functionality with optimal token utilization
    and relevance ranking.

    Examples:
        >>> service = DocumentRetrievalService(retrieval_service, context_service)
        >>> # Retrieve documents with context
        >>> context = await service.retrieve_documents(
        ...     query="What exercises help with back pain?",
        ...     max_context_tokens=3000,
        ...     max_documents=10
        ... )
        >>> len(context.document_references)
        5
        >>> context.get_token_usage_ratio()
        0.75
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_service: ContextService,
    ) -> None:
        """Initialize document retrieval service.

        Args:
            retrieval_service: Service for semantic search operations
            context_service: Service for context building
        """
        self._retrieval_service = retrieval_service
        self._context_service = context_service

    async def retrieve_documents(
        self,
        query: str,
        max_context_tokens: int = 4000,
        max_documents: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0,
        context_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalContext:
        """Retrieve documents and build context.

        Coordinates semantic search with context building to provide
        optimized document retrieval within token constraints.

        Args:
            query: Search query
            max_context_tokens: Maximum tokens for context
            max_documents: Maximum documents to consider
            filters: Optional search filters
            min_similarity: Minimum similarity threshold
            context_id: Optional context ID
            metadata: Additional context metadata

        Returns:
            RetrievalContext with fitted documents

        Raises:
            RetrievalError: If retrieval fails
            ContextBuildingError: If context building fails
        """
        try:
            # Step 1: Execute semantic search
            search_results = await self._retrieval_service.semantic_search(
                query=query.strip(),
                top_k=max_documents,
                min_similarity=min_similarity,
                filters=filters,
                use_cache=True,
                include_metadata=True,
            )

            # Step 2: Convert search results to document references
            document_references = self._convert_to_document_references(
                search_results, query.strip()
            )

            # Step 3: Build retrieval context
            retrieval_context = await self._context_service.build_context(
                query=query.strip(),
                document_references=document_references,
                max_tokens=max_context_tokens,
                context_id=context_id,
                metadata=metadata or {},
            )

            return retrieval_context

        except Exception as e:
            if isinstance(e, (RetrievalError, ContextBuildingError)):
                raise
            raise RetrievalError(
                message="Failed to retrieve documents",
                operation="retrieve_documents",
                details=str(e),
            ) from e

    async def retrieve_documents_with_metrics(
        self,
        query: str,
        max_context_tokens: int = 4000,
        max_documents: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0,
        context_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve documents with detailed performance metrics.

        Args:
            query: Search query
            max_context_tokens: Maximum tokens for context
            max_documents: Maximum documents to consider
            filters: Optional search filters
            min_similarity: Minimum similarity threshold
            context_id: Optional context ID
            metadata: Additional context metadata

        Returns:
            Dictionary with context and performance metrics

        Raises:
            RetrievalError: If retrieval fails
            ContextBuildingError: If context building fails
        """
        import time
        from datetime import datetime

        start_time = time.time()

        try:
            # Retrieve documents
            context = await self.retrieve_documents(
                query=query,
                max_context_tokens=max_context_tokens,
                max_documents=max_documents,
                filters=filters,
                min_similarity=min_similarity,
                context_id=context_id,
                metadata=metadata,
            )

            # Calculate metrics
            total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get search metrics from retrieval service
            search_metrics = await self._retrieval_service.get_retrieval_metrics()

            # Calculate additional metrics
            avg_similarity = 0.0
            if context.document_references:
                similarities = [
                    ref.get_similarity_score() for ref in context.document_references
                ]
                avg_similarity = sum(similarities) / len(similarities)

            return {
                "context": context,
                "metrics": {
                    "total_processing_time_ms": total_time,
                    "total_documents_found": len(context.document_references) + context.truncated_count,
                    "documents_in_context": len(context.document_references),
                    "documents_truncated": context.truncated_count,
                    "context_utilization": context.get_token_usage_ratio(),
                    "avg_similarity_score": avg_similarity,
                    "context_quality_score": self._context_service.calculate_context_quality_score(context),
                    "search_service_status": search_metrics.get("overall_status", "unknown"),
                    "query_length": len(query),
                    "max_context_tokens": max_context_tokens,
                    "max_documents_requested": max_documents,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "search_metrics": search_metrics,
            }

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            return {
                "context": None,
                "metrics": {
                    "total_processing_time_ms": total_time,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "search_metrics": {"error": str(e)},
            }

    async def retrieve_additional_documents(
        self,
        context_id: UUID,
        additional_query: str,
        max_additional_documents: int = 5,
        max_context_tokens: Optional[int] = None,
        min_similarity: float = 0.0,
    ) -> RetrievalContext:
        """Retrieve additional documents for existing context.

        Args:
            context_id: ID of existing context to augment
            additional_query: Additional search query
            max_additional_documents: Maximum additional documents
            max_context_tokens: New max tokens (keeps original if not provided)
            min_similarity: Minimum similarity threshold

        Returns:
            Augmented retrieval context

        Raises:
            RetrievalError: If additional retrieval fails
        """
        try:
            # For now, create new context with both queries
            # In a real implementation, this would load existing context and augment it
            combined_query = f"{additional_query} (additional)"

            context = await self.retrieve_documents(
                query=combined_query,
                max_context_tokens=max_context_tokens or 4000,
                max_documents=max_additional_documents,
                min_similarity=min_similarity,
                context_id=context_id,
                metadata={"augmented": True, "original_context_id": str(context_id)},
            )

            return context

        except Exception as e:
            raise RetrievalError(
                message="Failed to retrieve additional documents",
                operation="retrieve_additional_documents",
                details=str(e),
            ) from e

    async def optimize_context(
        self,
        context: RetrievalContext,
        optimization_target: str = "tokens",
    ) -> RetrievalContext:
        """Optimize an existing context.

        Delegates to context service for optimization.

        Args:
            context: Context to optimize
            optimization_target: Optimization goal ("tokens", "quality", "diversity")

        Returns:
            Optimized retrieval context
        """
        return await self._context_service.optimize_context(context, optimization_target)

    def _convert_to_document_references(
        self,
        search_results: List[Any],
        query: str,
    ) -> List[DocumentReference]:
        """Convert search results to document references.

        Args:
            search_results: List of search results from retrieval service
            query: Original query for relevance scoring

        Returns:
            List of document references
        """
        document_references = []
        for result in search_results:
            # Create relevance reason based on similarity score
            relevance_reason = f"Similarity score: {result.similarity_score.score:.3f}"
            if result.similarity_score.score >= 0.9:
                relevance_reason = "Highly relevant match"
            elif result.similarity_score.score >= 0.7:
                relevance_reason = "Relevant match"
            else:
                relevance_reason = "Potentially relevant match"

            # Create document reference
            reference = DocumentReference.from_search_result(
                search_result=result,
                relevance_reason=relevance_reason,
                metadata={
                    "query": query,
                    "similarity_score": result.similarity_score.score,
                    "conversion_timestamp": str(result.created_at) if hasattr(result, 'created_at') else None,
                },
            )
            document_references.append(reference)

        return document_references

    async def get_retrieval_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for document retrieval.

        Returns:
            Dictionary with performance statistics
        """
        try:
            # Get retrieval service metrics
            retrieval_metrics = await self._retrieval_service.get_retrieval_metrics()

            # Get context statistics (create a dummy context to use context service methods)
            dummy_context = RetrievalContext(
                context_id=UUID(),
                query="performance_check",
                max_tokens=4000,
                metadata={"operation": "metrics_check"},
            )

            context_stats = self._context_service.get_context_statistics(dummy_context)

            return {
                "retrieval_service_metrics": retrieval_metrics,
                "context_service_stats": context_stats,
                "document_retrieval_status": "operational",
                "overall_status": retrieval_metrics.get("overall_status", "unknown"),
            }
        except Exception as e:
            return {
                "error": f"Failed to get performance metrics: {str(e)}",
                "retrieval_service_metrics": {},
                "context_service_stats": {},
                "document_retrieval_status": "error",
                "overall_status": "error",
            }