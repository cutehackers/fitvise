"""Retrieve documents use case for chat context building (Phase 3 refactoring).

This module implements the RetrieveDocumentUseCase that coordinates semantic search
with context building to provide documents for chat conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.application.use_cases.retrieval.semantic_search import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchUseCase,
)
from app.domain.entities.document_reference import DocumentReference
from app.domain.entities.retrieval_context import RetrievalContext
from app.domain.entities.search_result import SearchResult
from app.domain.exceptions.retrieval_exceptions import (
    RetrievalError,
    ContextBuildingError,
)
from app.domain.services.context_service import ContextService


@dataclass
class RetrieveDocumentRequest:
    """Request for document retrieval with context building.

    Encapsulates the search query and context configuration options.
    """

    query: str
    max_context_tokens: int = 4000
    max_documents: int = 10
    min_similarity: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    include_metadata: bool = True
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_semantic_search_request(self) -> SemanticSearchRequest:
        """Convert to semantic search request.

        Returns:
            SemanticSearchRequest for underlying search
        """
        return SemanticSearchRequest(
            query_text=self.query,
            top_k=self.max_documents,
            min_similarity=self.min_similarity,
            filters=self.filters,
            include_metadata=self.include_metadata,
            session_id=self.session_id,
            user_id=self.user_id,
            search_metadata=self.retrieval_metadata,
        )


@dataclass
class RetrieveDocumentResponse:
    """Response from document retrieval with context.

    Contains the retrieval context with documents and metrics.
    """

    success: bool
    context_id: UUID
    context: Optional[RetrievalContext] = None
    total_documents_found: int = 0
    documents_in_context: int = 0
    documents_truncated: int = 0
    processing_time_ms: float = 0.0
    search_time_ms: float = 0.0
    context_time_ms: float = 0.0
    avg_similarity_score: float = 0.0
    context_utilization: float = 0.0  # Ratio of used tokens to max tokens
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the retrieval response
        """
        return {
            "success": self.success,
            "context_id": str(self.context_id),
            "context": self.context.as_dict() if self.context else None,
            "total_documents_found": self.total_documents_found,
            "documents_in_context": self.documents_in_context,
            "documents_truncated": self.documents_truncated,
            "processing_time_ms": self.processing_time_ms,
            "search_time_ms": self.search_time_ms,
            "context_time_ms": self.context_time_ms,
            "avg_similarity_score": self.avg_similarity_score,
            "context_utilization": self.context_utilization,
            "error": self.error,
            "metadata": self.metadata,
        }


class RetrieveDocumentUseCase:
    """Use case for retrieving documents with context building (Phase 3).

    Implements the document retrieval workflow for chat:
    1. Execute semantic search using existing SemanticSearchUseCase
    2. Convert search results to document references
    3. Build retrieval context with token management
    4. Return context with fitted documents

    Performance Targets:
    - Document search: <500ms (leveraging SemanticSearchUseCase)
    - Context building: <100ms
    - Total processing: <600ms end-to-end

    Examples:
        >>> use_case = RetrieveDocumentUseCase(
        ...     semantic_search_uc=semantic_search_uc,
        ...     context_service=context_service
        ... )
        >>> request = RetrieveDocumentRequest(
        ...     query="What exercises help with lower back pain?",
        ...     max_context_tokens=3000
        ... )
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> response.context
        RetrievalContext(...)
    """

    def __init__(
        self,
        semantic_search_uc: SemanticSearchUseCase,
        context_service: ContextService,
    ) -> None:
        """Initialize retrieve document use case.

        Args:
            semantic_search_uc: Use case for semantic search
            context_service: Domain service for context building
        """
        self._semantic_search_uc = semantic_search_uc
        self._context_service = context_service

    async def execute(self, request: RetrieveDocumentRequest) -> RetrieveDocumentResponse:
        """Execute document retrieval with context building.

        Args:
            request: Document retrieval request

        Returns:
            Retrieval response with context and metrics
        """
        start_time = datetime.now()
        context_id = uuid4()

        # Validate request
        if not request.query or not request.query.strip():
            return RetrieveDocumentResponse(
                success=False,
                context_id=context_id,
                error="Query cannot be empty",
            )

        try:
            # Step 1: Execute semantic search
            search_start = datetime.now()
            search_request = request.to_semantic_search_request()
            search_response = await self._semantic_search_uc.execute(search_request)
            search_time = (datetime.now() - search_start).total_seconds() * 1000

            if not search_response.success:
                return RetrieveDocumentResponse(
                    success=False,
                    context_id=context_id,
                    search_time_ms=search_time,
                    error=f"Search failed: {search_response.error}",
                )

            # Step 2: Convert search results to document references
            document_references = self._convert_to_document_references(
                search_response.results
            )

            # Step 3: Build retrieval context
            context_start = datetime.now()
            retrieval_context = await self._context_service.build_context(
                query=request.query,
                document_references=document_references,
                max_tokens=request.max_context_tokens,
                context_id=context_id,
            )
            context_time = (datetime.now() - context_start).total_seconds() * 1000

            # Step 4: Calculate metrics and build response
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            avg_score = (
                sum(ref.get_similarity_score() for ref in document_references)
                / len(document_references)
                if document_references
                else 0.0
            )

            return RetrieveDocumentResponse(
                success=True,
                context_id=context_id,
                context=retrieval_context,
                total_documents_found=len(search_response.results),
                documents_in_context=len(retrieval_context.document_references),
                documents_truncated=retrieval_context.truncated_count,
                processing_time_ms=total_time,
                search_time_ms=search_time,
                context_time_ms=context_time,
                avg_similarity_score=avg_score,
                context_utilization=retrieval_context.get_token_usage_ratio(),
                metadata={
                    "query_length": len(request.query),
                    "max_context_tokens": request.max_context_tokens,
                    "search_response_time_ms": search_response.processing_time_ms,
                    "embedding_cache_hit": search_response.metadata.get("cache_hit", False),
                },
            )

        except RetrievalError as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            return RetrieveDocumentResponse(
                success=False,
                context_id=context_id,
                processing_time_ms=total_time,
                error=f"Retrieval failed: {str(e)}",
            )

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            return RetrieveDocumentResponse(
                success=False,
                context_id=context_id,
                processing_time_ms=total_time,
                error=f"Unexpected error: {str(e)}",
            )

    def _convert_to_document_references(
        self,
        search_results: List[SearchResult],
    ) -> List[DocumentReference]:
        """Convert search results to document references.

        Args:
            search_results: List of search results

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

            reference = DocumentReference.from_search_result(
                search_result=result,
                relevance_reason=relevance_reason,
                metadata={
                    "rank": result.rank,
                    "similarity_score": result.similarity_score.score,
                    "converted_at": datetime.utcnow().isoformat(),
                },
            )
            document_references.append(reference)

        return document_references

    async def retrieve_additional_documents(
        self,
        context_id: UUID,
        additional_query: str,
        max_additional_documents: int = 5,
    ) -> RetrieveDocumentResponse:
        """Retrieve additional documents for existing context.

        Args:
            context_id: ID of existing context to augment
            additional_query: Additional search query
            max_additional_documents: Maximum additional documents

        Returns:
            Retrieval response with augmented context
        """
        start_time = datetime.now()

        try:
            # Get existing context (this would require context repository)
            # For now, create a new context - in real implementation this would load existing
            existing_context = RetrievalContext(
                context_id=context_id,
                query=additional_query,
                max_tokens=4000,
            )

            # Search for additional documents
            search_request = SemanticSearchRequest(
                query_text=additional_query,
                top_k=max_additional_documents,
                min_similarity=0.0,
            )
            search_response = await self._semantic_search_uc.execute(search_request)

            if not search_response.success:
                return RetrieveDocumentResponse(
                    success=False,
                    context_id=context_id,
                    error=f"Additional search failed: {search_response.error}",
                )

            # Convert and add to context
            additional_references = self._convert_to_document_references(
                search_response.results
            )

            added_count = existing_context.add_references(additional_references)
            total_time = (datetime.now() - start_time).total_seconds() * 1000

            return RetrieveDocumentResponse(
                success=True,
                context_id=context_id,
                context=existing_context,
                total_documents_found=len(search_response.results),
                documents_in_context=len(existing_context.document_references),
                documents_truncated=existing_context.truncated_count,
                processing_time_ms=total_time,
                search_time_ms=search_response.processing_time_ms,
                context_time_ms=0.0,  # Context building time is minimal here
                avg_similarity_score=search_response.avg_similarity_score,
                context_utilization=existing_context.get_token_usage_ratio(),
                metadata={
                    "additional_documents_added": added_count,
                    "augmented_context": True,
                },
            )

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            return RetrieveDocumentResponse(
                success=False,
                context_id=context_id,
                processing_time_ms=total_time,
                error=f"Failed to retrieve additional documents: {str(e)}",
            )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for document retrieval.

        Returns:
            Dictionary with performance statistics
        """
        try:
            # Get semantic search metrics
            search_metrics = await self._semantic_search_uc.get_performance_metrics()

            return {
                "semantic_search_metrics": search_metrics,
                "retrieve_document_status": "operational",
                "context_service_status": "operational",
            }
        except Exception as e:
            return {
                "error": f"Failed to get performance metrics: {str(e)}",
                "semantic_search_metrics": {},
                "retrieve_document_status": "error",
                "context_service_status": "unknown",
            }