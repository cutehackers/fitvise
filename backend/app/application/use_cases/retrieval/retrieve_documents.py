"""Retrieve documents use case for chat context building (Phase 3 refactoring).

This module implements the RetrieveDocumentUseCase that coordinates semantic search
with context building to provide documents for chat conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.domain.entities.document_reference import DocumentReference
from app.domain.entities.retrieval_context import RetrievalContext
from app.domain.exceptions.retrieval_exceptions import (
    RetrievalError,
    ContextBuildingError,
)
from app.domain.services.document_retrieval_service import DocumentRetrievalService


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
        document_retrieval_service: DocumentRetrievalService,
    ) -> None:
        """Initialize retrieve document use case.

        Args:
            document_retrieval_service: Domain service for document retrieval with context building
        """
        self._document_retrieval_service = document_retrieval_service

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
            # Delegate to DocumentRetrievalService for complete workflow
            retrieval_result = await self._document_retrieval_service.retrieve_documents_with_metrics(
                query=request.query.strip(),
                max_context_tokens=request.max_context_tokens,
                max_documents=request.max_documents,
                filters=request.filters,
                min_similarity=request.min_similarity,
                context_id=context_id,
                metadata=request.retrieval_metadata,
            )

            context = retrieval_result["context"]
            metrics = retrieval_result["metrics"]
            search_metrics = retrieval_result["search_metrics"]

            if context is None:
                # Error case - service returned metrics with error
                total_time = (datetime.now() - start_time).total_seconds() * 1000
                return RetrieveDocumentResponse(
                    success=False,
                    context_id=context_id,
                    processing_time_ms=total_time,
                    error=metrics.get("error", "Document retrieval failed"),
                )

            # Build response from service results
            total_time = (datetime.now() - start_time).total_seconds() * 1000

            return RetrieveDocumentResponse(
                success=True,
                context_id=context_id,
                context=context,
                total_documents_found=metrics.get("total_documents_found", 0),
                documents_in_context=metrics.get("documents_in_context", 0),
                documents_truncated=metrics.get("documents_truncated", 0),
                processing_time_ms=total_time,
                search_time_ms=search_metrics.get("embedding_processing_time_ms", 0.0),
                context_time_ms=metrics.get("context_processing_time_ms", 0.0),
                avg_similarity_score=metrics.get("avg_similarity_score", 0.0),
                context_utilization=metrics.get("context_utilization", 0.0),
                metadata={
                    "query_length": len(request.query),
                    "max_context_tokens": request.max_context_tokens,
                    "context_quality_score": metrics.get("context_quality_score", 0.0),
                    "search_service_status": metrics.get("search_service_status", "unknown"),
                    "retrieval_metrics": metrics,
                    "search_metrics": search_metrics,
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
            # Delegate to DocumentRetrievalService for additional documents
            context = await self._document_retrieval_service.retrieve_additional_documents(
                context_id=context_id,
                additional_query=additional_query,
                max_additional_documents=max_additional_documents,
                max_context_tokens=max_context_tokens,
                min_similarity=min_similarity,
            )

            total_time = (datetime.now() - start_time).total_seconds() * 1000

            return RetrieveDocumentResponse(
                success=True,
                context_id=context_id,
                context=context,
                total_documents_found=len(context.document_references) + context.truncated_count,
                documents_in_context=len(context.document_references),
                documents_truncated=context.truncated_count,
                processing_time_ms=total_time,
                search_time_ms=0.0,  # Embedded in DocumentRetrievalService timing
                context_time_ms=0.0,  # Embedded in DocumentRetrievalService timing
                avg_similarity_score=0.0,  # Would need to be calculated from service if needed
                context_utilization=context.get_token_usage_ratio(),
                metadata={
                    "additional_documents_added": len(context.document_references),
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
            # Get document retrieval metrics from the service
            retrieval_metrics = await self._document_retrieval_service.get_retrieval_performance_metrics()

            return {
                "document_retrieval_metrics": retrieval_metrics,
                "retrieve_document_status": "operational",
                "overall_status": retrieval_metrics.get("overall_status", "unknown"),
            }
        except Exception as e:
            return {
                "error": f"Failed to get performance metrics: {str(e)}",
                "document_retrieval_metrics": {},
                "retrieve_document_status": "error",
                "overall_status": "error",
            }