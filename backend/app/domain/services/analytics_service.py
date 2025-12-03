"""Analytics service interface.

This module contains the AnalyticsService abstract interface that defines
the contract for analytics, tracing, and insights operations for the RAG pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class TraceHandle(BaseModel):
    """Handle for managing analytics traces."""
    trace_id: str
    session_id: Optional[str] = None
    parent_trace_id: Optional[str] = None


class SpanHandle(BaseModel):
    """Handle for managing analytics spans within traces."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None


class AnalyticsService(ABC):
    """Abstract service interface for analytics and insights operations.

    This interface defines the contract for analytics services including
    tracing, error tracking, and performance monitoring. Implementations
    should provide detailed observability for RAG pipeline operations.

    Aligns with Phase 3 monitoring infrastructure planned in rag_phase3.md:
    - infrastructure/monitoring/metrics/ for performance metrics
    - infrastructure/monitoring/logging/ for operation logging
    """

    @abstractmethod
    async def trace_rag_pipeline(
        self,
        pipeline_id: str,
        phase: str,
        metadata: Dict[str, Any],
    ) -> Optional[TraceHandle]:
        """Create and start a trace for RAG pipeline operations.

        Args:
            pipeline_id: Unique identifier for the pipeline execution
            phase: Pipeline phase (infrastructure, ingestion, embedding, complete)
            metadata: Additional metadata for the trace (document counts, model info, etc.)

        Returns:
            TraceHandle if tracing is enabled, None otherwise
        """
        pass

    @abstractmethod
    async def create_span(
        self,
        parent_trace: str,
        operation: str,
        metadata: Dict[str, Any],
    ) -> Optional[SpanHandle]:
        """Create a span within an existing trace for specific operations.

        Args:
            parent_trace: Parent trace ID to associate this span with
            operation: Operation being traced (e.g., "chunk_generation", "embedding_creation")
            metadata: Operation-specific metadata

        Returns:
            SpanHandle if tracing is enabled, None otherwise
        """
        pass

    @abstractmethod
    async def update_trace(
        self,
        trace_id: str,
        metadata: Dict[str, Any],
        status: str = "running",
    ) -> None:
        """Update trace metadata and status.

        Args:
            trace_id: Trace identifier to update
            metadata: New metadata to add to the trace
            status: Trace status (running, success, error, timeout)
        """
        pass

    @abstractmethod
    async def update_span(
        self,
        span_id: str,
        metadata: Dict[str, Any],
        status: str = "running",
    ) -> None:
        """Update span metadata and status.

        Args:
            span_id: Span identifier to update
            metadata: New metadata to add to the span
            status: Span status (running, success, error, timeout)
        """
        pass

    @abstractmethod
    async def track_error(
        self,
        error: Exception,
        phase: str,
        context: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """Track and report errors with full context.

        Args:
            error: Exception that occurred
            phase: Pipeline phase where error occurred
            context: Additional context (document_id, operation, parameters, etc.)
            trace_id: Optional trace ID to associate error with
        """
        pass

    @abstractmethod
    async def log_event(
        self,
        event_name: str,
        level: str,
        message: str,
        metadata: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """Log analytics events with structured metadata.

        Args:
            event_name: Name of the event (e.g., "pipeline_started", "batch_completed")
            level: Log level (debug, info, warn, error)
            message: Human-readable message
            metadata: Event metadata
            trace_id: Optional trace ID to associate event with
        """
        pass

    @abstractmethod
    async def track_metrics(
        self,
        metrics: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """Track performance and quality metrics.

        Args:
            metrics: Dictionary of metrics (timing, counts, quality scores, etc.)
            trace_id: Optional trace ID to associate metrics with
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health and connectivity of the analytics service.

        Returns:
            Dictionary with health status, connectivity info, and service details
        """
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Flush any pending analytics data.

        Ensures all traces, spans, and metrics are sent to the analytics backend.
        """
        pass

    def is_enabled(self) -> bool:
        """Check if analytics service is enabled and configured.

        Returns:
            True if service is properly configured and enabled
        """
        return True

    def get_service_name(self) -> str:
        """Get the name of this analytics service.

        Returns:
            Service name for identification and logging
        """
        return "analytics"

    def get_default_metadata(self) -> Dict[str, Any]:
        """Get default metadata to include in all traces.

        Returns:
            Default metadata dictionary
        """
        return {
            "service": self.get_service_name(),
            "version": "1.0.0",
        }