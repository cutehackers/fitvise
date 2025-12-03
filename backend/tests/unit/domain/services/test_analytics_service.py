"""Tests for the AnalyticsService domain interface.

This module tests the AnalyticsService abstract interface and its concrete
implementations, ensuring they properly handle tracing, error tracking, and
performance monitoring for the RAG pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from app.domain.services.analytics_service import AnalyticsService, TraceHandle, SpanHandle


class MockAnalyticsService(AnalyticsService):
    """Mock implementation of AnalyticsService for testing."""

    def __init__(self):
        self.traces = {}
        self.spans = {}
        self.errors = []
        self.metrics = []
        self.events = []
        self.enabled = True

    async def trace_rag_pipeline(
        self,
        pipeline_id: str,
        phase: str,
        metadata: Dict[str, Any],
    ) -> Optional[TraceHandle]:
        if not self.enabled:
            return None
        self.traces[pipeline_id] = {
            "phase": phase,
            "metadata": metadata,
            "status": "running",
        }
        return TraceHandle(trace_id=pipeline_id)

    async def create_span(
        self,
        parent_trace: str,
        operation: str,
        metadata: Dict[str, Any],
    ) -> Optional[SpanHandle]:
        if not self.enabled:
            return None
        span_id = f"{parent_trace}_{operation}"
        self.spans[span_id] = {
            "parent_trace": parent_trace,
            "operation": operation,
            "metadata": metadata,
            "status": "running",
        }
        return SpanHandle(span_id=span_id, trace_id=parent_trace)

    async def update_trace(
        self,
        trace_id: str,
        metadata: Dict[str, Any],
        status: str = "running",
    ) -> None:
        if trace_id in self.traces:
            self.traces[trace_id]["metadata"].update(metadata)
            self.traces[trace_id]["status"] = status

    async def update_span(
        self,
        span_id: str,
        metadata: Dict[str, Any],
        status: str = "running",
    ) -> None:
        if span_id in self.spans:
            self.spans[span_id]["metadata"].update(metadata)
            self.spans[span_id]["status"] = status

    async def track_error(
        self,
        error: Exception,
        phase: str,
        context: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        self.errors.append({
            "error": str(error),
            "phase": phase,
            "context": context,
            "trace_id": trace_id,
            "error_type": type(error).__name__,
        })

    async def log_event(
        self,
        event_name: str,
        level: str,
        message: str,
        metadata: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        self.events.append({
            "event_name": event_name,
            "level": level,
            "message": message,
            "metadata": metadata,
            "trace_id": trace_id,
        })

    async def track_metrics(
        self,
        metrics: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        self.metrics.append({
            "metrics": metrics,
            "trace_id": trace_id,
        })

    async def health_check(self) -> Dict[str, Any]:
        return {
            "service": "mock_analytics",
            "enabled": self.enabled,
            "status": "healthy",
        }

    async def flush(self) -> None:
        pass

    def is_enabled(self) -> bool:
        return self.enabled

    def get_service_name(self) -> str:
        return "mock_analytics"


class TestAnalyticsService:
    """Test cases for the AnalyticsService domain interface."""

    @pytest.fixture
    def analytics_service(self):
        """Create a mock analytics service for testing."""
        return MockAnalyticsService()

    @pytest.mark.asyncio
    async def test_trace_rag_pipeline_success(self, analytics_service):
        """Test successful RAG pipeline tracing."""
        pipeline_id = "test-pipeline-123"
        phase = "ingestion"
        metadata = {"document_count": 10, "batch_size": 5}

        trace_handle = await analytics_service.trace_rag_pipeline(
            pipeline_id, phase, metadata
        )

        assert trace_handle is not None
        assert trace_handle.trace_id == pipeline_id
        assert pipeline_id in analytics_service.traces
        assert analytics_service.traces[pipeline_id]["phase"] == phase
        assert analytics_service.traces[pipeline_id]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_trace_rag_pipeline_disabled(self, analytics_service):
        """Test tracing when analytics service is disabled."""
        analytics_service.enabled = False

        trace_handle = await analytics_service.trace_rag_pipeline(
            "pipeline-123", "ingestion", {}
        )

        assert trace_handle is None

    @pytest.mark.asyncio
    async def test_create_span_success(self, analytics_service):
        """Test successful span creation."""
        parent_trace = "parent-trace-123"
        operation = "chunk_generation"
        metadata = {"chunk_count": 5}

        span_handle = await analytics_service.create_span(
            parent_trace, operation, metadata
        )

        assert span_handle is not None
        assert span_handle.span_id == f"{parent_trace}_{operation}"
        assert span_handle.trace_id == parent_trace
        assert f"{parent_trace}_{operation}" in analytics_service.spans
        assert analytics_service.spans[span_handle.span_id]["operation"] == operation
        assert analytics_service.spans[span_handle.span_id]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_update_trace(self, analytics_service):
        """Test updating trace metadata and status."""
        pipeline_id = "trace-123"
        initial_metadata = {"initial": True}
        update_metadata = {"updated": True, "duration_ms": 1000}
        final_status = "success"

        # Create trace first
        await analytics_service.trace_rag_pipeline(
            pipeline_id, "ingestion", initial_metadata
        )

        # Update trace
        await analytics_service.update_trace(
            pipeline_id, update_metadata, final_status
        )

        # Verify updates
        trace = analytics_service.traces[pipeline_id]
        assert "updated" in trace["metadata"]
        assert trace["status"] == final_status
        assert "initial" in trace["metadata"]  # Original metadata preserved

    @pytest.mark.asyncio
    async def test_track_error(self, analytics_service):
        """Test error tracking with context."""
        error = ValueError("Test error")
        phase = "embedding"
        context = {"document_id": "doc-123", "operation": "generate_embedding"}
        trace_id = "trace-123"

        await analytics_service.track_error(error, phase, context, trace_id)

        # Verify error was tracked
        assert len(analytics_service.errors) == 1
        tracked_error = analytics_service.errors[0]
        assert tracked_error["error"] == "Test error"
        assert tracked_error["phase"] == phase
        assert tracked_error["context"] == context
        assert tracked_error["trace_id"] == trace_id
        assert tracked_error["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_log_event(self, analytics_service):
        """Test event logging."""
        event_name = "pipeline_started"
        level = "info"
        message = "RAG pipeline started successfully"
        metadata = {"total_documents": 50}
        trace_id = "trace-123"

        await analytics_service.log_event(
            event_name, level, message, metadata, trace_id
        )

        # Verify event was logged
        assert len(analytics_service.events) == 1
        event = analytics_service.events[0]
        assert event["event_name"] == event_name
        assert event["level"] == level
        assert event["message"] == message
        assert event["metadata"] == metadata
        assert event["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_track_metrics(self, analytics_service):
        """Test metrics tracking."""
        metrics = {
            "total_documents": 100,
            "processing_time_ms": 5000,
            "success_rate": 0.95,
        }
        trace_id = "trace-123"

        await analytics_service.track_metrics(metrics, trace_id)

        # Verify metrics were tracked
        assert len(analytics_service.metrics) == 1
        tracked_metrics = analytics_service.metrics[0]
        assert tracked_metrics["metrics"] == metrics
        assert tracked_metrics["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_health_check(self, analytics_service):
        """Test health check functionality."""
        health = await analytics_service.health_check()

        assert "service" in health
        assert "enabled" in health
        assert "status" in health
        assert health["service"] == "mock_analytics"
        assert health["enabled"] == analytics_service.enabled

    @pytest.mark.asyncio
    async def test_is_enabled(self, analytics_service):
        """Test enabled status check."""
        # Default state
        assert analytics_service.is_enabled() is True

        # Disabled state
        analytics_service.enabled = False
        assert analytics_service.is_enabled() is False

    def test_get_service_name(self, analytics_service):
        """Test service name retrieval."""
        assert analytics_service.get_service_name() == "mock_analytics"

    def test_get_default_metadata(self, analytics_service):
        """Test default metadata generation."""
        default_metadata = analytics_service.get_default_metadata()

        assert "service" in default_metadata
        assert "version" in default_metadata
        assert default_metadata["service"] == "mock_analytics"

    @pytest.mark.asyncio
    async def test_complete_workflow(self, analytics_service):
        """Test complete analytics workflow with tracing and updates."""
        pipeline_id = "workflow-test"

        # Start tracing
        trace_handle = await analytics_service.trace_rag_pipeline(
            pipeline_id, "ingestion", {"document_count": 5}
        )
        assert trace_handle is not None

        # Create a span
        span_handle = await analytics_service.create_span(
            pipeline_id, "chunking", {"chunks_created": 25}
        )
        assert span_handle is not None

        # Update trace
        await analytics_service.update_trace(
            pipeline_id, {"processing_complete": True}, "success"
        )

        # Update span
        await analytics_service.update_span(
            span_handle.span_id, {"chunks_processed": 25}, "success"
        )

        # Log an event
        await analytics_service.log_event(
            "pipeline_completed",
            "info",
            "Pipeline completed successfully",
            {"final_document_count": 5},
            pipeline_id,
        )

        # Track metrics
        await analytics_service.track_metrics(
            {"total_duration_ms": 5000, "documents_processed": 5},
            pipeline_id,
        )

        # Verify all data was collected
        assert pipeline_id in analytics_service.traces
        assert analytics_service.traces[pipeline_id]["status"] == "success"
        assert len(analytics_service.events) == 1
        assert len(analytics_service.metrics) == 1
        assert len(analytics_service.spans) == 1

    @pytest.mark.asyncio
    async def test_error_handling_in_trace_creation(self, analytics_service):
        """Test graceful handling of errors during trace creation."""
        # Mock the trace method to raise an exception
        async def failing_trace(*args, **kwargs):
            raise Exception("Trace creation failed")

        analytics_service.trace_rag_pipeline = failing_trace

        # Should not raise exception
        trace_handle = await analytics_service.trace_rag_pipeline(
            "pipeline-123", "ingestion", {}
        )

        # Should return None due to error
        assert trace_handle is None

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test analytics service behavior when not properly configured."""
        # This will test the actual LangFuseService graceful degradation
        from app.infrastructure.external_services.analytics.langfuse_service import LangFuseService
        from app.core.settings import Settings

        # Create settings with missing LangFuse configuration
        settings = Settings()
        settings.langfuse_enabled = False
        settings.langfuse_secret_key = None
        settings.langfuse_public_key = None

        # Create service with disabled configuration
        langfuse_service = LangFuseService(settings)

        # Service should be disabled but not crash
        assert not langfuse_service.is_enabled()

        # All operations should gracefully return None
        trace = await langfuse_service.trace_rag_pipeline(
            "test-pipeline", "test-phase", {}
        )
        assert trace is None

        error_tracked = await langfuse_service.track_error(
            Exception("Test"), "test-phase", {}
        )
        # Should not raise exception

        health = await langfuse_service.health_check()
        assert "enabled" in health
        assert health["enabled"] is False