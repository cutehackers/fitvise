"""LangFuse analytics service implementation.

This module provides a LangFuse-based implementation of the AnalyticsService
interface for distributed tracing, error tracking, and performance monitoring
of the RAG pipeline.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, UTC
from typing import Any, Dict, Optional

from langfuse import Langfuse
from langfuse.client import State

from app.core.settings import Settings
from app.domain.services.analytics_service import (
    AnalyticsService,
    SpanHandle,
    TraceHandle,
)

logger = logging.getLogger(__name__)


class LangFuseService(AnalyticsService):
    """LangFuse implementation for analytics and distributed tracing.

    This service integrates with LangFuse to provide comprehensive observability
    for the RAG pipeline, including distributed tracing, error tracking, and
    performance analytics.

    Follows existing ML services patterns:
    - infrastructure/external_services/ml_services/llm_services/ollama_service.py
    - infrastructure/external_services/ml_services/embedding_models/sentence_transformer_service.py
    """

    def __init__(self, settings: Settings):
        """Initialize LangFuse service with configuration.

        Args:
            settings: Application settings containing LangFuse configuration
        """
        self.settings = settings
        self._client: Optional[Langfuse] = None
        self._enabled = False
        self._initialized = False
        self._service_name = "rag-analytics"

    def _initialize_client(self) -> None:
        """Initialize LangFuse client if not already initialized."""
        if self._initialized:
            return

        try:
            # Check if LangFuse is properly configured
            if not self._is_configured():
                logger.warning("LangFuse not configured - analytics disabled")
                self._enabled = False
                self._initialized = True
                return

            # Initialize LangFuse client
            self._client = Langfuse(
                secret_key=self.settings.langfuse_secret_key,
                public_key=self.settings.langfuse_public_key,
                host=self.settings.langfuse_base_url,
                debug=getattr(self.settings, "debug", False),
            )

            self._enabled = self.settings.langfuse_enabled
            self._initialized = True

            if self._enabled:
                logger.info("LangFuse analytics service initialized successfully")
            else:
                logger.info("LangFuse configured but disabled")

        except Exception as e:
            logger.error(f"Failed to initialize LangFuse client: {e}")
            self._enabled = False
            self._initialized = True

    def _is_configured(self) -> bool:
        """Check if LangFuse is properly configured.

        Returns:
            True if required configuration is present
        """
        return (
            self.settings.langfuse_secret_key is not None
            and self.settings.langfuse_public_key is not None
            and self.settings.langfuse_base_url is not None
        )

    def _ensure_enabled(self) -> bool:
        """Ensure service is enabled and initialized.

        Returns:
            True if service is ready to use
        """
        if not self._initialized:
            self._initialize_client()

        return self._enabled and self._client is not None

    async def trace_rag_pipeline(
        self,
        pipeline_id: str,
        phase: str,
        metadata: Dict[str, Any],
    ) -> Optional[TraceHandle]:
        """Create and start a trace for RAG pipeline operations."""
        if not self._ensure_enabled():
            return None

        try:
            # Prepare metadata with defaults
            trace_metadata = {
                **self.get_default_metadata(),
                **metadata,
                "pipeline_id": pipeline_id,
                "phase": phase,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Create trace
            trace = self._client.trace(
                name=f"rag_pipeline_{phase}",
                id=pipeline_id,
                metadata=trace_metadata,
            )

            return TraceHandle(
                trace_id=trace.id,
                session_id=trace_metadata.get("session_id"),
                parent_trace_id=trace_metadata.get("parent_trace_id"),
            )

        except Exception as e:
            logger.error(f"Failed to create trace {pipeline_id}: {e}")
            return None

    async def create_span(
        self,
        parent_trace: str,
        operation: str,
        metadata: Dict[str, Any],
    ) -> Optional[SpanHandle]:
        """Create a span within an existing trace for specific operations."""
        if not self._ensure_enabled():
            return None

        try:
            # Prepare metadata with defaults
            span_metadata = {
                **self.get_default_metadata(),
                **metadata,
                "operation": operation,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Create span
            span = self._client.trace(
                name=operation,
                id=f"{parent_trace}_{operation}_{datetime.now(UTC).timestamp()}",
                parent_trace_id=parent_trace,
                metadata=span_metadata,
            )

            return SpanHandle(
                span_id=span.id,
                trace_id=parent_trace,
                parent_span_id=span_metadata.get("parent_span_id"),
            )

        except Exception as e:
            logger.error(f"Failed to create span {operation} for trace {parent_trace}: {e}")
            return None

    async def update_trace(
        self,
        trace_id: str,
        metadata: Dict[str, Any],
        status: str = "running",
    ) -> None:
        """Update trace metadata and status."""
        if not self._ensure_enabled():
            return

        try:
            # Map status to Langfuse state
            langfuse_state = self._map_status_to_state(status)

            # Update trace
            self._client.trace(
                id=trace_id,
                metadata=metadata,
                status=langfuse_state,
            )

        except Exception as e:
            logger.error(f"Failed to update trace {trace_id}: {e}")

    async def update_span(
        self,
        span_id: str,
        metadata: Dict[str, Any],
        status: str = "running",
    ) -> None:
        """Update span metadata and status."""
        if not self._ensure_enabled():
            return

        try:
            # Map status to Langfuse state
            langfuse_state = self._map_status_to_state(status)

            # Update span (spans are traces in Langfuse)
            self._client.trace(
                id=span_id,
                metadata=metadata,
                status=langfuse_state,
            )

        except Exception as e:
            logger.error(f"Failed to update span {span_id}: {e}")

    async def track_error(
        self,
        error: Exception,
        phase: str,
        context: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """Track and report errors with full context."""
        if not self._ensure_enabled():
            return

        try:
            # Prepare error metadata
            error_metadata = {
                **self.get_default_metadata(),
                **context,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_traceback": traceback.format_exc(),
                "phase": phase,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Create error trace or update existing trace
            if trace_id:
                self._client.trace(
                    id=trace_id,
                    metadata=error_metadata,
                    status=State.ERROR,
                )
            else:
                self._client.trace(
                    name=f"error_{phase}",
                    metadata=error_metadata,
                    status=State.ERROR,
                )

        except Exception as e:
            logger.error(f"Failed to track error in phase {phase}: {e}")

    async def log_event(
        self,
        event_name: str,
        level: str,
        message: str,
        metadata: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """Log analytics events with structured metadata."""
        if not self._ensure_enabled():
            return

        try:
            # Prepare event metadata
            event_metadata = {
                **self.get_default_metadata(),
                **metadata,
                "event_name": event_name,
                "level": level,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Create event trace or update existing trace
            if trace_id:
                self._client.trace(
                    id=trace_id,
                    metadata=event_metadata,
                )
            else:
                self._client.trace(
                    name=f"event_{event_name}",
                    metadata=event_metadata,
                )

        except Exception as e:
            logger.error(f"Failed to log event {event_name}: {e}")

    async def track_metrics(
        self,
        metrics: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """Track performance and quality metrics."""
        if not self._ensure_enabled():
            return

        try:
            # Prepare metrics metadata
            metrics_metadata = {
                **self.get_default_metadata(),
                **metrics,
                "metrics_timestamp": datetime.now(UTC).isoformat(),
            }

            # Create metrics trace or update existing trace
            if trace_id:
                self._client.trace(
                    id=trace_id,
                    metadata=metrics_metadata,
                )
            else:
                self._client.trace(
                    name="metrics",
                    metadata=metrics_metadata,
                )

        except Exception as e:
            logger.error(f"Failed to track metrics: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check the health and connectivity of the analytics service."""
        health_status = {
            "service": self.get_service_name(),
            "enabled": self._enabled,
            "initialized": self._initialized,
            "configured": self._is_configured(),
            "host": self.settings.langfuse_base_url if self._is_configured() else None,
            "status": "healthy" if self._ensure_enabled() else "disabled",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Test connectivity if enabled
        if self._ensure_enabled():
            try:
                # Try to create a simple trace to test connectivity
                test_trace = self._client.trace(
                    name="health_check",
                    metadata={"test": True},
                )
                health_status["connectivity"] = "ok"
                health_status["last_test"] = datetime.now(UTC).isoformat()
            except Exception as e:
                health_status["connectivity"] = "failed"
                health_status["error"] = str(e)

        return health_status

    async def flush(self) -> None:
        """Flush any pending analytics data."""
        if self._client:
            try:
                self._client.flush()
                logger.debug("LangFuse analytics data flushed")
            except Exception as e:
                logger.error(f"Failed to flush analytics data: {e}")

    def is_enabled(self) -> bool:
        """Check if analytics service is enabled and configured."""
        return self._ensure_enabled()

    def get_service_name(self) -> str:
        """Get the name of this analytics service."""
        return self._service_name

    def get_default_metadata(self) -> Dict[str, Any]:
        """Get default metadata to include in all traces."""
        return {
            "service": self.get_service_name(),
            "version": "1.0.0",
            "environment": getattr(self.settings, "environment", "development"),
        }

    def _map_status_to_state(self, status: str) -> State:
        """Map status string to Langfuse State enum.

        Args:
            status: Status string (running, success, error, timeout)

        Returns:
            Langfuse State enum value
        """
        status_mapping = {
            "running": State.RUNNING,
            "success": State.COMPLETED,
            "completed": State.COMPLETED,
            "error": State.ERROR,
            "failed": State.ERROR,
            "timeout": State.TIMEOUT,
            "cancelled": State.CANCELLED,
        }

        return status_mapping.get(status.lower(), State.RUNNING)