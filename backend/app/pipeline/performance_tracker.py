"""Reusable performance tracking utilities for async operations."""

import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
from dataclasses import dataclass


@dataclass
class PerformanceResult:
    """Container for performance measurement results."""
    duration: float
    result: Any
    start_time: float
    end_time: float


class PerformanceTracker:
    """Tracks execution times for async operations with aggregated statistics."""

    def __init__(self):
        self._timings: Dict[str, list[float]] = {}

    def record_timing(self, operation: str, duration: float) -> None:
        """Record a timing for a specific operation."""
        if operation not in self._timings:
            self._timings[operation] = []
        self._timings[operation].append(duration)

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific operation."""
        if operation not in self._timings or not self._timings[operation]:
            return None

        timings = self._timings[operation]
        return {
            "count": len(timings),
            "total": sum(timings),
            "average": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded operations."""
        return {op: stats for op, stats in
                [(op, self.get_stats(op)) for op in self._timings] if stats}

    def clear(self, operation: Optional[str] = None) -> None:
        """Clear timings for a specific operation or all."""
        if operation:
            self._timings.pop(operation, None)
        else:
            self._timings.clear()

    @asynccontextmanager
    async def with_tracker(self, operation: str):
        """Context manager for timing async operations.

        Args:
            operation: Name of the operation being timed

        Yields:
            None

        Example:
            async with tracker.with_tracker("extract_content"):
                result = await some_async_function()
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record_timing(operation, duration)

    async def measure(self, operation: str, async_func: Callable, *args, **kwargs) -> PerformanceResult:
        """Execute and time an async operation, recording the result.

        Args:
            operation: Name of the operation
            async_func: Async function to execute
            *args: Arguments to pass to async_func
            **kwargs: Keyword arguments to pass to async_func

        Returns:
            PerformanceResult containing duration and function result

        Example:
            result = await tracker.measure("extract_content", extract_content, doc)
        """
        start_time = time.perf_counter()
        result = await async_func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time

        self.record_timing(operation, duration)

        return PerformanceResult(
            duration=duration,
            result=result,
            start_time=start_time,
            end_time=end_time
        )


class AnalyticsEnhancedPerformanceTracker(PerformanceTracker):
    """Enhanced performance tracker with analytics integration.

    This class extends the base PerformanceTracker to provide distributed tracing
    and analytics capabilities through the AnalyticsService interface. It maintains
    backward compatibility with existing performance tracking while adding
    comprehensive observability.

    Complements planned Phase 3 monitoring:
    - infrastructure/monitoring/metrics/generation_metrics.py
    - infrastructure/monitoring/metrics/conversation_metrics.py
    - infrastructure/monitoring/logging/generation_logger.py
    """

    def __init__(self, analytics_service=None):
        """Initialize enhanced performance tracker.

        Args:
            analytics_service: Optional AnalyticsService instance for distributed tracing
        """
        super().__init__()
        self.analytics_service = analytics_service
        self._current_trace = None
        self._current_spans = []
        self._pipeline_id = None
        self._logger = logging.getLogger(__name__)

    def set_pipeline_context(self, pipeline_id: str, phase: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set the current pipeline context for analytics tracing.

        Args:
            pipeline_id: Unique identifier for the pipeline execution
            phase: Current pipeline phase (infrastructure, ingestion, embedding, complete)
            metadata: Additional metadata for tracing
        """
        self._pipeline_id = pipeline_id
        if self.analytics_service and metadata:
            # Create trace context for the pipeline
            try:
                import asyncio
                if asyncio.get_event_loop().is_running():
                    # We're in an async context, but can't await here
                    # Store the context for later when async methods are called
                    self._pending_trace_context = {
                        "pipeline_id": pipeline_id,
                        "phase": phase,
                        "metadata": metadata
                    }
                else:
                    # Create trace synchronously if possible
                    self._current_trace = asyncio.run(
                        self.analytics_service.trace_rag_pipeline(pipeline_id, phase, metadata or {})
                    )
            except Exception as e:
                self._logger.warning(f"Failed to set pipeline context: {e}")

    async def start_trace(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a new analytics trace for an operation.

        Args:
            operation_name: Name of the operation being traced
            metadata: Additional metadata for the trace

        Returns:
            Trace ID if trace was created, None otherwise
        """
        if not self.analytics_service:
            return None

        try:
            trace_id = f"{self._pipeline_id}_{operation_name}_{time.time()}" if self._pipeline_id else f"{operation_name}_{time.time()}"

            trace_metadata = {
                "operation": operation_name,
                **(metadata or {}),
                "pipeline_id": self._pipeline_id,
            }

            trace_handle = await self.analytics_service.trace_rag_pipeline(
                pipeline_id=trace_id,
                phase="operation",
                metadata=trace_metadata
            )

            if trace_handle:
                self._current_spans.append({
                    "span_id": trace_handle.trace_id,
                    "operation": operation_name,
                    "start_time": time.time()
                })
                return trace_handle.trace_id

        except Exception as e:
            self._logger.error(f"Failed to start trace for {operation_name}: {e}")

        return None

    async def complete_trace(self, trace_id: str, metadata: Optional[Dict[str, Any]] = None, status: str = "success") -> None:
        """Complete an analytics trace with final metadata.

        Args:
            trace_id: ID of the trace to complete
            metadata: Final metadata to add to the trace
            status: Final status of the trace (success, error, timeout)
        """
        if not self.analytics_service or not trace_id:
            return

        try:
            # Find and update the span
            span_info = next((s for s in self._current_spans if s["span_id"] == trace_id), None)
            if span_info:
                duration = time.time() - span_info["start_time"]

                final_metadata = {
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                    **(metadata or {}),
                    "status": status,
                }

                await self.analytics_service.update_trace(trace_id, final_metadata, status)

                # Remove from active spans
                self._current_spans = [s for s in self._current_spans if s["span_id"] != trace_id]

        except Exception as e:
            self._logger.error(f"Failed to complete trace {trace_id}: {e}")

    async def track_stage_with_analytics(self, stage_name: str, func, *args, **kwargs) -> Any:
        """Track a stage with both performance timing and analytics tracing.

        Args:
            stage_name: Name of the stage being tracked
            func: Async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function execution
        """
        # Start analytics trace
        trace_id = await self.start_trace(stage_name)

        try:
            # Execute with performance tracking
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()

            duration = end_time - start_time

            # Record performance timing
            self.record_timing(stage_name, duration)

            # Complete analytics trace
            await self.complete_trace(trace_id, {
                "success": True,
                "duration_ms": duration * 1000
            }, "success")

            return result

        except Exception as e:
            # Record error timing
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.record_timing(f"{stage_name}_error", duration)

            # Complete analytics trace with error
            await self.complete_trace(trace_id, {
                "success": False,
                "error": str(e),
                "duration_ms": duration * 1000
            }, "error")

            # Track error with analytics service
            if self.analytics_service:
                await self.analytics_service.track_error(
                    error=e,
                    phase="pipeline_stage",
                    context={
                        "stage": stage_name,
                        "duration_ms": duration * 1000,
                        "pipeline_id": self._pipeline_id,
                    },
                    trace_id=trace_id
                )

            raise

    @asynccontextmanager
    async def with_analytics_tracker(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for timing async operations with analytics tracing.

        Args:
            operation: Name of the operation being timed
            metadata: Additional metadata for analytics tracing

        Yields:
            None

        Example:
            async with tracker.with_analytics_tracker("extract_content"):
                result = await some_async_function()
        """
        # Start analytics trace
        trace_id = await self.start_trace(operation, metadata)
        start_time = time.perf_counter()

        try:
            yield
        except Exception as e:
            # Handle error
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Record error timing
            self.record_timing(f"{operation}_error", duration)

            # Complete analytics trace with error
            await self.complete_trace(trace_id, {
                "success": False,
                "error": str(e),
                "duration_ms": duration * 1000
            }, "error")

            # Track error with analytics service
            if self.analytics_service:
                await self.analytics_service.track_error(
                    error=e,
                    phase="pipeline_stage",
                    context={
                        "stage": operation,
                        "duration_ms": duration * 1000,
                        "pipeline_id": self._pipeline_id,
                    },
                    trace_id=trace_id
                )

            raise
        else:
            # Success path
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Record performance timing
            self.record_timing(operation, duration)

            # Complete analytics trace
            await self.complete_trace(trace_id, {
                "success": True,
                "duration_ms": duration * 1000
            }, "success")

    async def measure_with_analytics(self, operation: str, async_func: Callable, *args, **kwargs) -> PerformanceResult:
        """Execute and time an async operation with analytics tracking.

        Args:
            operation: Name of the operation
            async_func: Async function to execute
            *args: Arguments to pass to async_func
            **kwargs: Keyword arguments to pass to async_func

        Returns:
            PerformanceResult containing duration and function result
        """
        # Start analytics trace
        trace_id = await self.start_trace(operation)

        try:
            # Execute with performance tracking
            start_time = time.perf_counter()
            result = await async_func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Record performance timing
            self.record_timing(operation, duration)

            # Complete analytics trace
            await self.complete_trace(trace_id, {
                "success": True,
                "duration_ms": duration * 1000
            }, "success")

            return PerformanceResult(
                duration=duration,
                result=result,
                start_time=start_time,
                end_time=end_time
            )

        except Exception as e:
            # Handle error
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Record error timing
            self.record_timing(f"{operation}_error", duration)

            # Complete analytics trace with error
            await self.complete_trace(trace_id, {
                "success": False,
                "error": str(e),
                "duration_ms": duration * 1000
            }, "error")

            # Track error with analytics service
            if self.analytics_service:
                await self.analytics_service.track_error(
                    error=e,
                    phase="pipeline_stage",
                    context={
                        "stage": operation,
                        "duration_ms": duration * 1000,
                        "pipeline_id": self._pipeline_id,
                    },
                    trace_id=trace_id
                )

            raise

    async def track_pipeline_metrics(self, metrics: Dict[str, Any]) -> None:
        """Track pipeline-level metrics with analytics service.

        Args:
            metrics: Dictionary of pipeline metrics to track
        """
        if self.analytics_service:
            try:
                # Add performance stats to metrics
                all_stats = self.get_all_stats()
                if all_stats:
                    metrics["performance_stats"] = all_stats

                await self.analytics_service.track_metrics(metrics)

            except Exception as e:
                self._logger.error(f"Failed to track pipeline metrics: {e}")

    async def finalize_pipeline(self, final_status: str = "success", final_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Finalize the current pipeline trace with final status and metadata.

        Args:
            final_status: Final status of the pipeline (success, error, timeout)
            final_metadata: Additional metadata to add to the pipeline trace
        """
        if not self.analytics_service or not self._pipeline_id:
            return

        try:
            # Prepare final metadata
            final_data = {
                "pipeline_completed": True,
                "final_status": final_status,
                "total_stages_tracked": len(self._current_spans),
                "performance_stats": self.get_all_stats(),
                **(final_metadata or {}),
            }

            # Complete any remaining spans
            for span_info in self._current_spans:
                duration = time.time() - span_info["start_time"]
                await self.analytics_service.update_trace(
                    span_info["span_id"],
                    {
                        "auto_completed": True,
                        "duration_ms": duration * 1000,
                        "status": "incomplete"
                    },
                    "cancelled"
                )

            # Update main pipeline trace
            await self.analytics_service.update_trace(
                self._pipeline_id,
                final_data,
                final_status
            )

            # Flush analytics data
            await self.analytics_service.flush()

        except Exception as e:
            self._logger.error(f"Failed to finalize pipeline trace: {e}")

        # Reset state
        self._current_trace = None
        self._current_spans = []
        self._pipeline_id = None


def timed_async(operation: str, tracker: Optional[PerformanceTracker] = None):
    """Decorator for timing async functions.

    Args:
        operation: Name of the operation being timed
        tracker: Optional performance tracker to record results. If None, uses global.

    Example:
        @timed_async("extract_content")
        async def extract_content(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> PerformanceResult:
            if tracker is None:
                # Use a global tracker if none provided
                global_tracker = _get_global_tracker()
                return await global_tracker.measure(operation, func, *args, **kwargs)
            else:
                return await tracker.measure(operation, func, *args, **kwargs)
        return wrapper
    return decorator


# Global tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def _get_global_tracker() -> PerformanceTracker:
    """Get or create the global performance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


# Convenience functions that use the global tracker
async def measure_performance(operation: str, async_func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Execute an async function and return result with timing using global tracker.

    Args:
        operation: Name of the operation
        async_func: Async function to execute
        *args: Arguments to pass to async_func
        **kwargs: Keyword arguments to pass to async_func

    Returns:
        Tuple of (result, duration)

    Example:
        result, duration = await measure_performance("extract_content", extract_content, doc)
    """
    result_obj = await _get_global_tracker().measure(operation, async_func, *args, **kwargs)
    return result_obj.result, result_obj.duration


@asynccontextmanager
async def with_timer(operation: str):
    """Context manager for timing operations using global tracker.

    Args:
        operation: Name of the operation being timed

    Example:
        async with with_timer("extract_content"):
            result = await some_async_function()
    """
    async with _get_global_tracker().with_tracker(operation):
        yield