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