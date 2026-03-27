"""Thread pool management for embedding service operations.

Provides dedicated thread pools for different types of embedding operations
with optimal sizing and resource isolation.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
from threading import Lock
from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingThreadPoolManager:
    """Manages dedicated thread pools for embedding operations.

    Provides separate pools for different operation types:
    - embedding_pool: For text embedding operations
    - model_pool: For heavy model operations (loading, initialization)
    - io_pool: For I/O bound operations (cache, storage)
    """

    def __init__(self) -> None:
        """Initialize thread pool manager with optimal sizing."""
        self._embedding_pool_size = self._calculate_embedding_pool_size()
        self._model_pool_size = self._calculate_model_pool_size()
        self._io_pool_size = self._calculate_io_pool_size()

        # Create dedicated thread pools
        self._embedding_pool = ThreadPoolExecutor(
            max_workers=self._embedding_pool_size,
            thread_name_prefix="embedding"
        )

        self._model_pool = ThreadPoolExecutor(
            max_workers=self._model_pool_size,
            thread_name_prefix="model_ops"
        )

        self._io_pool = ThreadPoolExecutor(
            max_workers=self._io_pool_size,
            thread_name_prefix="io_ops"
        )

        # Priority queue for real-time queries
        self._priority_queue = PriorityQueue()
        self._queue_lock = Lock()

        self._stats = {
            "embedding_tasks": 0,
            "model_tasks": 0,
            "io_tasks": 0,
            "priority_tasks": 0,
        }

    def _calculate_embedding_pool_size(self) -> int:
        """Calculate optimal size for embedding thread pool.

        Embedding operations are CPU-bound with some I/O, so we use
        (cpu_count * 2) + 1 for optimal throughput.

        Returns:
            Optimal thread pool size for embedding operations
        """
        cpu_count = os.cpu_count() or 4
        return (cpu_count * 2) + 1

    def _calculate_model_pool_size(self) -> int:
        """Calculate optimal size for model operations pool.

        Model operations are heavy CPU tasks, so we use fewer threads
        to avoid context switching overhead.

        Returns:
            Optimal thread pool size for model operations
        """
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count // 2)

    def _calculate_io_pool_size(self) -> int:
        """Calculate optimal size for I/O operations pool.

        I/O operations benefit from more threads due to blocking nature.

        Returns:
            Optimal thread pool size for I/O operations
        """
        cpu_count = os.cpu_count() or 4
        return cpu_count * 4

    @property
    def embedding_pool(self) -> ThreadPoolExecutor:
        """Get thread pool for embedding operations."""
        return self._embedding_pool

    @property
    def model_pool(self) -> ThreadPoolExecutor:
        """Get thread pool for model operations."""
        return self._model_pool

    @property
    def io_pool(self) -> ThreadPoolExecutor:
        """Get thread pool for I/O operations."""
        return self._io_pool

    def submit_priority_task(self, fn, *args, **kwargs) -> Any:
        """Submit high-priority task for real-time processing.

        Args:
            fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Future for the task
        """
        with self._queue_lock:
            self._stats["priority_tasks"] += 1
            # Use embedding pool for priority tasks with high priority value
            return self._embedding_pool.submit(fn, *args, **kwargs)

    def submit_embedding_task(self, fn, *args, **kwargs) -> Any:
        """Submit embedding task to appropriate pool.

        Args:
            fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Future for the task
        """
        self._stats["embedding_tasks"] += 1
        return self._embedding_pool.submit(fn, *args, **kwargs)

    def submit_model_task(self, fn, *args, **kwargs) -> Any:
        """Submit model operation task to model pool.

        Args:
            fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Future for the task
        """
        self._stats["model_tasks"] += 1
        return self._model_pool.submit(fn, *args, **kwargs)

    def submit_io_task(self, fn, *args, **kwargs) -> Any:
        """Submit I/O task to I/O pool.

        Args:
            fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Future for the task
        """
        self._stats["io_tasks"] += 1
        return self._io_pool.submit(fn, *args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            **self._stats,
            "embedding_pool_size": self._embedding_pool_size,
            "model_pool_size": self._model_pool_size,
            "io_pool_size": self._io_pool_size,
            "embedding_pool_active": len(self._embedding_pool._threads),
            "model_pool_active": len(self._model_pool._threads),
            "io_pool_active": len(self._io_pool._threads),
        }

    async def shutdown(self, wait: bool = True) -> None:
        """Shutdown all thread pools gracefully.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down thread pools...")

        self._embedding_pool.shutdown(wait=wait)
        self._model_pool.shutdown(wait=wait)
        self._io_pool.shutdown(wait=wait)

        logger.info("All thread pools shutdown complete")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        # Don't call async shutdown from __del__ as it creates warnings
        try:
            self._embedding_pool.shutdown(wait=False)
            self._model_pool.shutdown(wait=False)
            self._io_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore errors during cleanup


# Global thread pool manager instance
_thread_pool_manager: Optional[EmbeddingThreadPoolManager] = None


def get_thread_pool_manager() -> EmbeddingThreadPoolManager:
    """Get global thread pool manager instance.

    Returns:
        Thread pool manager singleton instance
    """
    global _thread_pool_manager
    if _thread_pool_manager is None:
        _thread_pool_manager = EmbeddingThreadPoolManager()
    return _thread_pool_manager


async def shutdown_thread_pools() -> None:
    """Shutdown global thread pool manager."""
    global _thread_pool_manager
    if _thread_pool_manager is not None:
        await _thread_pool_manager.shutdown()
        _thread_pool_manager = None