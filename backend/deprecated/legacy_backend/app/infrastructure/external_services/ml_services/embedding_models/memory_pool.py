"""Memory pool management for embedding vectors.

Provides efficient memory allocation and management for high-performance
embedding operations with zero-copy optimizations.
"""

from __future__ import annotations

import gc
import threading
from collections import deque
from typing import Any, Deque, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)


class MemoryBlock:
    """Represents a memory block for embedding storage."""

    def __init__(self, size: int, dimension: int) -> None:
        """Initialize memory block.

        Args:
            size: Number of vectors this block can hold
            dimension: Dimension of each vector
        """
        self.size = size
        self.dimension = dimension
        self.data = np.zeros((size, dimension), dtype=np.float32)
        self.allocated = 0
        self.free_indices = deque(range(size))
        self.lock = threading.Lock()

    def allocate(self) -> Optional[int]:
        """Allocate a slot in this memory block.

        Returns:
            Index of allocated slot, or None if block is full
        """
        with self.lock:
            if not self.free_indices:
                return None

            index = self.free_indices.popleft()
            self.allocated += 1
            return index

    def deallocate(self, index: int) -> None:
        """Deallocate a slot in this memory block.

        Args:
            index: Index to deallocate
        """
        with self.lock:
            if 0 <= index < self.size:
                self.free_indices.append(index)
                self.allocated = max(0, self.allocated - 1)

    def get_vector(self, index: int) -> npt.NDArray[np.float32]:
        """Get vector at specific index.

        Args:
            index: Vector index

        Returns:
            Vector array view (zero-copy)
        """
        return self.data[index]

    def is_full(self) -> bool:
        """Check if block is fully allocated.

        Returns:
            True if block is full
        """
        return len(self.free_indices) == 0

    def utilization(self) -> float:
        """Get block utilization ratio.

        Returns:
            Utilization ratio (0.0 to 1.0)
        """
        return self.allocated / self.size if self.size > 0 else 0.0


class EmbeddingMemoryPool:
    """Memory pool for efficient embedding vector allocation.

    Provides pre-allocated memory blocks for different vector dimensions
    to reduce allocation overhead and improve cache locality.
    """

    def __init__(self, initial_block_size: int = 1000, max_blocks: int = 10) -> None:
        """Initialize memory pool.

        Args:
            initial_block_size: Initial size of memory blocks
            max_blocks: Maximum number of blocks per dimension
        """
        self.initial_block_size = initial_block_size
        self.max_blocks = max_blocks
        self.blocks: Dict[int, List[MemoryBlock]] = {}
        self.block_lock = threading.Lock()
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "active_blocks": 0,
            "memory_usage_mb": 0.0,
        }

    def get_block(self, dimension: int) -> Optional[MemoryBlock]:
        """Get available memory block for specific dimension.

        Args:
            dimension: Vector dimension

        Returns:
            Available memory block, or None if none available
        """
        with self.block_lock:
            if dimension not in self.blocks:
                self.blocks[dimension] = []

            # Find block with available space
            for block in self.blocks[dimension]:
                if not block.is_full():
                    return block

            # Create new block if under limit
            if len(self.blocks[dimension]) < self.max_blocks:
                new_block = MemoryBlock(self.initial_block_size, dimension)
                self.blocks[dimension].append(new_block)
                self.stats["active_blocks"] += 1
                self._update_memory_usage()
                return new_block

            return None

    def allocate_vector(self, dimension: int) -> Optional[Tuple[npt.NDArray[np.float32], MemoryBlock, int]]:
        """Allocate vector storage.

        Args:
            dimension: Vector dimension

        Returns:
            Tuple of (vector_array, block, index), or None if allocation failed
        """
        block = self.get_block(dimension)
        if block is None:
            return None

        index = block.allocate()
        if index is None:
            return None

        vector = block.get_vector(index)
        self.stats["total_allocations"] += 1
        return vector, block, index

    def deallocate_vector(self, block: MemoryBlock, index: int) -> None:
        """Deallocate vector storage.

        Args:
            block: Memory block containing the vector
            index: Index of the vector in the block
        """
        block.deallocate(index)
        self.stats["total_deallocations"] += 1

        # Consider garbage collection for empty blocks
        if block.utilization() == 0.0 and len(self.blocks.get(block.dimension, [])) > 1:
            self._cleanup_empty_blocks()

    def _cleanup_empty_blocks(self) -> None:
        """Remove empty memory blocks to free memory."""
        with self.block_lock:
            for dimension, blocks in list(self.blocks.items()):
                # Keep at least one block per dimension
                if len(blocks) > 1:
                    self.blocks[dimension] = [
                        block for block in blocks
                        if block.utilization() > 0.0 or len(self.blocks[dimension]) == 1
                    ]
            self._update_memory_usage()

    def _update_memory_usage(self) -> None:
        """Update memory usage statistics."""
        total_memory = 0
        for blocks in self.blocks.values():
            for block in blocks:
                total_memory += block.data.nbytes
        self.stats["memory_usage_mb"] = total_memory / (1024 * 1024)
        self.stats["active_blocks"] = sum(len(blocks) for blocks in self.blocks.values())

    def get_stats(self) -> dict:
        """Get memory pool statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "pool_efficiency": (
                self.stats["total_allocations"] /
                max(1, self.stats["total_allocations"] + self.stats["total_deallocations"])
            ),
        }

    def cleanup(self) -> None:
        """Cleanup memory pool and force garbage collection."""
        with self.block_lock:
            self.blocks.clear()
            self.stats["active_blocks"] = 0
            self.stats["memory_usage_mb"] = 0.0

        gc.collect()
        logger.info("Memory pool cleaned up")


# Global memory pool instance
_memory_pool: Optional[EmbeddingMemoryPool] = None


def get_memory_pool() -> EmbeddingMemoryPool:
    """Get global memory pool instance.

    Returns:
        Memory pool singleton instance
    """
    global _memory_pool
    if _memory_pool is None:
        _memory_pool = EmbeddingMemoryPool()
    return _memory_pool


def cleanup_memory_pool() -> None:
    """Cleanup global memory pool."""
    global _memory_pool
    if _memory_pool is not None:
        _memory_pool.cleanup()
        _memory_pool = None