"""High-performance caching system for embedding operations.

Provides LRU caching with collision-resistant hashing, intelligent eviction,
and asynchronous operations for optimal embedding service performance.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import xxhash
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    embedding: Any  # EmbeddingVector
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()

    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    def idle_seconds(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


@dataclass
class CacheStats:
    """Cache performance statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    max_size_reached: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class EmbeddingCache:
    """High-performance LRU cache for embedding vectors.

    Features:
    - Collision-resistant hashing with xxHash
    - LRU eviction with intelligent sizing
    - Multi-part cache keys including model version
    - Async operations for non-blocking caching
    - Performance monitoring and statistics
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 512,
        ttl_seconds: Optional[float] = None,
        cleanup_interval: float = 300.0  # 5 minutes
    ) -> None:
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cache entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
            cleanup_interval: Interval for cleanup operations in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds

        # LRU cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_lock = asyncio.Lock()

        # Hash function for collision resistance
        self.hash_function = xxhash.xxh3_64()

        # Statistics
        self.stats = CacheStats()

        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = cleanup_interval

        # Model information for cache keys
        self.model_name: str = "unknown"
        self.model_version: str = "1.0"

    def set_model_info(self, model_name: str, model_version: str = "1.0") -> None:
        """Set model information for cache key generation.

        Args:
            model_name: Name of the embedding model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        logger.info(f"Cache model info updated: {model_name} v{model_version}")

    def _compute_cache_key(
        self,
        text: str,
        normalize: bool = False,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Compute collision-resistant cache key.

        Args:
            text: Input text
            normalize: Whether embeddings were normalized
            additional_params: Additional parameters affecting embedding

        Returns:
            Cache key string
        """
        # Multi-part hash for maximum collision resistance
        parts = [
            self.model_name,
            self.model_version,
            str(normalize),
            text
        ]

        # Add additional parameters if provided
        if additional_params:
            for key in sorted(additional_params.keys()):
                parts.append(f"{key}:{additional_params[key]}")

        # Compute hash of all parts
        combined = "|".join(parts).encode('utf-8')
        self.hash_function.reset()
        self.hash_function.update(combined)
        hash_value = self.hash_function.hexdigest()

        return hash_value

    def _estimate_size(self, embedding: Any) -> int:
        """Estimate memory size of cache entry.

        Args:
            embedding: EmbeddingVector object

        Returns:
            Estimated size in bytes
        """
        try:
            # Base object overhead
            base_size = 200  # Python object overhead

            # Vector data size
            if hasattr(embedding, '_vector'):
                vector_size = embedding._vector.nbytes
            else:
                # Fallback estimation
                vector_size = len(embedding) * 4 if hasattr(embedding, '__len__') else 768 * 4

            return base_size + vector_size + 50  # +50 for cache metadata
        except Exception:
            return 1000  # Conservative fallback estimate

    async def get(self, text: str, normalize: bool = False) -> Optional[Any]:
        """Get embedding from cache.

        Args:
            text: Input text
            normalize: Whether embedding was normalized

        Returns:
            Cached embedding vector, or None if not found
        """
        self.stats.total_requests += 1

        cache_key = self._compute_cache_key(text, normalize)

        async with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]

                # Check TTL if enabled
                if self.ttl_seconds and entry.age_seconds() > self.ttl_seconds:
                    del self.cache[cache_key]
                    self.stats.cache_misses += 1
                    logger.debug(f"Cache entry expired: {cache_key[:16]}...")
                    return None

                # Update access statistics and move to end (LRU)
                entry.touch()
                self.cache.move_to_end(cache_key)
                self.stats.cache_hits += 1

                logger.debug(f"Cache hit: {cache_key[:16]}...")
                return entry.embedding
            else:
                self.stats.cache_misses += 1
                logger.debug(f"Cache miss: {cache_key[:16]}...")
                return None

    async def put(
        self,
        text: str,
        embedding: Any,
        normalize: bool = False,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store embedding in cache.

        Args:
            text: Input text
            embedding: Embedding vector to cache
            normalize: Whether embedding was normalized
            additional_params: Additional parameters affecting embedding

        Returns:
            True if successfully cached, False if evicted or failed
        """
        cache_key = self._compute_cache_key(text, normalize, additional_params)

        # Estimate entry size
        entry_size = self._estimate_size(embedding)

        async with self.cache_lock:
            # Check if we need to evict entries
            await self._ensure_capacity(entry_size)

            # Create cache entry
            entry = CacheEntry(embedding=embedding, size_bytes=entry_size)
            entry.touch()

            # Store in cache
            self.cache[cache_key] = entry
            self.stats.memory_usage_bytes += entry_size

            # Update max size tracking
            if len(self.cache) > self.stats.max_size_reached:
                self.stats.max_size_reached = len(self.cache)

            logger.debug(f"Cached embedding: {cache_key[:16]}...")
            return True

    async def _ensure_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry.

        Args:
            new_entry_size: Size of entry being added
        """
        # Evict based on size limits
        while (
            len(self.cache) >= self.max_size or
            self.stats.memory_usage_bytes + new_entry_size > self.max_memory_bytes
        ):
            if not self.cache:
                break

            # Get oldest entry (LRU)
            oldest_key, oldest_entry = next(iter(self.cache.items()))

            # Remove entry
            del self.cache[oldest_key]
            self.stats.memory_usage_bytes -= oldest_entry.size_bytes
            self.stats.evictions += 1

            logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")

    async def get_or_compute(
        self,
        text: str,
        compute_func: Callable[[], Any],
        normalize: bool = False,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get cached embedding or compute and cache it.

        Args:
            text: Input text
            compute_func: Async function to compute embedding
            normalize: Whether embedding should be normalized
            additional_params: Additional parameters affecting embedding

        Returns:
            Embedding vector (from cache or computed)
        """
        # Try cache first
        cached = await self.get(text, normalize)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = await compute_func()

        # Cache the result
        await self.put(text, embedding, normalize, additional_params)

        return embedding

    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self.cache_lock:
            count = len(self.cache)
            self.cache.clear()
            self.stats.memory_usage_bytes = 0
            logger.info(f"Cleared {count} cache entries")
            return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0

        async with self.cache_lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.age_seconds() > self.ttl_seconds
            ]

            for key in expired_keys:
                entry = self.cache.pop(key)
                self.stats.memory_usage_bytes -= entry.size_bytes
                self.stats.evictions += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started cache cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped cache cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    def get_stats(self) -> dict:
        """Get comprehensive cache statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": self.stats.hit_rate,
            "miss_rate": self.stats.miss_rate,
            "evictions": self.stats.evictions,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "max_size_reached": self.stats.max_size_reached,
            "memory_usage_mb": self.stats.memory_usage_bytes / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "utilization_percent": (len(self.cache) / self.max_size) * 100,
            "memory_utilization_percent": (self.stats.memory_usage_bytes / self.max_memory_bytes) * 100,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "ttl_enabled": self.ttl_seconds is not None,
        }

    def get_hot_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed cache keys.

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of (cache_key_prefix, access_count) tuples
        """
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda item: item[1].access_count,
            reverse=True
        )

        return [
            (key[:16] + "...", entry.access_count)
            for key, entry in sorted_entries[:limit]
        ]


def cached_embedding(cache_instance: EmbeddingCache):
    """Decorator for caching embedding function results.

    Args:
        cache_instance: EmbeddingCache instance to use

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs) -> Any:
            normalize = kwargs.get('normalize', False)

            async def compute():
                return await func(text, *args, **kwargs)

            return await cache_instance.get_or_compute(text, compute, normalize)

        return wrapper
    return decorator


# Global cache instance
_global_cache: Optional[EmbeddingCache] = None


def get_global_cache() -> EmbeddingCache:
    """Get global embedding cache instance.

    Returns:
        Global cache singleton
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache


async def cleanup_global_cache() -> None:
    """Cleanup global cache instance."""
    global _global_cache
    if _global_cache is not None:
        await _global_cache.stop_cleanup_task()
        await _global_cache.clear()
        _global_cache = None