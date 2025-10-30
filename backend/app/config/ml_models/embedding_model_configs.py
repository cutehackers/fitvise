"""Embedding model configuration for Task 2.2.1.

This module defines configuration classes for embedding models,
supporting model selection, batch processing, device management,
and caching strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DeviceType(str, Enum):
    """Supported compute devices for embedding models."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    AUTO = "auto"  # Automatic device selection


class CacheStrategy(str, Enum):
    """Caching strategies for embedding operations."""

    NONE = "none"  # No caching
    MEMORY = "memory"  # In-memory cache
    DISK = "disk"  # Disk-based cache
    HYBRID = "hybrid"  # Memory + disk cache


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models (Task 2.2.1).

    Supports comprehensive configuration for model selection,
    batch processing, device management, and performance optimization.

    Attributes:
        model_name: HuggingFace model identifier
        model_dimension: Embedding vector dimension
        device: Target compute device
        batch_size: Batch processing size
        max_seq_length: Maximum sequence length
        normalize_embeddings: Whether to L2-normalize embeddings
        cache_strategy: Embedding cache strategy
        num_workers: Number of worker threads
        show_progress: Show progress bars during processing
        model_kwargs: Additional model initialization kwargs
        encode_kwargs: Additional encoding kwargs

    Examples:
        >>> config = EmbeddingModelConfig.default()
        >>> config.model_name
        'Alibaba-NLP/gte-multilingual-base'
        >>> config.model_dimension
        768

        >>> batch_config = EmbeddingModelConfig.for_batch_processing()
        >>> batch_config.batch_size
        32

        >>> realtime_config = EmbeddingModelConfig.for_realtime()
        >>> realtime_config.batch_size
        1
    """

    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    model_dimension: int = 768
    device: DeviceType = DeviceType.AUTO
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    num_workers: int = 4
    show_progress: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> EmbeddingModelConfig:
        """Create default embedding model configuration.

        Returns:
            Default configuration with Alibaba-NLP/gte-multilingual-base model.
        """
        return cls()

    @classmethod
    def for_batch_processing(
        cls,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingModelConfig:
        """Create configuration optimized for batch processing.

        Args:
            batch_size: Number of texts to process in parallel
            show_progress: Whether to show progress bars

        Returns:
            Configuration optimized for batch embedding (1000+ chunks/minute).
        """
        return cls(
            batch_size=batch_size,
            show_progress=show_progress,
            cache_strategy=CacheStrategy.HYBRID,
            num_workers=4,
        )

    @classmethod
    def for_realtime(
        cls,
        cache_strategy: CacheStrategy = CacheStrategy.MEMORY,
    ) -> EmbeddingModelConfig:
        """Create configuration optimized for real-time embedding.

        Args:
            cache_strategy: Cache strategy for repeated queries

        Returns:
            Configuration optimized for low-latency (<100ms).
        """
        return cls(
            batch_size=1,
            show_progress=False,
            cache_strategy=cache_strategy,
            num_workers=1,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> EmbeddingModelConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            EmbeddingModelConfig instance
        """
        # Convert string enums
        if "device" in config_dict and isinstance(config_dict["device"], str):
            config_dict["device"] = DeviceType(config_dict["device"])
        if "cache_strategy" in config_dict and isinstance(
            config_dict["cache_strategy"], str
        ):
            config_dict["cache_strategy"] = CacheStrategy(config_dict["cache_strategy"])

        return cls(**config_dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "model_name": self.model_name,
            "model_dimension": self.model_dimension,
            "device": self.device.value,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_strategy": self.cache_strategy.value,
            "num_workers": self.num_workers,
            "show_progress": self.show_progress,
            "model_kwargs": self.model_kwargs,
            "encode_kwargs": self.encode_kwargs,
        }

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥1, got {self.batch_size}")
        if self.model_dimension < 1:
            raise ValueError(
                f"model_dimension must be ≥1, got {self.model_dimension}"
            )
        if self.max_seq_length < 1:
            raise ValueError(
                f"max_seq_length must be ≥1, got {self.max_seq_length}"
            )
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be ≥1, got {self.num_workers}")


def get_embedding_config(
    config_dict: Optional[Dict[str, Any]] = None,
) -> EmbeddingModelConfig:
    """Get embedding model configuration.

    Args:
        config_dict: Optional configuration dictionary override

    Returns:
        EmbeddingModelConfig instance
    """
    if config_dict:
        return EmbeddingModelConfig.from_dict(config_dict)
    return EmbeddingModelConfig.default()
