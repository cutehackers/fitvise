"""Vector retrieval configuration domain entity.

This module contains the VectorRetrievalConfig entity that encapsulates
configuration for vector-based retrieval operations with business validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from app.domain.exceptions.retrieval_exceptions import QueryValidationError


class SearchMode(Enum):
    """Search execution modes for vector retrieval."""

    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    VECTOR = "vector"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SEMANTIC_SEARCH = "semantic_search"

    def supports_hybrid_search(self) -> bool:
        """Check if search mode supports hybrid search."""
        return self in {SearchMode.HYBRID, SearchMode.VECTOR}

    def requires_embedding(self) -> bool:
        """Check if search mode requires embeddings."""
        return self in {SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.SIMILARITY}

    def supports_filters(self) -> bool:
        """Check if search mode supports metadata filtering."""
        return self != SearchMode.KNOWLEDGE_GRAPH


class RerankingStrategy(Enum):
    """Reranking strategies for search results."""

    NONE = "none"
    RELEVANCE_SCORE = "relevance_score"
    RECIPROCAL_RANK = "reciprocal_rank"
    LEARNING_TO_RANK = "learning_to_rank"

    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self != RerankingStrategy.NONE


@dataclass
class VectorRetrievalConfig:
    """Configuration entity for vector-based retrieval operations.

    This entity contains all configuration parameters for vector retrieval
    with validation logic and business rules embedded in the domain.
    """

    similarity_threshold: float = 0.75
    max_results: int = 10
    reranking_enabled: bool = True
    rerank_top_k: int = 5
    search_mode: SearchMode = SearchMode.SIMILARITY
    reranking_strategy: RerankingStrategy = RerankingStrategy.RELEVANCE_SCORE
    namespace: Optional[str] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    include_metadata: bool = True
    boost_recent: bool = False
    boost_factor: float = 1.2
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    timeout_seconds: int = 30

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate_similarity_threshold()
        self.validate_max_results()
        self.validate_reranking_config()
        self.validate_boost_config()
        self.validate_cache_config()
        self.validate_timeout_config()
        self.validate_search_mode_consistency()

    def validate_similarity_threshold(self) -> None:
        """Validate similarity threshold is within valid range."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise QueryValidationError(
                f"Similarity threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )

    def validate_max_results(self) -> None:
        """Validate maximum results is positive and reasonable."""
        if self.max_results <= 0:
            raise QueryValidationError(
                f"Max results must be positive, got {self.max_results}"
            )

        if self.max_results > 1000:
            raise QueryValidationError(
                f"Max results too large for performance: {self.max_results} > 1000"
            )

    def validate_reranking_config(self) -> None:
        """Validate reranking configuration consistency."""
        if self.reranking_enabled and self.rerank_top_k <= 0:
            raise QueryValidationError(
                f"Rerank top_k must be positive when reranking enabled, got {self.rerank_top_k}"
            )

        if self.rerank_top_k > self.max_results:
            raise QueryValidationError(
                f"Rerank top_k ({self.rerank_top_k}) cannot exceed max_results ({self.max_results})"
            )

    def validate_boost_config(self) -> None:
        """Validate document boosting configuration."""
        if self.boost_factor <= 0:
            raise QueryValidationError(
                f"Boost factor must be positive, got {self.boost_factor}"
            )

        if self.boost_factor > 10.0:
            raise QueryValidationError(
                f"Boost factor too large: {self.boost_factor} > 10.0"
            )

    def validate_cache_config(self) -> None:
        """Validate caching configuration."""
        if self.cache_ttl_seconds <= 0:
            raise QueryValidationError(
                f"Cache TTL must be positive, got {self.cache_ttl_seconds}"
            )

        if self.cache_ttl_seconds > 86400:  # 24 hours
            raise QueryValidationError(
                f"Cache TTL too large: {self.cache_ttl_seconds} > 86400 (24 hours)"
            )

    def validate_timeout_config(self) -> None:
        """Validate timeout configuration."""
        if self.timeout_seconds <= 0:
            raise QueryValidationError(
                f"Timeout must be positive, got {self.timeout_seconds}"
            )

        if self.timeout_seconds > 300:  # 5 minutes
            raise QueryValidationError(
                f"Timeout too large: {self.timeout_seconds} > 300 (5 minutes)"
            )

    def validate_search_mode_consistency(self) -> None:
        """Validate consistency between search mode and other settings."""
        if not self.search_mode.supports_filters() and self.metadata_filters:
            raise QueryValidationError(
                f"Search mode {self.search_mode.value} does not support metadata filters"
            )

        if not self.search_mode.requires_embedding() and self.similarity_threshold > 0:
            # Warning for non-embedding search modes with similarity threshold
            pass  # Could log a warning here

    def should_apply_reranking(self) -> bool:
        """Check if reranking should be applied based on configuration."""
        return self.reranking_enabled and self.reranking_strategy.is_enabled()

    def get_effective_top_k(self) -> int:
        """Get the effective top_k after considering reranking."""
        return self.rerank_top_k if self.should_apply_reranking() else self.max_results

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled for this configuration."""
        return self.enable_caching and self.cache_ttl_seconds > 0

    def should_boost_recent_documents(self) -> bool:
        """Check if recent documents should be boosted."""
        return self.boost_recent and self.boost_factor > 1.0

    def apply_boost_factor(self, score: float, is_recent: bool = False) -> float:
        """Apply boost factor to a similarity score."""
        if not self.should_boost_recent_documents() or not is_recent:
            return score

        boosted_score = score * self.boost_factor
        return min(boosted_score, 1.0)  # Cap at 1.0

    def get_cache_key_suffix(self) -> str:
        """Generate a cache key suffix based on configuration."""
        key_parts = [
            f"threshold_{self.similarity_threshold:.3f}",
            f"top_{self.max_results}",
            f"mode_{self.search_mode.value}",
        ]

        if self.should_apply_reranking():
            key_parts.append(f"rerank_{self.reranking_strategy.value}_{self.rerank_top_k}")

        if self.metadata_filters:
            # Sort filter keys for consistent cache keys
            sorted_filters = sorted(self.metadata_filters.items())
            filter_str = "_".join(f"{k}_{v}" for k, v in sorted_filters)
            key_parts.append(f"filters_{filter_str}")

        return "_".join(key_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "reranking_enabled": self.reranking_enabled,
            "rerank_top_k": self.rerank_top_k,
            "search_mode": self.search_mode.value,
            "reranking_strategy": self.reranking_strategy.value,
            "namespace": self.namespace,
            "metadata_filters": self.metadata_filters,
            "include_metadata": self.include_metadata,
            "boost_recent": self.boost_recent,
            "boost_factor": self.boost_factor,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorRetrievalConfig:
        """Create configuration from dictionary representation."""
        # Handle enum conversions
        if "search_mode" in data and isinstance(data["search_mode"], str):
            data["search_mode"] = SearchMode(data["search_mode"])

        if "reranking_strategy" in data and isinstance(data["reranking_strategy"], str):
            data["reranking_strategy"] = RerankingStrategy(data["reranking_strategy"])

        return cls(**data)

    def copy_with_changes(self, **kwargs) -> VectorRetrievalConfig:
        """Create a copy with specified changes."""
        current_data = self.to_dict()
        current_data.update(kwargs)
        return self.from_dict(current_data)

    def is_strict_search(self) -> bool:
        """Check if this is a strict search configuration."""
        return (
            self.similarity_threshold >= 0.8 and
            self.max_results <= 20 and
            not self.reranking_enabled
        )

    def is_performance_optimized(self) -> bool:
        """Check if this configuration is optimized for performance."""
        return (
            self.enable_caching and
            self.timeout_seconds <= 30 and
            self.max_results <= 50
        )