"""Reranking service interface.

This module contains the RerankingService abstract interface that defines
the contract for result reranking operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class RerankingStrategy(str, Enum):
    """Enumeration of available reranking strategies."""

    RELEVANCE_SCORE = "relevance_score"
    """Rerank by relevance score (default)."""

    SEMANTIC_SIMILARITY = "semantic_similarity"
    """Rerank by semantic similarity to query."""

    DIVERSITY = "diversity"
    """Rerank to maximize result diversity."""

    RECENCY = "recency"
    """Rerank by recency/temporal ordering."""

    COMBINED = "combined"
    """Combine multiple ranking factors."""

    CUSTOM = "custom"
    """Use custom reranking logic."""


class RerankingService(ABC):
    """Abstract service interface for result reranking operations.

    This interface defines the contract for reranking services that can
    reorder and optimize search results based on various strategies.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: int = 10,
        strategy: RerankingStrategy = RerankingStrategy.RELEVANCE_SCORE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Rerank search results based on query and strategy.

        Args:
            query: Original search query
            results: Initial search results to rerank
            top_k: Number of top results to return
            strategy: Reranking strategy to use
            metadata: Optional metadata for reranking

        Returns:
            Reranked list of results

        Raises:
            RetrievalError: If reranking operation fails
        """
        pass

    @abstractmethod
    async def batch_rerank(
        self,
        queries: List[str],
        results_lists: List[List[Any]],
        top_k: int = 10,
        strategy: RerankingStrategy = RerankingStrategy.RELEVANCE_SCORE,
    ) -> List[List[Any]]:
        """Rerank multiple result lists.

        Args:
            queries: List of original search queries
            results_lists: List of result lists to rerank
            top_k: Number of top results to return per query
            strategy: Reranking strategy to use

        Returns:
            List of reranked result lists

        Raises:
            RetrievalError: If batch reranking operation fails
        """
        pass

    @abstractmethod
    async def get_supported_strategies(self) -> List[RerankingStrategy]:
        """Get list of supported reranking strategies.

        Returns:
            List of supported strategies

        Raises:
            RetrievalError: If strategy listing fails
        """
        pass

    @abstractmethod
    async def validate_strategy(
        self,
        strategy: RerankingStrategy,
    ) -> bool:
        """Validate if a reranking strategy is supported.

        Args:
            strategy: Strategy to validate

        Returns:
            True if strategy is supported, False otherwise

        Raises:
            RetrievalError: If validation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the reranking service.

        Returns:
            Dictionary with health status information

        Raises:
            RetrievalError: If health check fails
        """
        pass

    def get_default_strategy(self) -> RerankingStrategy:
        """Get the default reranking strategy."""
        return RerankingStrategy.RELEVANCE_SCORE

    def supports_batch_operations(self) -> bool:
        """Check if this service supports batch operations."""
        return True

    def get_max_batch_size(self) -> int:
        """Get maximum batch size for operations."""
        return 50
