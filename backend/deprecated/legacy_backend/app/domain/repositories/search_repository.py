"""Search repository interface for semantic search operations (Task 2.4.1).

This module defines the SearchRepository interface that abstracts
the underlying search implementation (Weaviate, Elasticsearch, etc.)
for semantic search functionality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.entities.search_query import SearchQuery
from app.domain.entities.search_result import SearchResult
from app.domain.exceptions.retrieval_exceptions import RetrievalError


class SearchRepository(ABC):
    """Repository interface for semantic search operations.

    Defines the contract for search implementations that can perform
    semantic similarity search with filtering and ranking capabilities.
    """

    @abstractmethod
    async def semantic_search(
        self,
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Perform semantic similarity search.

        Args:
            query: Search query with text, filters, and configuration

        Returns:
            List of search results ranked by relevance

        Raises:
            RetrievalError: If search operation fails
        """
        pass

    @abstractmethod
    async def find_similar_chunks(
        self,
        chunk_ids: List[str],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Find chunks similar to given chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to find similar items for
            top_k: Maximum number of results per chunk
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar chunks grouped by input chunk

        Raises:
            RetrievalError: If similarity search fails
        """
        pass

    @abstractmethod
    async def search_by_document_ids(
        self,
        document_ids: List[str],
        query_text: Optional[str] = None,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """Search within specific documents.

        Args:
            document_ids: List of document IDs to search within
            query_text: Optional query text for semantic filtering
            top_k: Maximum number of results per document

        Returns:
            List of chunks from specified documents

        Raises:
            RetrievalError: If document search fails
        """
        pass

    @abstractmethod
    async def get_search_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 5,
        min_similarity: float = 0.3,
    ) -> List[str]:
        """Get search suggestions based on partial query.

        Args:
            partial_query: Partial search query text
            max_suggestions: Maximum number of suggestions
            min_similarity: Minimum similarity for suggestions

        Returns:
            List of suggested query completions

        Raises:
            RetrievalError: If suggestion generation fails
        """
        pass

    @abstractmethod
    async def aggregate_search_results(
        self,
        queries: List[SearchQuery],
        aggregation_method: str = "reciprocal_rank",
    ) -> List[SearchResult]:
        """Aggregate results from multiple search queries.

        Args:
            queries: List of search queries to execute
            aggregation_method: Method for combining results
                - "reciprocal_rank": Reciprocal rank fusion
                - "average_score": Average similarity scores
                - "max_score": Maximum score per document

        Returns:
            Aggregated list of search results

        Raises:
            RetrievalError: If aggregation fails
        """
        pass

    @abstractmethod
    async def get_popular_queries(
        self,
        limit: int = 10,
        time_range_days: int = 30,
    ) -> List[tuple[str, int]]:
        """Get most popular search queries.

        Args:
            limit: Maximum number of queries to return
            time_range_days: Number of days to look back

        Returns:
            List of (query_text, frequency) tuples

        Raises:
            RetrievalError: If query analytics fail
        """
        pass

    @abstractmethod
    async def log_search_interaction(
        self,
        query_id: str,
        result_ids: List[str],
        clicked_result_id: Optional[str] = None,
        feedback_score: Optional[float] = None,
    ) -> None:
        """Log user interaction with search results.

        Args:
            query_id: ID of the search query
            result_ids: IDs of returned results
            clicked_result_id: ID of result user clicked (if any)
            feedback_score: User feedback score (1-5, if provided)

        Raises:
            RetrievalError: If logging fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict:
        """Check health and performance of search repository.

        Returns:
            Dictionary with health status and metrics

        Raises:
            RetrievalError: If health check fails
        """
        pass

    @abstractmethod
    async def get_search_statistics(
        self,
        time_range_days: int = 7,
    ) -> dict:
        """Get search performance and usage statistics.

        Args:
            time_range_days: Number of days to analyze

        Returns:
            Dictionary with search statistics

        Raises:
            RetrievalError: If statistics retrieval fails
        """
        pass

    async def validate_query(self, query: SearchQuery) -> bool:
        """Validate search query before execution.

        Args:
            query: Search query to validate

        Returns:
            True if query is valid

        Raises:
            RetrievalError: If validation fails
        """
        try:
            # Basic validation
            if not query.text or not query.text.strip():
                raise RetrievalError("Query text cannot be empty")

            if query.top_k <= 0:
                raise RetrievalError("top_k must be greater than 0")

            if not 0.0 <= query.min_similarity <= 1.0:
                raise RetrievalError("min_similarity must be between 0.0 and 1.0")

            # Validate filters if provided
            if query.filters:
                query.filters.to_weaviate_filters()  # This will validate filters

            return True

        except Exception as e:
            raise RetrievalError(f"Query validation failed: {str(e)}") from e

    async def estimate_search_cost(self, query: SearchQuery) -> dict:
        """Est computational cost of search query.

        Args:
            query: Search query to analyze

        Returns:
            Dictionary with cost estimates
        """
        base_cost = 1.0
        filter_cost = 0.1 if not query.filters.is_empty() else 0.0
        result_cost = query.top_k * 0.01

        return {
            "estimated_cost_tokens": base_cost + filter_cost + result_cost,
            "estimated_latency_ms": 50 + (query.top_k * 2),
            "complexity": "low" if query.filters.is_empty() else "medium",
        }