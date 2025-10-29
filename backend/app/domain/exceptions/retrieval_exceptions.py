"""Retrieval-related exceptions for semantic search (Task 2.4.1).

This module defines custom exceptions for retrieval and search operations
to provide specific error handling and debugging information.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID


class RetrievalError(Exception):
    """Base exception for retrieval operations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        query_id: Optional[UUID] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize retrieval error.

        Args:
            message: Error message
            operation: Operation that failed (e.g., "semantic_search")
            query_id: ID of the query that failed
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.query_id = query_id
        self.details = details

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]

        if self.operation:
            parts.append(f"operation: {self.operation}")

        if self.query_id:
            parts.append(f"query_id: {self.query_id}")

        if self.details:
            parts.append(f"details: {self.details}")

        return " | ".join(parts)


class SearchExecutionError(RetrievalError):
    """Exception raised when search execution fails."""

    def __init__(
        self,
        message: str,
        query_id: Optional[UUID] = None,
        search_method: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize search execution error.

        Args:
            message: Error message
            query_id: ID of the query that failed
            search_method: Search method that failed
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="search_execution",
            query_id=query_id,
            details=details,
        )
        self.search_method = search_method


class SimilaritySearchError(RetrievalError):
    """Exception raised when similarity search fails."""

    def __init__(
        self,
        message: str,
        query_id: Optional[UUID] = None,
        vector_dimension: Optional[int] = None,
        similarity_method: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize similarity search error.

        Args:
            message: Error message
            query_id: ID of the query that failed
            vector_dimension: Dimension of vectors being compared
            similarity_method: Similarity calculation method
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="similarity_search",
            query_id=query_id,
            details=details,
        )
        self.vector_dimension = vector_dimension
        self.similarity_method = similarity_method


class FilterError(RetrievalError):
    """Exception raised when search filtering fails."""

    def __init__(
        self,
        message: str,
        query_id: Optional[UUID] = None,
        filter_expression: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize filter error.

        Args:
            message: Error message
            query_id: ID of the query that failed
            filter_expression: Filter expression that failed
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="filtering",
            query_id=query_id,
            details=details,
        )
        self.filter_expression = filter_expression


class RankingError(RetrievalError):
    """Exception raised when result ranking fails."""

    def __init__(
        self,
        message: str,
        query_id: Optional[UUID] = None,
        result_count: Optional[int] = None,
        ranking_method: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize ranking error.

        Args:
            message: Error message
            query_id: ID of the query that failed
            result_count: Number of results being ranked
            ranking_method: Ranking method that failed
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="ranking",
            query_id=query_id,
            details=details,
        )
        self.result_count = result_count
        self.ranking_method = ranking_method


class IndexNotFoundError(RetrievalError):
    """Exception raised when search index is not found."""

    def __init__(
        self,
        index_name: str,
        details: Optional[str] = None,
    ) -> None:
        """Initialize index not found error.

        Args:
            index_name: Name of the missing index
            details: Additional error details
        """
        super().__init__(
            message=f"Search index '{index_name}' not found",
            operation="index_access",
            details=details,
        )
        self.index_name = index_name


class SearchTimeoutError(RetrievalError):
    """Exception raised when search operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        query_id: Optional[UUID] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout duration in seconds
            query_id: ID of the query that timed out
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="search_timeout",
            query_id=query_id,
            details=details,
        )
        self.timeout_seconds = timeout_seconds


class QueryValidationError(RetrievalError):
    """Exception raised when query validation fails."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list[str]] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize query validation error.

        Args:
            message: Error message
            validation_errors: List of specific validation errors
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="query_validation",
            details=details,
        )
        self.validation_errors = validation_errors or []


class SearchCapacityError(RetrievalError):
    """Exception raised when search service is at capacity."""

    def __init__(
        self,
        message: str,
        current_load: Optional[float] = None,
        max_capacity: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize capacity error.

        Args:
            message: Error message
            current_load: Current service load percentage
            max_capacity: Maximum capacity percentage
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="capacity_limit",
            details=details,
        )
        self.current_load = current_load
        self.max_capacity = max_capacity


class SearchResultFormatError(RetrievalError):
    """Exception raised when search result formatting fails."""

    def __init__(
        self,
        message: str,
        result_type: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize result format error.

        Args:
            message: Error message
            result_type: Type of result being formatted
            details: Additional error details
        """
        super().__init__(
            message=message,
            operation="result_formatting",
            details=details,
        )
        self.result_type = result_type