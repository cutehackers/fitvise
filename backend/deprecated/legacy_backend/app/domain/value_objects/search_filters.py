"""Search filters value object for semantic search criteria (Task 2.4.1).

This module defines the SearchFilters value object that encapsulates
filtering criteria for semantic search queries including document types,
date ranges, and metadata constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID


@dataclass
class SearchFilters:
    """Value object representing search filters and constraints.

    Encapsulates all filtering criteria that can be applied to a semantic
    search query to narrow down the result set.

    Attributes:
        doc_types: Set of document types to include (e.g., ['pdf', 'docx'])
        exclude_doc_types: Document types to exclude
        source_types: Set of source types to include (e.g., ['file', 'database'])
        departments: Set of departments to filter by
        categories: Set of categories to filter by
        tags: Set of tags to filter by
        date_range: Tuple of (start_date, end_date) for temporal filtering
        author_ids: Set of author IDs to filter by
        min_token_count: Minimum token count for chunks
        max_token_count: Maximum token count for chunks
        metadata_filters: Custom metadata key-value filters
        language_codes: Set of language codes (e.g., ['en', 'es'])
        quality_threshold: Minimum quality score threshold
        created_after: Filter results created after this date
        created_before: Filter results created before this date
    """

    doc_types: Set[str] = field(default_factory=set)
    exclude_doc_types: Set[str] = field(default_factory=set)
    source_types: Set[str] = field(default_factory=set)
    departments: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    date_range: Optional[tuple[datetime, datetime]] = None
    author_ids: Set[UUID] = field(default_factory=set)
    min_token_count: Optional[int] = None
    max_token_count: Optional[int] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    language_codes: Set[str] = field(default_factory=set)
    quality_threshold: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate filters after initialization."""
        # Validate date range
        if self.date_range:
            start_date, end_date = self.date_range
            if start_date >= end_date:
                raise ValueError("start_date must be before end_date in date_range")

        # Validate token counts
        if self.min_token_count is not None and self.min_token_count <= 0:
            raise ValueError("min_token_count must be greater than 0")

        if self.max_token_count is not None and self.max_token_count <= 0:
            raise ValueError("max_token_count must be greater than 0")

        if (
            self.min_token_count is not None
            and self.max_token_count is not None
            and self.min_token_count > self.max_token_count
        ):
            raise ValueError("min_token_count cannot be greater than max_token_count")

        # Validate quality threshold
        if self.quality_threshold is not None and not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0.0 and 1.0")

        # Validate created dates
        if self.created_after and self.created_before:
            if self.created_after >= self.created_before:
                raise ValueError("created_after must be before created_before")

    @classmethod
    def create_empty(cls) -> SearchFilters:
        """Create empty search filters.

        Returns:
            SearchFilters instance with no criteria applied
        """
        return SearchFilters()

    @classmethod
    def by_document_types(cls, doc_types: List[str]) -> SearchFilters:
        """Create filters for specific document types.

        Args:
            doc_types: List of document types to include

        Returns:
            SearchFilters with document type criteria
        """
        return SearchFilters(doc_types=set(doc_types))

    @classmethod
    def by_date_range(
        cls, start_date: datetime, end_date: datetime
    ) -> SearchFilters:
        """Create filters for date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            SearchFilters with date range criteria
        """
        return SearchFilters(date_range=(start_date, end_date))

    @classmethod
    def by_departments(cls, departments: List[str]) -> SearchFilters:
        """Create filters for specific departments.

        Args:
            departments: List of department names

        Returns:
            SearchFilters with department criteria
        """
        return SearchFilters(departments=set(departments))

    @classmethod
    def by_quality_threshold(cls, threshold: float) -> SearchFilters:
        """Create filters for minimum quality threshold.

        Args:
            threshold: Minimum quality score (0.0-1.0)

        Returns:
            SearchFilters with quality threshold
        """
        return SearchFilters(quality_threshold=threshold)

    def with_doc_types(self, doc_types: List[str]) -> SearchFilters:
        """Return new filters with updated document types.

        Args:
            doc_types: New document types to include

        Returns:
            New SearchFilters with updated document types
        """
        return SearchFilters(
            doc_types=set(doc_types),
            exclude_doc_types=self.exclude_doc_types,
            source_types=self.source_types,
            departments=self.departments,
            categories=self.categories,
            tags=self.tags,
            date_range=self.date_range,
            author_ids=self.author_ids,
            min_token_count=self.min_token_count,
            max_token_count=self.max_token_count,
            metadata_filters=self.metadata_filters.copy(),
            language_codes=self.language_codes,
            quality_threshold=self.quality_threshold,
            created_after=self.created_after,
            created_before=self.created_before,
        )

    def with_tags(self, tags: List[str]) -> SearchFilters:
        """Return new filters with updated tags.

        Args:
            tags: New tags to filter by

        Returns:
            New SearchFilters with updated tags
        """
        return SearchFilters(
            doc_types=self.doc_types,
            exclude_doc_types=self.exclude_doc_types,
            source_types=self.source_types,
            departments=self.departments,
            categories=self.categories,
            tags=set(tags),
            date_range=self.date_range,
            author_ids=self.author_ids,
            min_token_count=self.min_token_count,
            max_token_count=self.max_token_count,
            metadata_filters=self.metadata_filters.copy(),
            language_codes=self.language_codes,
            quality_threshold=self.quality_threshold,
            created_after=self.created_after,
            created_before=self.created_before,
        )

    def with_metadata_filter(self, key: str, value: Any) -> SearchFilters:
        """Return new filters with additional metadata filter.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            New SearchFilters with updated metadata filters
        """
        new_metadata = self.metadata_filters.copy()
        new_metadata[key] = value
        return SearchFilters(
            doc_types=self.doc_types,
            exclude_doc_types=self.exclude_doc_types,
            source_types=self.source_types,
            departments=self.departments,
            categories=self.categories,
            tags=self.tags,
            date_range=self.date_range,
            author_ids=self.author_ids,
            min_token_count=self.min_token_count,
            max_token_count=self.max_token_count,
            metadata_filters=new_metadata,
            language_codes=self.language_codes,
            quality_threshold=self.quality_threshold,
            created_after=self.created_after,
            created_before=self.created_before,
        )

    def is_empty(self) -> bool:
        """Check if filters are empty (no criteria applied).

        Returns:
            True if no filtering criteria are set
        """
        return (
            not self.doc_types
            and not self.exclude_doc_types
            and not self.source_types
            and not self.departments
            and not self.categories
            and not self.tags
            and self.date_range is None
            and not self.author_ids
            and self.min_token_count is None
            and self.max_token_count is None
            and not self.metadata_filters
            and not self.language_codes
            and self.quality_threshold is None
            and self.created_after is None
            and self.created_before is None
        )

    def to_weaviate_filters(self) -> Dict[str, Any]:
        """Convert to Weaviate filter format.

        Returns:
            Weaviate-compatible filter dictionary
        """
        operands = []

        # Document types
        if self.doc_types:
            if len(self.doc_types) == 1:
                operands.append({
                    "path": ["doc_type"],
                    "operator": "Equal",
                    "valueString": next(iter(self.doc_types)),
                })
            else:
                operands.append({
                    "operator": "Or",
                    "operands": [
                        {
                            "path": ["doc_type"],
                            "operator": "Equal",
                            "valueString": doc_type,
                        }
                        for doc_type in self.doc_types
                    ],
                })

        # Quality threshold
        if self.quality_threshold is not None:
            operands.append({
                "path": ["quality_score"],
                "operator": "GreaterThanEqual",
                "valueNumber": self.quality_threshold,
            })

        # Token count range
        if self.min_token_count is not None:
            operands.append({
                "path": ["token_count"],
                "operator": "GreaterThanEqual",
                "valueInt": self.min_token_count,
            })

        if self.max_token_count is not None:
            operands.append({
                "path": ["token_count"],
                "operator": "LessThanEqual",
                "valueInt": self.max_token_count,
            })

        # Date range
        if self.date_range:
            start_date, end_date = self.date_range
            operands.extend([
                {
                    "path": ["created_at"],
                    "operator": "GreaterThanEqual",
                    "valueDate": start_date.isoformat(),
                },
                {
                    "path": ["created_at"],
                    "operator": "LessThanEqual",
                    "valueDate": end_date.isoformat(),
                },
            ])

        # Combine operands with AND
        if len(operands) == 0:
            return {}
        elif len(operands) == 1:
            return operands[0]
        else:
            return {"operator": "And", "operands": operands}

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the filters
        """
        return {
            "doc_types": list(self.doc_types),
            "exclude_doc_types": list(self.exclude_doc_types),
            "source_types": list(self.source_types),
            "departments": list(self.departments),
            "categories": list(self.categories),
            "tags": list(self.tags),
            "date_range": (
                (self.date_range[0].isoformat(), self.date_range[1].isoformat())
                if self.date_range
                else None
            ),
            "author_ids": [str(uid) for uid in self.author_ids],
            "min_token_count": self.min_token_count,
            "max_token_count": self.max_token_count,
            "metadata_filters": self.metadata_filters,
            "language_codes": list(self.language_codes),
            "quality_threshold": self.quality_threshold,
            "created_after": self.created_after.isoformat() if self.created_after else None,
            "created_before": self.created_before.isoformat() if self.created_before else None,
        }

    def __str__(self) -> str:
        """String representation of search filters."""
        criteria = []
        if self.doc_types:
            criteria.append(f"doc_types={self.doc_types}")
        if self.departments:
            criteria.append(f"departments={self.departments}")
        if self.tags:
            criteria.append(f"tags={self.tags}")
        if self.quality_threshold is not None:
            criteria.append(f"quality≥{self.quality_threshold}")

        return f"SearchFilters({', '.join(criteria)})" if criteria else "SearchFilters(empty)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SearchFilters(doc_types={self.doc_types}, departments={self.departments}, "
            f"tags={self.tags}, quality_threshold={self.quality_threshold}, "
            f"metadata_filters={self.metadata_filters})"
        )