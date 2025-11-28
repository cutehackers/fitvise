"""Similarity score value object.

This module contains the SimilarityScore value object that represents
similarity scores with validation and business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.domain.exceptions.retrieval_exceptions import QueryValidationError


@dataclass(frozen=True)
class SimilarityScore:
    """Immutable value object representing a similarity score.

    This value object encapsulates similarity scores with validation,
    comparison logic, and business rules for relevance determination.
    """

    value: float
    confidence: Optional[float] = None
    rank: Optional[int] = None

    def __post_init__(self):
        """Validate similarity score after initialization."""
        if not isinstance(self.value, (int, float)):
            raise QueryValidationError(f"Similarity score must be numeric, got {type(self.value)}")

        if not 0.0 <= self.value <= 1.0:
            raise QueryValidationError(
                f"Similarity score must be between 0.0 and 1.0, got {self.value}"
            )

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise QueryValidationError(f"Confidence must be numeric, got {type(self.confidence)}")

            if not 0.0 <= self.confidence <= 1.0:
                raise QueryValidationError(
                    f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
                )

        if self.rank is not None:
            if not isinstance(self.rank, int):
                raise QueryValidationError(f"Rank must be an integer, got {type(self.rank)}")

            if self.rank <= 0:
                raise QueryValidationError(f"Rank must be positive, got {self.rank}")

    @property
    def is_perfect_match(self) -> bool:
        """Check if this is a perfect match."""
        return self.value >= 0.99

    @property
    def is_high_relevance(self) -> bool:
        """Check if this score indicates high relevance."""
        return self.value >= 0.8

    @property
    def is_relevant(self) -> bool:
        """Check if this score indicates relevance."""
        return self.value >= 0.5

    @property
    def is_low_relevance(self) -> bool:
        """Check if this score indicates low relevance."""
        return self.value < 0.3

    @property
    def is_high_confidence(self) -> bool:
        """Check if the confidence is high."""
        return self.confidence is not None and self.confidence >= 0.8

    @property
    def is_low_confidence(self) -> bool:
        """Check if the confidence is low."""
        return self.confidence is not None and self.confidence < 0.5

    @property
    def relevance_level(self) -> str:
        """Get the relevance level as a string."""
        if self.is_perfect_match:
            return "perfect"
        elif self.is_high_relevance:
            return "high"
        elif self.is_relevant:
            return "medium"
        elif self.value > 0.2:
            return "low"
        else:
            return "very_low"

    @property
    def confidence_level(self) -> str:
        """Get the confidence level as a string."""
        if self.confidence is None:
            return "unknown"
        elif self.is_high_confidence:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        else:
            return "low"

    def is_better_than(self, other: SimilarityScore) -> bool:
        """Compare this score with another similarity score."""
        if not isinstance(other, SimilarityScore):
            raise QueryValidationError("Can only compare with SimilarityScore objects")

        # Primary comparison by value
        if self.value != other.value:
            return self.value > other.value

        # Secondary comparison by confidence
        if self.confidence is not None and other.confidence is not None:
            return self.confidence > other.confidence

        # Prefer scores with confidence when the other doesn't have it
        if self.confidence is not None and other.confidence is None:
            return True

        # Prefer scores without confidence when this one doesn't have it
        if self.confidence is None and other.confidence is not None:
            return False

        # Tertiary comparison by rank (lower rank is better)
        if self.rank is not None and other.rank is not None:
            return self.rank < other.rank

        return False

    def is_equivalent_to(self, other: SimilarityScore, tolerance: float = 0.01) -> bool:
        """Check if this score is equivalent to another within tolerance."""
        if not isinstance(other, SimilarityScore):
            raise QueryValidationError("Can only compare with SimilarityScore objects")

        return abs(self.value - other.value) <= tolerance

    def with_adjusted_value(self, factor: float, min_value: float = 0.0, max_value: float = 1.0) -> SimilarityScore:
        """Create a new SimilarityScore with adjusted value."""
        adjusted_value = self.value * factor
        adjusted_value = max(min_value, min(max_value, adjusted_value))

        return SimilarityScore(
            value=adjusted_value,
            confidence=self.confidence,
            rank=self.rank
        )

    def with_confidence(self, confidence: float) -> SimilarityScore:
        """Create a new SimilarityScore with specified confidence."""
        return SimilarityScore(
            value=self.value,
            confidence=confidence,
            rank=self.rank
        )

    def with_rank(self, rank: int) -> SimilarityScore:
        """Create a new SimilarityScore with specified rank."""
        return SimilarityScore(
            value=self.value,
            confidence=self.confidence,
            rank=rank
        )

    def to_percentage(self) -> str:
        """Convert score to percentage string."""
        return f"{self.value * 100:.1f}%"

    def to_detailed_string(self) -> str:
        """Convert to detailed string representation."""
        parts = [f"score={self.value:.3f} ({self.relevance_level})"]

        if self.confidence is not None:
            parts.append(f"confidence={self.confidence:.3f} ({self.confidence_level})")

        if self.rank is not None:
            parts.append(f"rank={self.rank}")

        return "SimilarityScore(" + ", ".join(parts) + ")"

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "value": self.value,
            "relevance_level": self.relevance_level,
            "is_perfect_match": self.is_perfect_match,
            "is_high_relevance": self.is_high_relevance,
            "is_relevant": self.is_relevant,
            "is_low_relevance": self.is_low_relevance,
        }

        if self.confidence is not None:
            result.update({
                "confidence": self.confidence,
                "confidence_level": self.confidence_level,
                "is_high_confidence": self.is_high_confidence,
                "is_low_confidence": self.is_low_confidence,
            })

        if self.rank is not None:
            result["rank"] = self.rank

        return result

    @classmethod
    def from_float(cls, value: float, confidence: Optional[float] = None, rank: Optional[int] = None) -> SimilarityScore:
        """Create from float value."""
        return cls(value=value, confidence=confidence, rank=rank)

    @classmethod
    def from_percentage(cls, percentage: float, confidence: Optional[float] = None, rank: Optional[int] = None) -> SimilarityScore:
        """Create from percentage value (0-100)."""
        value = percentage / 100.0
        return cls(value=value, confidence=confidence, rank=rank)

    @classmethod
    def max(cls) -> SimilarityScore:
        """Create a maximum similarity score."""
        return cls(value=1.0, confidence=1.0, rank=1)

    @classmethod
    def min(cls) -> SimilarityScore:
        """Create a minimum similarity score."""
        return cls(value=0.0, confidence=0.0)

    @classmethod
    def threshold(cls, threshold_value: float) -> SimilarityScore:
        """Create a similarity score at a specific threshold."""
        return cls(value=threshold_value)

    def __str__(self) -> str:
        """String representation."""
        return f"SimilarityScore({self.value:.3f})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.to_detailed_string()

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, SimilarityScore):
            return False

        return (
            abs(self.value - other.value) < 1e-9 and
            self.confidence == other.confidence and
            self.rank == other.rank
        )

    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.value >= other.value

    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        return hash((self.value, self.confidence, self.rank))
