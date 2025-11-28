"""Similarity score value object.

This module defines the SimilarityScore value object for representing
similarity scores with validation and utility methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SimilarityScore:
    """Immutable similarity score value object.

    Represents a similarity score with validation and utility methods
    for comparing and converting to different formats.

    Attributes:
        score: The similarity score value (0.0 to 1.0)
        method: The method used to calculate the score

    Examples:
        >>> score = SimilarityScore(0.85, "cosine")
        >>> score.is_high_similarity()
        True
        >>> score.as_dict()
        {'score': 0.85, 'method': 'cosine'}
    """

    score: float
    method: str = "cosine"

    def __post_init__(self) -> None:
        """Validate similarity score after initialization."""
        if not isinstance(self.score, (int, float)):
            raise ValueError("Score must be a number")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Method must be a non-empty string")

    @classmethod
    def cosine(cls, score: float) -> SimilarityScore:
        """Create a cosine similarity score.

        Args:
            score: Cosine similarity score (0.0 to 1.0)

        Returns:
            SimilarityScore instance with cosine method
        """
        return cls(score, "cosine")

    @classmethod
    def euclidean(cls, score: float) -> SimilarityScore:
        """Create an euclidean similarity score.

        Args:
            score: Euclidean similarity score (0.0 to 1.0)

        Returns:
            SimilarityScore instance with euclidean method
        """
        return cls(score, "euclidean")

    def is_high_similarity(self, threshold: float = 0.8) -> bool:
        """Check if this is a high similarity score.

        Args:
            threshold: Threshold for high similarity (default 0.8)

        Returns:
            True if score is above threshold
        """
        return self.score >= threshold

    def is_medium_similarity(self, min_threshold: float = 0.5, max_threshold: float = 0.8) -> bool:
        """Check if this is a medium similarity score.

        Args:
            min_threshold: Minimum threshold for medium similarity
            max_threshold: Maximum threshold for medium similarity

        Returns:
            True if score is in medium range
        """
        return min_threshold <= self.score < max_threshold

    def is_low_similarity(self, threshold: float = 0.5) -> bool:
        """Check if this is a low similarity score.

        Args:
            threshold: Threshold for low similarity (default 0.5)

        Returns:
            True if score is below threshold
        """
        return self.score < threshold

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with score and method
        """
        return {
            "score": self.score,
            "method": self.method,
        }

    def __str__(self) -> str:
        """String representation of similarity score."""
        return f"SimilarityScore({self.score:.3f}, {self.method})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"SimilarityScore(score={self.score}, method='{self.method}')"