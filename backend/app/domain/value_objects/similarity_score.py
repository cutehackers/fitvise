"""Similarity score value object for semantic search results (Task 2.4.1).

This module defines the SimilarityScore value object that represents
the similarity score between a query and a document chunk, with
various scoring methods and confidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SimilarityMethod(Enum):
    """Enum for different similarity calculation methods."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@dataclass
class SimilarityScore:
    """Value object representing similarity between query and document.

    Encapsulates the similarity score along with metadata about how it was
    calculated, confidence levels, and additional metrics.

    Attributes:
        score: Primary similarity score (0.0-1.0, higher is more similar)
        method: Method used to calculate similarity
        confidence: Confidence level in the score (0.0-1.0)
        rank: Result rank in the search results
        distance: Original distance value before normalization
        vector_dimension: Dimension of the vectors used for comparison
        query_vector_norm: Normalization of the query vector
        document_vector_norm: Normalization of the document vector
        calculation_time_ms: Time taken to calculate similarity
        metadata: Additional scoring metadata
    """

    score: float
    method: SimilarityMethod
    confidence: float = 0.0
    rank: int = 0
    distance: Optional[float] = None
    vector_dimension: int = 0
    query_vector_norm: Optional[float] = None
    document_vector_norm: Optional[float] = None
    calculation_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate similarity score after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if self.rank < 0:
            raise ValueError("Rank must be non-negative")

        if self.vector_dimension < 0:
            raise ValueError("Vector dimension must be non-negative")

    @classmethod
    def cosine_similarity(
        cls,
        score: float,
        confidence: float = 0.0,
        distance: Optional[float] = None,
        vector_dimension: int = 0,
        query_vector_norm: Optional[float] = None,
        document_vector_norm: Optional[float] = None,
    ) -> SimilarityScore:
        """Create similarity score from cosine similarity calculation.

        Args:
            score: Cosine similarity score (0.0-1.0)
            confidence: Confidence in the score
            distance: Original cosine distance (1 - score)
            vector_dimension: Dimension of compared vectors
            query_vector_norm: L2 norm of query vector
            document_vector_norm: L2 norm of document vector

        Returns:
            SimilarityScore with cosine method
        """
        return SimilarityScore(
            score=score,
            method=SimilarityMethod.COSINE,
            confidence=confidence,
            distance=distance if distance is not None else (1.0 - score),
            vector_dimension=vector_dimension,
            query_vector_norm=query_vector_norm,
            document_vector_norm=document_vector_norm,
        )

    @classmethod
    def from_weaviate_certainty(
        cls,
        certainty: float,
        vector_dimension: int = 0,
    ) -> SimilarityScore:
        """Create similarity score from Weaviate certainty value.

        Args:
            certainty: Weaviate certainty score (0.0-1.0)
            vector_dimension: Dimension of compared vectors

        Returns:
            SimilarityScore derived from Weaviate certainty
        """
        # Weaviate certainty is already normalized to 0-1
        # Convert to similarity score with confidence based on certainty
        confidence = min(certainty * 1.2, 1.0)  # Boost confidence slightly

        return SimilarityScore(
            score=certainty,
            method=SimilarityMethod.COSINE,
            confidence=confidence,
            distance=1.0 - certainty,
            vector_dimension=vector_dimension,
            metadata={"source": "weaviate_certainty"},
        )

    @classmethod
    def low_confidence(cls, score: float, method: SimilarityMethod) -> SimilarityScore:
        """Create a low confidence similarity score.

        Args:
            score: Similarity score
            method: Calculation method used

        Returns:
            SimilarityScore with low confidence
        """
        return SimilarityScore(
            score=score,
            method=method,
            confidence=0.3,
            metadata={"confidence_level": "low"},
        )

    @classmethod
    def high_confidence(cls, score: float, method: SimilarityMethod) -> SimilarityScore:
        """Create a high confidence similarity score.

        Args:
            score: Similarity score
            method: Calculation method used

        Returns:
            SimilarityScore with high confidence
        """
        return SimilarityScore(
            score=score,
            method=method,
            confidence=0.9,
            metadata={"confidence_level": "high"},
        )

    def with_rank(self, rank: int) -> SimilarityScore:
        """Return new similarity score with updated rank.

        Args:
            rank: New rank value

        Returns:
            SimilarityScore with updated rank
        """
        return SimilarityScore(
            score=self.score,
            method=self.method,
            confidence=self.confidence,
            rank=rank,
            distance=self.distance,
            vector_dimension=self.vector_dimension,
            query_vector_norm=self.query_vector_norm,
            document_vector_norm=self.document_vector_norm,
            calculation_time_ms=self.calculation_time_ms,
            metadata=self.metadata.copy(),
        )

    def with_metadata(self, key: str, value: Any) -> SimilarityScore:
        """Return new similarity score with additional metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            SimilarityScore with updated metadata
        """
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        return SimilarityScore(
            score=self.score,
            method=self.method,
            confidence=self.confidence,
            rank=self.rank,
            distance=self.distance,
            vector_dimension=self.vector_dimension,
            query_vector_norm=self.query_vector_norm,
            document_vector_norm=self.document_vector_norm,
            calculation_time_ms=self.calculation_time_ms,
            metadata=new_metadata,
        )

    def is_above_threshold(self, threshold: float) -> bool:
        """Check if score meets minimum threshold.

        Args:
            threshold: Minimum similarity threshold

        Returns:
            True if score is above threshold
        """
        return self.score >= threshold

    def is_high_confidence(self, min_confidence: float = 0.7) -> bool:
        """Check if score has high confidence.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            True if confidence is above threshold
        """
        return self.confidence >= min_confidence

    def get_quality_label(self) -> str:
        """Get qualitative label for the similarity score.

        Returns:
            String label describing the quality of the match
        """
        if self.score >= 0.9:
            return "excellent"
        elif self.score >= 0.8:
            return "very_good"
        elif self.score >= 0.7:
            return "good"
        elif self.score >= 0.6:
            return "fair"
        elif self.score >= 0.5:
            return "poor"
        else:
            return "very_poor"

    def get_confidence_label(self) -> str:
        """Get qualitative label for confidence.

        Returns:
            String label describing confidence level
        """
        if self.confidence >= 0.9:
            return "high"
        elif self.confidence >= 0.7:
            return "medium"
        elif self.confidence >= 0.5:
            return "low"
        else:
            return "very_low"

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the similarity score
        """
        return {
            "score": self.score,
            "method": self.method.value,
            "confidence": self.confidence,
            "rank": self.rank,
            "distance": self.distance,
            "vector_dimension": self.vector_dimension,
            "query_vector_norm": self.query_vector_norm,
            "document_vector_norm": self.document_vector_norm,
            "calculation_time_ms": self.calculation_time_ms,
            "quality_label": self.get_quality_label(),
            "confidence_label": self.get_confidence_label(),
            "metadata": self.metadata,
        }

    def as_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary with essential fields.

        Returns:
            Concise dictionary representation for API responses
        """
        return {
            "score": self.score,
            "rank": self.rank,
            "quality": self.get_quality_label(),
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        """String representation of similarity score."""
        return f"SimilarityScore(score={self.score:.3f}, method={self.method.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SimilarityScore(score={self.score:.3f}, method={self.method.value}, "
            f"confidence={self.confidence:.3f}, rank={self.rank}, "
            f"quality={self.get_quality_label()})"
        )

    def __lt__(self, other: "SimilarityScore") -> bool:
        """Compare similarity scores (higher is better)."""
        return self.score < other.score

    def __le__(self, other: "SimilarityScore") -> bool:
        """Compare similarity scores."""
        return self.score <= other.score

    def __gt__(self, other: "SimilarityScore") -> bool:
        """Compare similarity scores."""
        return self.score > other.score

    def __ge__(self, other: "SimilarityScore") -> bool:
        """Compare similarity scores."""
        return self.score >= other.score

    def __eq__(self, other: object) -> bool:
        """Check equality of similarity scores."""
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return abs(self.score - other.score) < 1e-6

    def __hash__(self) -> int:
        """Generate hash for similarity score."""
        return hash((self.score, self.method, self.confidence, self.rank))