"""Vector embedding value object.

This module contains the VectorEmbedding value object that represents
vector embeddings with validation and similarity calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.domain.exceptions.retrieval_exceptions import QueryValidationError
from app.domain.value_objects.similarity_score import SimilarityScore


@dataclass(frozen=True)
class VectorEmbedding:
    """Immutable value object representing a vector embedding.

    This value object encapsulates vector embeddings with validation,
    similarity calculations, and utility methods for vector operations.
    """

    values: List[float]
    dimension: int
    model_name: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate vector embedding after initialization."""
        if not isinstance(self.values, list):
            raise QueryValidationError(f"Values must be a list, got {type(self.values)}")

        if not self.values:
            raise QueryValidationError("Values cannot be empty")

        if not all(isinstance(v, (int, float)) for v in self.values):
            raise QueryValidationError("All values must be numeric")

        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise QueryValidationError(f"Dimension must be a positive integer, got {self.dimension}")

        if len(self.values) != self.dimension:
            raise QueryValidationError(
                f"Vector length ({len(self.values)}) must match dimension ({self.dimension})"
            )

        # Check for NaN or infinite values
        for i, v in enumerate(self.values):
            if np.isnan(v):
                raise QueryValidationError(f"Value at index {i} is NaN")
            if np.isinf(v):
                raise QueryValidationError(f"Value at index {i} is infinite")

        if self.model_name is not None and not isinstance(self.model_name, str):
            raise QueryValidationError(f"Model name must be a string, got {type(self.model_name)}")

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude (L2 norm) of the vector."""
        return float(np.linalg.norm(self.values))

    @property
    def is_normalized(self) -> bool:
        """Check if the vector is normalized (unit vector)."""
        return abs(self.magnitude - 1.0) < 1e-6

    @property
    def is_zero_vector(self) -> bool:
        """Check if this is a zero vector."""
        return all(abs(v) < 1e-10 for v in self.values)

    @property
    def sparsity(self) -> float:
        """Calculate the sparsity ratio of the vector."""
        zero_count = sum(1 for v in self.values if abs(v) < 1e-10)
        return zero_count / len(self.values)

    @property
    def density(self) -> float:
        """Calculate the density ratio of the vector."""
        return 1.0 - self.sparsity

    def normalize(self) -> VectorEmbedding:
        """Create a normalized version of this vector."""
        if self.is_zero_vector:
            raise QueryValidationError("Cannot normalize a zero vector")

        if self.is_normalized:
            return self

        magnitude = self.magnitude
        normalized_values = [v / magnitude for v in self.values]

        return VectorEmbedding(
            values=normalized_values,
            dimension=self.dimension,
            model_name=self.model_name,
            metadata={** (self.metadata or {}), "normalized": True}
        )

    def similarity(self, other: VectorEmbedding) -> SimilarityScore:
        """Calculate cosine similarity with another vector."""
        if not isinstance(other, VectorEmbedding):
            raise QueryValidationError("Can only calculate similarity with VectorEmbedding objects")

        if self.dimension != other.dimension:
            raise QueryValidationError(
                f"Cannot compare vectors of different dimensions: {self.dimension} vs {other.dimension}"
            )

        # Handle zero vectors
        if self.is_zero_vector or other.is_zero_vector:
            return SimilarityScore.from_float(0.0)

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(self.values, other.values))
        magnitude_product = self.magnitude * other.magnitude

        if magnitude_product == 0:
            return SimilarityScore.from_float(0.0)

        similarity = dot_product / magnitude_product
        similarity = max(-1.0, min(1.0, similarity))  # Clamp to [-1, 1]

        # Convert to [0, 1] range for similarity scores
        normalized_similarity = (similarity + 1.0) / 2.0

        return SimilarityScore.from_float(normalized_similarity)

    def euclidean_distance(self, other: VectorEmbedding) -> float:
        """Calculate Euclidean distance to another vector."""
        if not isinstance(other, VectorEmbedding):
            raise QueryValidationError("Can only calculate distance with VectorEmbedding objects")

        if self.dimension != other.dimension:
            raise QueryValidationError(
                f"Cannot calculate distance between vectors of different dimensions: {self.dimension} vs {other.dimension}"
            )

        return float(np.linalg.norm([a - b for a, b in zip(self.values, other.values)]))

    def manhattan_distance(self, other: VectorEmbedding) -> float:
        """Calculate Manhattan distance to another vector."""
        if not isinstance(other, VectorEmbedding):
            raise QueryValidationError("Can only calculate distance with VectorEmbedding objects")

        if self.dimension != other.dimension:
            raise QueryValidationError(
                f"Cannot calculate distance between vectors of different dimensions: {self.dimension} vs {other.dimension}"
            )

        return sum(abs(a - b) for a, b in zip(self.values, other.values))

    def dot_product(self, other: VectorEmbedding) -> float:
        """Calculate dot product with another vector."""
        if not isinstance(other, VectorEmbedding):
            raise QueryValidationError("Can only calculate dot product with VectorEmbedding objects")

        if self.dimension != other.dimension:
            raise QueryValidationError(
                f"Cannot calculate dot product between vectors of different dimensions: {self.dimension} vs {other.dimension}"
            )

        return sum(a * b for a, b in zip(self.values, other.values))

    def add(self, other: VectorEmbedding) -> VectorEmbedding:
        """Add another vector to this one."""
        if not isinstance(other, VectorEmbedding):
            raise QueryValidationError("Can only add VectorEmbedding objects")

        if self.dimension != other.dimension:
            raise QueryValidationError(
                f"Cannot add vectors of different dimensions: {self.dimension} vs {other.dimension}"
            )

        result_values = [a + b for a, b in zip(self.values, other.values)]

        return VectorEmbedding(
            values=result_values,
            dimension=self.dimension,
            model_name=self.model_name,
            metadata={
                ** (self.metadata or {}),
                "operation": "addition",
                "other_model": other.model_name
            }
        )

    def subtract(self, other: VectorEmbedding) -> VectorEmbedding:
        """Subtract another vector from this one."""
        if not isinstance(other, VectorEmbedding):
            raise QueryValidationError("Can only subtract VectorEmbedding objects")

        if self.dimension != other.dimension:
            raise QueryValidationError(
                f"Cannot subtract vectors of different dimensions: {self.dimension} vs {other.dimension}"
            )

        result_values = [a - b for a, b in zip(self.values, other.values)]

        return VectorEmbedding(
            values=result_values,
            dimension=self.dimension,
            model_name=self.model_name,
            metadata={
                ** (self.metadata or {}),
                "operation": "subtraction",
                "other_model": other.model_name
            }
        )

    def multiply(self, scalar: float) -> VectorEmbedding:
        """Multiply this vector by a scalar."""
        if not isinstance(scalar, (int, float)):
            raise QueryValidationError(f"Scalar must be numeric, got {type(scalar)}")

        if np.isnan(scalar) or np.isinf(scalar):
            raise QueryValidationError("Scalar cannot be NaN or infinite")

        result_values = [v * scalar for v in self.values]

        return VectorEmbedding(
            values=result_values,
            dimension=self.dimension,
            model_name=self.model_name,
            metadata={
                ** (self.metadata or {}),
                "operation": "scalar_multiplication",
                "scalar": scalar
            }
        )

    def top_k_indices(self, k: int) -> List[int]:
        """Get indices of the top k largest values."""
        if not isinstance(k, int) or k <= 0:
            raise QueryValidationError(f"k must be a positive integer, got {k}")

        if k > self.dimension:
            raise QueryValidationError(f"k ({k}) cannot exceed dimension ({self.dimension})")

        # Get indices of top k values
        indexed_values = [(i, v) for i, v in enumerate(self.values)]
        indexed_values.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in indexed_values[:k]]

    def top_k_values(self, k: int) -> List[float]:
        """Get the top k largest values."""
        if not isinstance(k, int) or k <= 0:
            raise QueryValidationError(f"k must be a positive integer, got {k}")

        if k > self.dimension:
            raise QueryValidationError(f"k ({k}) cannot exceed dimension ({self.dimension})")

        sorted_values = sorted(self.values, reverse=True)
        return sorted_values[:k]

    def slice(self, start: int, end: Optional[int] = None) -> VectorEmbedding:
        """Create a new vector from a slice of this vector."""
        if not isinstance(start, int) or start < 0:
            raise QueryValidationError(f"Start index must be a non-negative integer, got {start}")

        if start >= self.dimension:
            raise QueryValidationError(f"Start index ({start}) must be less than dimension ({self.dimension})")

        if end is None:
            end = self.dimension
        elif not isinstance(end, int) or end < 0:
            raise QueryValidationError(f"End index must be a non-negative integer, got {end}")

        if end > self.dimension:
            raise QueryValidationError(f"End index ({end}) cannot exceed dimension ({self.dimension})")

        if start >= end:
            raise QueryValidationError(f"Start index ({start}) must be less than end index ({end})")

        sliced_values = self.values[start:end]
        new_dimension = end - start

        return VectorEmbedding(
            values=sliced_values,
            dimension=new_dimension,
            model_name=self.model_name,
            metadata={
                ** (self.metadata or {}),
                "operation": "slice",
                "slice_range": [start, end]
            }
        )

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values, dtype=np.float32)

    def to_list(self) -> List[float]:
        """Convert to plain list."""
        return self.values.copy()

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "values": self.values,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "metadata": self.metadata or {},
            "magnitude": self.magnitude,
            "is_normalized": self.is_normalized,
            "is_zero_vector": self.is_zero_vector,
            "sparsity": self.sparsity,
            "density": self.density,
        }

    @classmethod
    def from_numpy(cls, array: np.ndarray, model_name: Optional[str] = None, metadata: Optional[dict] = None) -> VectorEmbedding:
        """Create from numpy array."""
        if not isinstance(array, np.ndarray):
            raise QueryValidationError(f"Input must be a numpy array, got {type(array)}")

        if array.ndim != 1:
            raise QueryValidationError(f"Array must be 1-dimensional, got {array.ndim} dimensions")

        values = array.astype(float).tolist()
        return cls(values=values, dimension=len(values), model_name=model_name, metadata=metadata)

    @classmethod
    def from_list(cls, values: List[float], model_name: Optional[str] = None, metadata: Optional[dict] = None) -> VectorEmbedding:
        """Create from list of values."""
        return cls(values=values, dimension=len(values), model_name=model_name, metadata=metadata)

    @classmethod
    def zeros(cls, dimension: int, model_name: Optional[str] = None) -> VectorEmbedding:
        """Create a zero vector."""
        return cls(values=[0.0] * dimension, dimension=dimension, model_name=model_name)

    @classmethod
    def random(cls, dimension: int, model_name: Optional[str] = None) -> VectorEmbedding:
        """Create a random vector with normal distribution."""
        random_values = np.random.normal(0, 1, dimension).astype(float).tolist()
        return cls(values=random_values, dimension=dimension, model_name=model_name)

    @classmethod
    def random_uniform(cls, dimension: int, model_name: Optional[str] = None) -> VectorEmbedding:
        """Create a random vector with uniform distribution."""
        random_values = np.random.uniform(-1, 1, dimension).astype(float).tolist()
        return cls(values=random_values, dimension=dimension, model_name=model_name)

    def __str__(self) -> str:
        """String representation."""
        return f"VectorEmbedding(dimension={self.dimension}, model={self.model_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        preview = self.values[:3]
        preview_str = ", ".join(f"{v:.3f}" for v in preview)
        if len(self.values) > 3:
            preview_str += f", ... ({len(self.values) - 3} more)"
        return f"VectorEmbedding([{preview_str}], dimension={self.dimension})"

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, VectorEmbedding):
            return False

        if self.dimension != other.dimension:
            return False

        # Check values with tolerance for floating point precision
        tolerance = 1e-6
        for a, b in zip(self.values, other.values):
            if abs(a - b) > tolerance:
                return False

        return True

    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        # Use tuple of rounded values for hash
        rounded_values = tuple(round(v, 6) for v in self.values)
        return hash((rounded_values, self.dimension, self.model_name))
