"""Embedding vector value object (Epic 2.2).

This module defines the EmbeddingVector value object representing
an immutable vector with similarity computation methods.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import numpy.typing as npt


class EmbeddingVector:
    """Immutable embedding vector value object (Task 2.2.1).

    Represents a dense embedding vector with utility methods for
    similarity computation, normalization, and validation.

    Attributes:
        _vector: Underlying numpy array (immutable)

    Examples:
        >>> import numpy as np
        >>> vec1 = EmbeddingVector(np.array([1.0, 2.0, 3.0]))
        >>> vec2 = EmbeddingVector(np.array([2.0, 3.0, 4.0]))
        >>> similarity = vec1.cosine_similarity(vec2)
        >>> 0.0 <= similarity <= 1.0
        True

        >>> # Normalization
        >>> vec = EmbeddingVector(np.array([3.0, 4.0]))
        >>> norm_vec = vec.normalize()
        >>> norm_vec.is_normalized()
        True

        >>> # From list
        >>> vec_from_list = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        >>> len(vec_from_list)
        3
    """

    __slots__ = ("_vector",)

    def __init__(self, vector: npt.NDArray[np.float32]) -> None:
        """Initialize embedding vector.

        Args:
            vector: Numpy array of embedding values

        Raises:
            ValueError: If vector is invalid
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError("vector must be numpy array")
        if vector.ndim != 1:
            raise ValueError(f"vector must be 1-dimensional, got {vector.ndim}D")
        if len(vector) == 0:
            raise ValueError("vector cannot be empty")

        # Store as immutable by copying
        self._vector = vector.astype(np.float32).copy()
        self._vector.setflags(write=False)

    @classmethod
    def from_list(cls, values: List[float]) -> EmbeddingVector:
        """Create embedding vector from list of floats.

        Args:
            values: List of float values

        Returns:
            EmbeddingVector instance

        Raises:
            ValueError: If values are invalid
        """
        if not values:
            raise ValueError("values cannot be empty")

        array = np.array(values, dtype=np.float32)
        return cls(array)

    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float32]) -> EmbeddingVector:
        """Create embedding vector from numpy array.

        Args:
            array: Numpy array

        Returns:
            EmbeddingVector instance
        """
        return cls(array)

    def to_list(self) -> List[float]:
        """Convert vector to list of floats.

        Returns:
            List of float values
        """
        return self._vector.tolist()

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Get underlying numpy array (read-only copy).

        Returns:
            Read-only numpy array
        """
        return self._vector.copy()

    def cosine_similarity(self, other: EmbeddingVector) -> float:
        """Calculate cosine similarity with another vector.

        Cosine similarity ranges from -1 (opposite) to 1 (identical).
        For normalized vectors, this is equivalent to dot product.

        Args:
            other: Other embedding vector

        Returns:
            Cosine similarity score

        Raises:
            ValueError: If vectors have different dimensions
        """
        if len(self) != len(other):
            raise ValueError(
                f"dimension mismatch: {len(self)} != {len(other)}"
            )

        # Compute cosine similarity
        dot_product = float(np.dot(self._vector, other._vector))
        norm_product = float(np.linalg.norm(self._vector) * np.linalg.norm(other._vector))

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    def dot_product(self, other: EmbeddingVector) -> float:
        """Calculate dot product with another vector.

        Args:
            other: Other embedding vector

        Returns:
            Dot product value

        Raises:
            ValueError: If vectors have different dimensions
        """
        if len(self) != len(other):
            raise ValueError(
                f"dimension mismatch: {len(self)} != {len(other)}"
            )

        return float(np.dot(self._vector, other._vector))

    def euclidean_distance(self, other: EmbeddingVector) -> float:
        """Calculate Euclidean (L2) distance to another vector.

        Args:
            other: Other embedding vector

        Returns:
            Euclidean distance

        Raises:
            ValueError: If vectors have different dimensions
        """
        if len(self) != len(other):
            raise ValueError(
                f"dimension mismatch: {len(self)} != {len(other)}"
            )

        return float(np.linalg.norm(self._vector - other._vector))

    def normalize(self) -> EmbeddingVector:
        """Create normalized (L2) copy of this vector.

        Returns:
            New EmbeddingVector with L2 norm = 1
        """
        norm = np.linalg.norm(self._vector)
        if norm == 0:
            return EmbeddingVector(self._vector.copy())

        normalized = self._vector / norm
        return EmbeddingVector(normalized)

    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """Check if vector is normalized (L2 norm ≈ 1).

        Args:
            tolerance: Tolerance for norm comparison

        Returns:
            True if vector is normalized
        """
        norm = np.linalg.norm(self._vector)
        return abs(norm - 1.0) < tolerance

    def magnitude(self) -> float:
        """Calculate vector magnitude (L2 norm).

        Returns:
            Vector magnitude
        """
        return float(np.linalg.norm(self._vector))

    def norm(self) -> float:
        """Alias for magnitude() - calculate vector L2 norm.

        Returns:
            Vector L2 norm
        """
        return self.magnitude()

    def validate(self) -> None:
        """Validate vector integrity.

        Raises:
            ValueError: If vector is invalid
        """
        if len(self) == 0:
            raise ValueError("vector cannot be empty")
        if np.any(np.isnan(self._vector)):
            raise ValueError("vector contains NaN values")
        if np.any(np.isinf(self._vector)):
            raise ValueError("vector contains infinite values")

    @property
    def dimension(self) -> int:
        """Get vector dimension.

        Returns:
            Vector dimension (length)
        """
        return len(self._vector)

    def __len__(self) -> int:
        """Get vector dimension.

        Returns:
            Vector length
        """
        return len(self._vector)

    def __eq__(self, other: object) -> bool:
        """Check equality with another vector.

        Args:
            other: Object to compare

        Returns:
            True if vectors are equal
        """
        if not isinstance(other, EmbeddingVector):
            return False
        return np.array_equal(self._vector, other._vector)

    def __hash__(self) -> int:
        """Calculate hash of vector.

        Returns:
            Hash value
        """
        return hash(self._vector.tobytes())

    def __getitem__(self, index: int) -> float:
        """Get vector element by index.

        Args:
            index: Element index

        Returns:
            Vector element value
        """
        return float(self._vector[index])

    def __repr__(self) -> str:
        """String representation of vector."""
        preview = self._vector[:3].tolist()
        if len(self) > 3:
            preview_str = f"{preview[0]:.3f}, {preview[1]:.3f}, {preview[2]:.3f}..."
        else:
            preview_str = ", ".join(f"{v:.3f}" for v in preview)
        return f"EmbeddingVector(dim={len(self)}, values=[{preview_str}])"
