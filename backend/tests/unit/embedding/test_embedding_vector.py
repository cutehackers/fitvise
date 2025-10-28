"""Unit tests for EmbeddingVector value object (Task 2.2.1).

Tests cover:
- Initialization and validation
- Immutability guarantees
- Similarity operations (cosine, dot product, euclidean)
- Normalization and vector operations
- Serialization/deserialization
"""

import numpy as np
import pytest

from app.domain.value_objects.embedding_vector import EmbeddingVector


class TestEmbeddingVectorInitialization:
    """Test EmbeddingVector initialization and validation."""

    def test_create_from_numpy_array(self):
        """Test creating vector from numpy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vector = EmbeddingVector(arr)

        assert vector.dimension == 3
        assert np.allclose(vector.to_numpy(), arr)

    def test_create_from_list(self):
        """Test creating vector from list using factory method."""
        values = [1.0, 2.0, 3.0]
        vector = EmbeddingVector.from_list(values)

        assert vector.dimension == 3
        assert np.allclose(vector.to_list(), values)

    def test_create_from_numpy_factory(self):
        """Test creating vector from numpy array using factory method."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vector = EmbeddingVector.from_numpy(arr)

        assert vector.dimension == 3
        assert np.allclose(vector.to_numpy(), arr)

    def test_create_empty_vector_raises_error(self):
        """Test that empty vector raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            EmbeddingVector.from_list([])

    def test_create_invalid_dimension_raises_error(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError):
            EmbeddingVector.from_list([[1, 2], [3, 4]])  # 2D array


class TestEmbeddingVectorImmutability:
    """Test that EmbeddingVector is truly immutable."""

    def test_vector_data_immutable(self):
        """Test that internal vector data cannot be modified."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        arr = vector.to_numpy()

        # Attempt to modify should not affect original
        arr[0] = 999.0

        assert vector.to_numpy()[0] == 1.0  # Original unchanged

    def test_internal_array_is_readonly(self):
        """Test that internal array is marked as read-only."""
        vector = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        # Access internal array through __dict__ or _vector
        internal_arr = vector._vector

        assert not internal_arr.flags.writeable


class TestEmbeddingVectorSimilarity:
    """Test similarity and distance operations."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        v1 = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        v2 = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        similarity = v1.cosine_similarity(v2)

        assert pytest.approx(similarity, abs=1e-6) == 1.0

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        v1 = EmbeddingVector.from_list([1.0, 0.0, 0.0])
        v2 = EmbeddingVector.from_list([0.0, 1.0, 0.0])

        similarity = v1.cosine_similarity(v2)

        assert pytest.approx(similarity, abs=1e-6) == 0.0

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        v1 = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        v2 = EmbeddingVector.from_list([-1.0, -2.0, -3.0])

        similarity = v1.cosine_similarity(v2)

        assert pytest.approx(similarity, abs=1e-6) == -1.0

    def test_cosine_similarity_normalized_vectors(self):
        """Test cosine similarity with pre-normalized vectors."""
        # Unit vectors at 60 degrees (cos(60°) = 0.5)
        v1 = EmbeddingVector.from_list([1.0, 0.0, 0.0])
        v2 = EmbeddingVector.from_list([0.5, 0.866, 0.0])

        similarity = v1.cosine_similarity(v2)

        assert pytest.approx(similarity, abs=1e-3) == 0.5

    def test_dot_product(self):
        """Test dot product calculation."""
        v1 = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        v2 = EmbeddingVector.from_list([4.0, 5.0, 6.0])

        dot = v1.dot_product(v2)

        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert pytest.approx(dot) == 32.0

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        v1 = EmbeddingVector.from_list([0.0, 0.0, 0.0])
        v2 = EmbeddingVector.from_list([3.0, 4.0, 0.0])

        distance = v1.euclidean_distance(v2)

        # sqrt(3^2 + 4^2) = 5.0
        assert pytest.approx(distance) == 5.0

    def test_similarity_different_dimensions_raises_error(self):
        """Test that comparing vectors of different dimensions raises error."""
        v1 = EmbeddingVector.from_list([1.0, 2.0])
        v2 = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="dimension"):
            v1.cosine_similarity(v2)


class TestEmbeddingVectorOperations:
    """Test vector mathematical operations."""

    def test_normalize_unit_vector(self):
        """Test normalizing a unit vector returns identical vector."""
        v = EmbeddingVector.from_list([1.0, 0.0, 0.0])
        normalized = v.normalize()

        assert pytest.approx(normalized.norm()) == 1.0
        assert np.allclose(normalized.to_numpy(), v.to_numpy())

    def test_normalize_arbitrary_vector(self):
        """Test normalizing arbitrary vector produces unit vector."""
        v = EmbeddingVector.from_list([3.0, 4.0, 0.0])
        normalized = v.normalize()

        assert pytest.approx(normalized.norm(), abs=1e-6) == 1.0
        # Direction preserved: [3,4,0] -> [0.6, 0.8, 0]
        assert pytest.approx(normalized.to_numpy()[0]) == 0.6
        assert pytest.approx(normalized.to_numpy()[1]) == 0.8

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero vector."""
        v = EmbeddingVector.from_list([0.0, 0.0, 0.0])
        normalized = v.normalize()

        assert np.allclose(normalized.to_numpy(), [0.0, 0.0, 0.0])

    def test_norm_calculation(self):
        """Test L2 norm calculation."""
        v = EmbeddingVector.from_list([3.0, 4.0, 0.0])

        norm = v.norm()

        assert pytest.approx(norm) == 5.0

    def test_norm_zero_vector(self):
        """Test norm of zero vector is 0."""
        v = EmbeddingVector.from_list([0.0, 0.0, 0.0])

        norm = v.norm()

        assert pytest.approx(norm) == 0.0


class TestEmbeddingVectorSerialization:
    """Test serialization and deserialization."""

    def test_to_list_conversion(self):
        """Test converting vector to list."""
        values = [1.0, 2.0, 3.0]
        vector = EmbeddingVector.from_list(values)

        result = vector.to_list()

        assert result == values
        assert isinstance(result, list)

    def test_to_numpy_conversion(self):
        """Test converting vector to numpy array."""
        values = [1.0, 2.0, 3.0]
        vector = EmbeddingVector.from_list(values)

        result = vector.to_numpy()

        assert np.allclose(result, values)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_round_trip_list(self):
        """Test round-trip conversion list -> vector -> list."""
        original = [1.0, 2.0, 3.0]
        vector = EmbeddingVector.from_list(original)
        result = vector.to_list()

        assert result == original

    def test_round_trip_numpy(self):
        """Test round-trip conversion numpy -> vector -> numpy."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vector = EmbeddingVector.from_numpy(original)
        result = vector.to_numpy()

        assert np.allclose(result, original)


class TestEmbeddingVectorEquality:
    """Test equality and comparison operations."""

    def test_equality_identical_vectors(self):
        """Test that identical vectors are equal."""
        v1 = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        v2 = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        assert v1 == v2

    def test_inequality_different_vectors(self):
        """Test that different vectors are not equal."""
        v1 = EmbeddingVector.from_list([1.0, 2.0, 3.0])
        v2 = EmbeddingVector.from_list([1.0, 2.0, 4.0])

        assert v1 != v2

    def test_equality_different_dimensions(self):
        """Test that vectors of different dimensions are not equal."""
        v1 = EmbeddingVector.from_list([1.0, 2.0])
        v2 = EmbeddingVector.from_list([1.0, 2.0, 3.0])

        assert v1 != v2


class TestEmbeddingVectorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_vectors(self):
        """Test handling of large dimensional vectors (e.g., 4096)."""
        size = 4096
        values = np.random.rand(size).astype(np.float32)
        vector = EmbeddingVector.from_numpy(values)

        assert vector.dimension == size
        assert np.allclose(vector.to_numpy(), values)

    def test_very_small_values(self):
        """Test handling of very small floating point values."""
        values = [1e-10, 2e-10, 3e-10]
        vector = EmbeddingVector.from_list(values)

        assert vector.dimension == 3
        # Normalization should work with small values
        normalized = vector.normalize()
        assert pytest.approx(normalized.norm(), abs=1e-6) == 1.0

    def test_mixed_positive_negative_values(self):
        """Test handling of mixed positive/negative values."""
        values = [-1.0, 2.0, -3.0, 4.0]
        vector = EmbeddingVector.from_list(values)

        assert vector.dimension == 4
        assert np.allclose(vector.to_list(), values)

    def test_single_dimension_vector(self):
        """Test handling of 1D vector."""
        vector = EmbeddingVector.from_list([5.0])

        assert vector.dimension == 1
        assert vector.norm() == 5.0


class TestEmbeddingVectorPerformance:
    """Test performance characteristics (not strict assertions)."""

    def test_large_batch_similarity(self):
        """Test similarity calculation performance with realistic dimensions."""
        # Simulate all-MiniLM-L6-v2 dimension (384)
        v1 = EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))
        v2 = EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))

        # Should complete quickly
        similarity = v1.cosine_similarity(v2)

        assert -1.0 <= similarity <= 1.0

    def test_normalization_performance(self):
        """Test normalization performance with realistic dimensions."""
        vector = EmbeddingVector.from_numpy(np.random.rand(384).astype(np.float32))

        # Should complete quickly
        normalized = vector.normalize()

        assert pytest.approx(normalized.norm(), abs=1e-6) == 1.0
