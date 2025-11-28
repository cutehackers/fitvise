"""Embedding service interface.

This module contains the EmbeddingService abstract interface that defines
the contract for embedding generation and management operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.domain.value_objects.vector_embedding import VectorEmbedding


class EmbeddingService(ABC):
    """Abstract service interface for embedding generation operations.

    This interface defines the contract for embedding generation services
    including text processing, model management, and caching operations.
    """

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VectorEmbedding:
        """Generate embedding for a single text.

        Args:
            text: Text to generate embedding for
            model_name: Optional model name to use
            metadata: Optional metadata for the embedding

        Returns:
            Generated vector embedding

        Raises:
            RetrievalError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorEmbedding]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to generate embeddings for
            model_name: Optional model name to use
            metadata: Optional metadata for the embeddings

        Returns:
            List of generated vector embeddings

        Raises:
            RetrievalError: If batch embedding generation fails
        """
        pass

    @abstractmethod
    async def get_model_info(
        self,
        model_name: str,
    ) -> Dict[str, Any]:
        """Get information about an embedding model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information

        Raises:
            RetrievalError: If model info lookup fails
        """
        pass

    @abstractmethod
    async def list_available_models(self) -> List[str]:
        """List all available embedding models.

        Returns:
            List of model names

        Raises:
            RetrievalError: If model listing fails
        """
        pass

    @abstractmethod
    async def validate_model(
        self,
        model_name: str,
    ) -> bool:
        """Validate if a model is available and functional.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model is valid, False otherwise

        Raises:
            RetrievalError: If validation check fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the embedding service.

        Returns:
            Dictionary with health status information

        Raises:
            RetrievalError: If health check fails
        """
        pass

    def get_default_model(self) -> str:
        """Get the default embedding model name."""
        return "default"

    def get_max_text_length(self) -> int:
        """Get maximum text length for embedding generation."""
        return 8192

    def get_max_batch_size(self) -> int:
        """Get maximum batch size for operations."""
        return 100

    def supports_batch_operations(self) -> bool:
        """Check if this service supports batch operations."""
        return True

    def supports_caching(self) -> bool:
        """Check if this service supports caching."""
        return False
