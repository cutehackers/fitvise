"""Sentence-Transformers embedding service (Task 2.2.1).

This module implements the concrete embedding service using Sentence-Transformers
with all-MiniLM-L6-v2 model, supporting batch and real-time embedding with
CPU optimization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
import numpy as np

from app.config.ml_models.embedding_model_configs import (
    DeviceType,
    EmbeddingModelConfig,
)
from app.domain.exceptions.embedding_exceptions import (
    EmbeddingGenerationError,
    ModelLoadError,
)
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.base_embedding_service import (
    BaseEmbeddingService,
)

logger = logging.getLogger(__name__)


class SentenceTransformerService(BaseEmbeddingService):
    """Sentence-Transformers embedding service implementation (Task 2.2.1).

    Implements embedding generation using Sentence-Transformers library
    with all-MiniLM-L6-v2 model (384 dimensions), optimized for CPU.

    Performance Targets:
        - Batch: ≥1000 chunks/minute
        - Real-time: <100ms latency
        - CPU-optimized with thread pooling

    Examples:
        >>> config = EmbeddingModelConfig.default()
        >>> service = SentenceTransformerService(config)
        >>> await service.initialize()
        >>> service.is_loaded
        True

        >>> # Single embedding
        >>> vector = await service.embed("Hello world")
        >>> len(vector)
        384

        >>> # Batch embedding
        >>> texts = ["text1", "text2", "text3"]
        >>> vectors = await service.embed_batch(texts)
        >>> len(vectors)
        3
        >>> all(len(v) == 384 for v in vectors)
        True
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        """Initialize Sentence-Transformers service.

        Args:
            config: Embedding model configuration
        """
        super().__init__(config)
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        """Resolve target compute device.

        Returns:
            Device string ("cpu", "cuda", "mps")
        """
        if self.config.device == DeviceType.AUTO:
            # Auto-detect best available device
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"

        return self.config.device.value

    async def initialize(self) -> None:
        """Initialize and load Sentence-Transformers model.

        Raises:
            ModelLoadError: If model loading fails
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_name} already loaded")
            return

        try:
            logger.info(
                f"Loading Sentence-Transformers model: {self.model_name} "
                f"on device: {self._device}"
            )

            # Import here to avoid loading at module level
            from sentence_transformers import SentenceTransformer

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.model_name,
                    device=self._device,
                    **self.config.model_kwargs,
                ),
            )

            # Verify model dimension
            test_embedding = self._model.encode(
                ["test"],
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            actual_dim = test_embedding.shape[1]

            if actual_dim != self.model_dimension:
                logger.warning(
                    f"Model dimension mismatch: expected {self.model_dimension}, "
                    f"got {actual_dim}. Updating config."
                )
                self.model_dimension = actual_dim

            self.is_loaded = True
            logger.info(
                f"Successfully loaded {self.model_name} "
                f"(dim={self.model_dimension}, device={self._device})"
            )

        except ImportError as e:
            raise ModelLoadError(
                message="sentence-transformers not installed",
                model_name=self.model_name,
                details=str(e),
            ) from e
        except Exception as e:
            raise ModelLoadError(
                message=f"Failed to load model {self.model_name}",
                model_name=self.model_name,
                details=str(e),
            ) from e

    async def embed(
        self,
        text: str,
        normalize: Optional[bool] = None,
    ) -> EmbeddingVector:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            normalize: Whether to L2-normalize (defaults to config)

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        self.validate_model_loaded()

        if not text or not text.strip():
            raise EmbeddingGenerationError(
                message="Cannot embed empty text",
                model_name=self.model_name,
            )

        try:
            should_normalize = (
                normalize if normalize is not None else self.config.normalize_embeddings
            )

            # Run encoding in thread pool
            loop = asyncio.get_event_loop()
            embedding_array = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    [text],
                    convert_to_numpy=True,
                    normalize_embeddings=should_normalize,
                    batch_size=1,
                    show_progress_bar=False,
                    **self.config.encode_kwargs,
                ),
            )

            # Extract single embedding
            vector_array = embedding_array[0].astype(np.float32)
            vector = EmbeddingVector.from_numpy(vector_array)

            self._stats["total_embeddings"] += 1
            return vector

        except Exception as e:
            self._stats["errors"] += 1
            raise EmbeddingGenerationError(
                message=f"Failed to embed text: {text[:50]}...",
                model_name=self.model_name,
                details=str(e),
            ) from e

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to embed
            batch_size: Batch size (defaults to config)
            show_progress: Show progress bar (defaults to config)
            normalize: Whether to L2-normalize (defaults to config)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingGenerationError: If batch embedding fails
        """
        self.validate_model_loaded()

        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise EmbeddingGenerationError(
                message="All texts are empty",
                model_name=self.model_name,
            )

        try:
            effective_batch_size = batch_size or self.config.batch_size
            should_show_progress = (
                show_progress if show_progress is not None else self.config.show_progress
            )
            should_normalize = (
                normalize if normalize is not None else self.config.normalize_embeddings
            )

            logger.info(
                f"Embedding {len(valid_texts)} texts in batches of {effective_batch_size}"
            )

            # Run batch encoding in thread pool
            loop = asyncio.get_event_loop()
            embeddings_array = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    valid_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=should_normalize,
                    batch_size=effective_batch_size,
                    show_progress_bar=should_show_progress,
                    **self.config.encode_kwargs,
                ),
            )

            # Convert to EmbeddingVector objects
            vectors = [
                EmbeddingVector.from_numpy(arr.astype(np.float32))
                for arr in embeddings_array
            ]

            self._stats["total_embeddings"] += len(vectors)
            logger.info(f"Successfully embedded {len(vectors)} texts")
            return vectors

        except Exception as e:
            self._stats["errors"] += 1
            raise EmbeddingGenerationError(
                message=f"Failed to embed batch of {len(texts)} texts",
                model_name=self.model_name,
                details=str(e),
            ) from e

    async def shutdown(self) -> None:
        """Shutdown service and release resources."""
        if self._model is not None:
            # Clear model from memory
            del self._model
            self._model = None
            self.is_loaded = False
            logger.info(f"Shutdown {self.model_name} service")

        # Clear cache
        self.clear_cache()


def create_sentence_transformer_service(
    config: Optional[EmbeddingModelConfig] = None,
) -> SentenceTransformerService:
    """Create Sentence-Transformers embedding service.

    Args:
        config: Optional embedding model configuration

    Returns:
        SentenceTransformerService instance
    """
    if config is None:
        config = EmbeddingModelConfig.default()

    return SentenceTransformerService(config)
