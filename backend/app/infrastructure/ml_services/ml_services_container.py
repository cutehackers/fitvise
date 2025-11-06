"""ML Services container for dependency injection.

Provides unified access to ML model instances (embeddings, etc.) for both
FastAPI endpoints and standalone scripts. Follows the same container pattern
as RepositoryContainer for consistency.

The container manages model lifecycle with lazy initialization and caching
to ensure models are created once per container instance.
"""

from typing import Optional

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.core.settings import Settings
from llama_index.core.embeddings import BaseEmbedding

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
except ImportError:
    HuggingFaceEmbedding = None  # type: ignore


class MLServicesError(Exception):
    """Raised when ML services initialization fails."""

    pass


class MLServicesContainer:
    """Container for ML service instances.

    This container provides:
    - Unified embedding model access for FastAPI and scripts
    - Lazy initialization of embedding models
    - Single instance per container (no redundant initialization)
    - Easy mocking for testing

    Usage in FastAPI/scripts:
        container = MLServicesContainer(settings)
        embed_model = container.embedding_model
        # Model is initialized on first access, cached for subsequent accesses

    Usage in tests:
        mock_embed = MockHuggingFaceEmbedding()
        container = MLServicesContainer(settings, embedding_model=mock_embed)
        # Mock is used instead of real model
    """

    def __init__(
        self,
        settings: Settings,
        embedding_model: Optional[BaseEmbedding] = None,
    ):
        """Initialize container with configuration.

        Args:
            settings: Application settings
            embedding_model: Optional pre-initialized embedding model for testing.
                           If not provided, will be lazily initialized on first access.

        Raises:
            MLServicesError: If required dependencies are missing
        """
        self.settings = settings

        # Allow injection of embedding model for testing
        self._embedding_model: Optional[BaseEmbedding] = embedding_model

        # Validate dependencies
        if HuggingFaceEmbedding is None:
            raise MLServicesError(
                "llama-index[huggingface] is required for embedding services. "
                "Install with: pip install llama-index llama-index-embeddings-huggingface"
            )

    @property
    def embedding_model(self) -> BaseEmbedding:
        """Get embedding model instance.

        Returns embedding model based on configuration. Lazily initializes
        on first access and caches for subsequent accesses.

        Returns:
            HuggingFaceEmbedding model instance ready for use

        Raises:
            MLServicesError: If model initialization fails
        """
        if self._embedding_model is None:
            try:
                config = EmbeddingModelConfig.default()
                self._embedding_model = HuggingFaceEmbedding(
                    model_name=config.model_name,
                    trust_remote_code=True,  # Required for Alibaba-NLP models
                )
            except Exception as exc:
                raise MLServicesError(
                    f"Failed to initialize embedding model: {str(exc)}"
                ) from exc

        return self._embedding_model
