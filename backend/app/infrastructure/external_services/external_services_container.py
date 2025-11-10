"""External Services container for dependency injection.

Provides unified access to external service instances (Weaviate, embedding services) for both
FastAPI endpoints and standalone scripts.

The container manages service lifecycle with lazy initialization and caching
to ensure services are created once per container instance.
"""

from typing import Optional

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.core.settings import Settings
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.services.embedding_service import EmbeddingService
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.infrastructure.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)
from llama_index.core.embeddings import BaseEmbedding

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
except ImportError:
    HuggingFaceEmbedding = None  # type: ignore


class ExternalServicesError(Exception):
    """Raised when external services initialization fails."""
    pass


class ExternalServicesContainer:
    """Container for all external service instances.

    This container provides:
    - Unified access to all external services (ML models, Weaviate, embedding services) for FastAPI and scripts
    - Lazy initialization of services with caching
    - Single instance per container (no redundant initialization)
    - Easy mocking for testing
    - Integrated ML services and external services management

    Usage in FastAPI/scripts:
        container = ExternalServicesContainer(settings)
        embedding_model = container.embedding_model
        embedding_service = container.sentence_transformer_service
        embedding_repository = container.embedding_repository
        domain_service = container.embedding_domain_service
        # Services are initialized on first access, cached for subsequent accesses

    Usage in tests:
        mock_embed = MockHuggingFaceEmbedding()
        container = ExternalServicesContainer(settings, embedding_model=mock_embed)
        # Mock is used instead of real service
    """

    def __init__(
        self,
        settings: Settings,
        embedding_model: Optional[BaseEmbedding] = None,
        sentence_transformer_service: Optional[SentenceTransformerService] = None,
        embedding_repository: Optional[EmbeddingRepository] = None,
        embedding_domain_service: Optional[EmbeddingService] = None,
    ):
        """Initialize container with configuration.

        Args:
            settings: Application settings
            embedding_model: Optional pre-initialized embedding model for testing.
            sentence_transformer_service: Optional pre-initialized sentence transformer service for testing.
            embedding_repository: Optional pre-initialized embedding repository for testing.
            embedding_domain_service: Optional pre-initialized embedding domain service for testing.

        Raises:
            ExternalServicesError: If required dependencies are missing
        """
        self.settings = settings

        # Allow injection of services for testing
        self._embedding_model: Optional[BaseEmbedding] = embedding_model
        self._sentence_transformer_service: Optional[SentenceTransformerService] = sentence_transformer_service
        self._embedding_repository: Optional[EmbeddingRepository] = embedding_repository
        self._embedding_domain_service: Optional[EmbeddingService] = embedding_domain_service
        self._weaviate_client: Optional[WeaviateClient] = None

        # Validate dependencies
        if HuggingFaceEmbedding is None:
            raise ExternalServicesError(
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
            ExternalServicesError: If model initialization fails
        """
        if self._embedding_model is None:
            try:
                config = EmbeddingModelConfig.default()
                self._embedding_model = HuggingFaceEmbedding(
                    model_name=config.model_name,
                    trust_remote_code=True,  # Required for Alibaba-NLP models
                )
            except Exception as exc:
                raise ExternalServicesError(
                    f"Failed to initialize embedding model: {str(exc)}"
                ) from exc

        return self._embedding_model

    @property
    def sentence_transformer_service(self) -> SentenceTransformerService:
        """Get sentence transformer service instance.

        Returns a configured sentence transformer service. Lazily initializes
        on first access and caches for subsequent accesses.

        WARNING: The returned service requires async initialization before use.
        Callers must call `await service.initialize()` before using embedding
        methods. See RagEmbeddingTask.execute() for the initialization pattern.

        Returns:
            Configured SentenceTransformerService instance (requires initialization)

        Raises:
            ExternalServicesError: If service initialization fails
        """
        if self._sentence_transformer_service is None:
            try:
                config = EmbeddingModelConfig.for_realtime()
                self._sentence_transformer_service = SentenceTransformerService(config)
                # Note: The service's initialize() method must be called asynchronously
                # by the caller before using embedding methods. This property only
                # creates the service instance. See RagEmbeddingTask.execute() for
                # the proper initialization pattern.
            except Exception as exc:
                raise ExternalServicesError(
                    f"Failed to initialize sentence transformer service: {str(exc)}"
                ) from exc

        return self._sentence_transformer_service

    @property
    def weaviate_client(self) -> WeaviateClient:
        """Get Weaviate client instance.

        Returns a configured Weaviate client. Lazily initializes
        on first access and caches for subsequent accesses.

        Returns:
            Configured WeaviateClient instance ready for use

        Raises:
            ExternalServicesError: If client initialization fails
        """
        if self._weaviate_client is None:
            try:
                config = WeaviateConfig()
                self._weaviate_client = WeaviateClient(config)
            except Exception as exc:
                raise ExternalServicesError(
                    f"Failed to initialize Weaviate client: {str(exc)}"
                ) from exc

        return self._weaviate_client

    @property
    def embedding_repository(self) -> EmbeddingRepository:
        """Get embedding repository instance.

        Returns a configured embedding repository. Lazily initializes
        on first access and caches for subsequent accesses.

        Returns:
            Configured EmbeddingRepository instance ready for use

        Raises:
            ExternalServicesError: If repository initialization fails
        """
        if self._embedding_repository is None:
            try:
                client = self.weaviate_client
                self._embedding_repository = WeaviateEmbeddingRepository(client)
            except Exception as exc:
                raise ExternalServicesError(
                    f"Failed to initialize embedding repository: {str(exc)}"
                ) from exc

        return self._embedding_repository

    @property
    def embedding_domain_service(self) -> EmbeddingService:
        """Get embedding domain service instance.

        Returns a configured embedding domain service. Lazily initializes
        on first access and caches for subsequent accesses.

        Returns:
            Configured EmbeddingService instance ready for use

        Raises:
            ExternalServicesError: If service initialization fails
        """
        if self._embedding_domain_service is None:
            try:
                repository = self.embedding_repository
                self._embedding_domain_service = EmbeddingService(repository)
            except Exception as exc:
                raise ExternalServicesError(
                    f"Failed to initialize embedding domain service: {str(exc)}"
                ) from exc

        return self._embedding_domain_service