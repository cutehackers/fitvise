"""Service container for dependency injection.

Provides unified service access for both FastAPI endpoints and standalone scripts.
The container manages service lifecycle and configuration-based instantiation.
"""
from typing import Optional

from app.core.settings import Settings
from app.domain.services.context_service import ContextService
from app.domain.services.document_retrieval_service import DocumentRetrievalService
from app.domain.services.embedding_service import EmbeddingService
from app.domain.services.retrieval_service import RetrievalService
from app.domain.services.session_service import SessionService
from app.infrastructure.persistence.repositories.container import RepositoryContainer


class ServiceContainer:
    """Container for service instances.

    This container provides:
    - Unified service access for FastAPI and scripts
    - Configuration-driven service instantiation
    - Lazy initialization of services
    - Service dependency management

    Usage in FastAPI:
        container = ServiceContainer(settings, repository_container)
        service = container.embedding_service

    Usage in scripts:
        settings = Settings()
        repo_container = RepositoryContainer(settings)
        container = ServiceContainer(settings, repo_container)
        service = container.embedding_service
    """

    def __init__(
        self,
        settings: Settings,
        repository_container: Optional[RepositoryContainer] = None,
    ):
        """Initialize container with configuration.

        Args:
            settings: Application settings
            repository_container: Optional repository container
        """
        self.settings = settings
        self._repository_container = repository_container

        # Create repository container if not provided
        if self._repository_container is None:
            self._repository_container = RepositoryContainer(settings)

        # Lazy-initialized services
        self._context_service: Optional[ContextService] = None
        self._document_retrieval_service: Optional[DocumentRetrievalService] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._retrieval_service: Optional[RetrievalService] = None
        self._session_service: Optional[SessionService] = None
        self._sentence_transformer_service = None  # Will be initialized when needed

    @property
    def repository_container(self) -> RepositoryContainer:
        """Get repository container instance.

        Returns:
            RepositoryContainer instance
        """
        return self._repository_container

    @property
    def context_service(self) -> ContextService:
        """Get context service instance.

        Returns:
            ContextService implementation
        """
        if self._context_service is None:
            self._context_service = ContextService()

        return self._context_service

    @property
    def document_retrieval_service(self) -> DocumentRetrievalService:
        """Get document retrieval service instance.

        Returns:
            DocumentRetrievalService implementation
        """
        if self._document_retrieval_service is None:
            self._document_retrieval_service = DocumentRetrievalService(
                retrieval_service=self.retrieval_service,
                context_service=self.context_service,
            )

        return self._document_retrieval_service

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get embedding service instance.

        Returns:
            EmbeddingService implementation
        """
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                repository=self._repository_container.embedding_repository,
                sentence_transformer_service=self.sentence_transformer_service,
            )

        return self._embedding_service

    @property
    def retrieval_service(self) -> RetrievalService:
        """Get retrieval service instance.

        Returns:
            RetrievalService implementation
        """
        if self._retrieval_service is None:
            self._retrieval_service = RetrievalService(
                embedding_service=self.embedding_service,
                search_repository=self._repository_container.search_repository,
            )

        return self._retrieval_service

    @property
    def session_service(self) -> SessionService:
        """Get session service instance.

        Returns:
            SessionService implementation
        """
        if self._session_service is None:
            self._session_service = SessionService(
                session_repository=self._repository_container.session_repository,
            )

        return self._session_service

    @property
    def sentence_transformer_service(self):
        """Get sentence transformer service instance.

        Returns:
            Sentence transformer service implementation
        """
        if self._sentence_transformer_service is None:
            # Import here to avoid circular imports
            from app.infrastructure.external_services.sentence_transformer_service import (
                SentenceTransformerService,
            )

            self._sentence_transformer_service = SentenceTransformerService(
                model_name=self.settings.embedding_model_name,
                device=self.settings.embedding_device,
            )

        return self._sentence_transformer_service

    def get_service_status(self) -> dict:
        """Get status of all services.

        Returns:
            Dictionary with service status information
        """
        return {
            "context_service": "initialized" if self._context_service else "not_initialized",
            "document_retrieval_service": "initialized" if self._document_retrieval_service else "not_initialized",
            "embedding_service": "initialized" if self._embedding_service else "not_initialized",
            "retrieval_service": "initialized" if self._retrieval_service else "not_initialized",
            "session_service": "initialized" if self._session_service else "not_initialized",
            "sentence_transformer_service": "initialized" if self._sentence_transformer_service else "not_initialized",
            "repository_container": "initialized" if self._repository_container else "not_initialized",
        }

    async def health_check(self) -> dict:
        """Perform health check on all services.

        Returns:
            Dictionary with health check results
        """
        health_status = {
            "overall": "healthy",
            "services": {},
        }

        try:
            # Check embedding service
            if self._embedding_service:
                embedding_healthy = await self._embedding_service.health_check()
                health_status["services"]["embedding_service"] = "healthy" if embedding_healthy else "unhealthy"
            else:
                health_status["services"]["embedding_service"] = "not_initialized"

            # Check retrieval service (via embedding service health)
            if self._retrieval_service:
                retrieval_metrics = await self._retrieval_service.get_retrieval_metrics()
                retrieval_healthy = retrieval_metrics.get("overall_status") == "healthy"
                health_status["services"]["retrieval_service"] = "healthy" if retrieval_healthy else "unhealthy"
            else:
                health_status["services"]["retrieval_service"] = "not_initialized"

            # Check repository container
            if self._repository_container:
                # Repository container doesn't have health check, but we can check if repositories exist
                health_status["services"]["repository_container"] = "initialized"
            else:
                health_status["services"]["repository_container"] = "not_initialized"

            # Determine overall health
            service_healths = list(health_status["services"].values())
            if any(status in ["unhealthy", "not_initialized"] for status in service_healths):
                health_status["overall"] = "degraded"

        except Exception as e:
            health_status["overall"] = "error"
            health_status["error"] = str(e)

        return health_status