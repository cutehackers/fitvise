"""FastAPI dependencies for service injection.

Provides dependency injection functions for services to be used in FastAPI endpoints.
"""

from fastapi import Depends

from app.domain.services.context_service import ContextService
from app.domain.services.document_retrieval_service import DocumentRetrievalService
from app.domain.services.embedding_service import EmbeddingService
from app.domain.services.retrieval_service import RetrievalService
from app.domain.services.session_service import SessionService
from app.infrastructure.dependencies.service_container import ServiceContainer
from app.infrastructure.persistence.repositories.dependencies import get_repository_container


def get_service_container(
    repository_container=Depends(get_repository_container),
) -> ServiceContainer:
    """Get service container instance.

    Args:
        repository_container: Repository container dependency

    Returns:
        ServiceContainer instance
    """
    from app.core.settings import Settings

    settings = Settings()
    return ServiceContainer(settings, repository_container)


def get_context_service(
    service_container: ServiceContainer = Depends(get_service_container),
) -> ContextService:
    """Get context service instance.

    Args:
        service_container: Service container dependency

    Returns:
        ContextService instance
    """
    return service_container.context_service


def get_document_retrieval_service(
    service_container: ServiceContainer = Depends(get_service_container),
) -> DocumentRetrievalService:
    """Get document retrieval service instance.

    Args:
        service_container: Service container dependency

    Returns:
        DocumentRetrievalService instance
    """
    return service_container.document_retrieval_service


def get_embedding_service(
    service_container: ServiceContainer = Depends(get_service_container),
) -> EmbeddingService:
    """Get embedding service instance.

    Args:
        service_container: Service container dependency

    Returns:
        EmbeddingService instance
    """
    return service_container.embedding_service


def get_retrieval_service(
    service_container: ServiceContainer = Depends(get_service_container),
) -> RetrievalService:
    """Get retrieval service instance.

    Args:
        service_container: Service container dependency

    Returns:
        RetrievalService instance
    """
    return service_container.retrieval_service


def get_session_service(
    service_container: ServiceContainer = Depends(get_service_container),
) -> SessionService:
    """Get session service instance.

    Args:
        service_container: Service container dependency

    Returns:
        SessionService instance
    """
    return service_container.session_service


# Convenience dependencies for common service combinations
def get_search_services(
    service_container: ServiceContainer = Depends(get_service_container),
) -> tuple[EmbeddingService, RetrievalService]:
    """Get embedding and retrieval services for search operations.

    Args:
        service_container: Service container dependency

    Returns:
        Tuple of (EmbeddingService, RetrievalService)
    """
    return service_container.embedding_service, service_container.retrieval_service


def get_context_services(
    service_container: ServiceContainer = Depends(get_service_container),
) -> tuple[RetrievalService, ContextService]:
    """Get retrieval and context services for document retrieval.

    Args:
        service_container: Service container dependency

    Returns:
        Tuple of (RetrievalService, ContextService)
    """
    return service_container.retrieval_service, service_container.context_service