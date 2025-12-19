from dependency_injector import containers, providers

from app.domain.services.context_service import ContextService, ContextBuildingConfig
from app.domain.services.context_window_manager import ContextWindowManager
from app.domain.services.data_source_scanner import DataSourceScanner
from app.domain.services.document_retrieval_service import DocumentRetrievalService
from app.domain.services.embedding_service import EmbeddingService
from app.domain.services.prompt_service import PromptService, PromptConfig
from app.domain.services.reranking_service import RerankingService
from app.domain.services.retrieval_service import RetrievalService
from app.domain.services.session_service import SessionService, SessionConfig


class DomainContainer(containers.DeclarativeContainer):
    """
    Domain Layer Container

    Contains domain services and business logic that can be easily instantiated.
    Complex domain services with infrastructure dependencies are handled
    in the application layer or through factories.
    """

    # CONFIGURATION
    context_config = providers.Singleton(ContextBuildingConfig)
    prompt_config = providers.Singleton(PromptConfig)
    session_config = providers.Singleton(SessionConfig)

    # DOMAIN SERVICES

    # Core context management
    context_service = providers.Singleton(ContextService, config=context_config)
    context_window_manager = providers.Singleton(ContextWindowManager)

    # Data source management
    data_source_scanner = providers.Singleton(DataSourceScanner)

    # Content processing services
    reranking_service = providers.Singleton(RerankingService)
    retrieval_service = providers.Singleton(RetrievalService)
    document_retrieval_service = providers.Singleton(DocumentRetrievalService)
    embedding_service = providers.Singleton(EmbeddingService)

    # User interaction services
    prompt_service = providers.Singleton(PromptService, config=prompt_config)
    session_service = providers.Singleton(SessionService, config=session_config)
