from dependency_injector import containers, providers

from app.core.settings import Settings, settings
from app.di.containers.infra_container import InfraContainer
from app.di.containers.domain_container import DomainContainer


class AppContainer(containers.DeclarativeContainer):
    """
    Application Layer Container

    Main entry point for dependency injection. Orchestrates infrastructure
    and domain layers for application use cases.

    Layer Dependencies:
    - InfraContainer: Infrastructure services (database, external APIs, repositories)
    - DomainContainer: Domain services (business logic, domain entities)
    """

    # SETTINGS
    settings: providers.Singleton[Settings] = providers.Singleton(Settings)
    configs = providers.Configuration()
    configs.from_pydantic(settings())

    # LAYERED CONTAINERS
    infra = providers.Container(InfraContainer, configs=configs)
    domain = providers.Container(DomainContainer)

    # EXPOSED SERVICES (for easy access from application layer)
    # Infrastructure services
    db_session = infra.db_session
    weaviate_client = infra.weaviate_client
    llama_index_retriever = infra.llama_index_retriever
    document_repository = infra.document_repository
    data_source_repository = infra.data_source_repository
    embedding_repository = infra.embedding_repository
    sentence_transformer_service = infra.sentence_transformer_service

    # Domain services
    context_service = domain.context_service
    context_window_manager = domain.context_window_manager
    data_source_scanner = domain.data_source_scanner
    retrieval_service = domain.retrieval_service
    document_retrieval_service = domain.document_retrieval_service
    embedding_service = domain.embedding_service
    reranking_service = domain.reranking_service
    prompt_service = domain.prompt_service
    session_service = domain.session_service
