"""Service Providers.

This module provides service providers for the DI container.
It handles business logic services, use cases, and application services.
"""

from dependency_injector import containers, providers

from app.application.use_cases.data_ingestion.connect_databases import ConnectDatabasesUseCase
from app.application.use_cases.data_ingestion.integrate_tika import IntegrateTikaUseCase
from app.application.use_cases.data_ingestion.setup_airflow import SetupAirflowUseCase
from app.application.use_cases.data_ingestion.setup_web_scraping import SetupWebScrapingUseCase
from app.application.use_cases.embedding import (
    BatchEmbedUseCase,
    EmbedDocumentChunksUseCase,
    EmbedQueryUseCase,
    SearchEmbeddingsUseCase,
    SetupEmbeddingInfrastructureUseCase,
)
from app.application.use_cases.knowledge_sources.audit_data_sources import (
    AuditDataSourcesUseCase,
)
from app.application.use_cases.knowledge_sources.categorize_sources import (
    CategorizeSourcesUseCase,
)
from app.application.use_cases.knowledge_sources.document_external_apis import (
    DocumentExternalApisUseCase,
)
from app.application.use_cases.llm import PromptUseCase
from app.application.use_cases.retrieval.semantic_search import SemanticSearchUseCase
from app.application.use_cases.storage_management import SetupObjectStorageUseCase
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.repositories.search_repository import SearchRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.services.llm_service import LLMService
from app.domain.services.retrieval_service import RetrievalService
from app.infrastructure.llm.services.ollama_service import OllamaService
from app.infrastructure.external_services.ml_services.categorization.sklearn_categorizer import (
    SklearnDocumentCategorizer,
)
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)


class ServiceProviders(containers.DeclarativeContainer):
    """Service providers for the FitVise application.

    Provides access to business logic services and use cases with proper
    dependency injection and lifecycle management.
    """

    # Configuration dependencies
    config = providers.Dependency()
    repositories = providers.Dependency()
    external_services = providers.Dependency()

    # Core service instances
    llm_service = providers.Singleton(
        OllamaService,
        base_url=config.settings.llm_base_url,
        model=config.settings.llm_model,
        timeout=config.settings.llm_timeout,
        temperature=config.settings.llm_temperature,
        max_tokens=config.settings.llm_max_tokens,
    )

    embedding_domain_service = providers.Singleton(
        EmbeddingService,
        repository=repositories.embedding_repo_interface,
    )

    # Use case providers
    prompt_use_case = providers.Factory(
        PromptUseCase,
        llm_service=llm_service,
    )

    embed_chunks_use_case = providers.Factory(
        EmbedDocumentChunksUseCase,
        embedding_service=external_services.sentence_transformer_service,
        embedding_repository=repositories.embedding_repo_interface,
        domain_service=embedding_domain_service,
    )

    embed_query_use_case = providers.Factory(
        EmbedQueryUseCase,
        embedding_service=external_services.sentence_transformer_service,
        embedding_repository=repositories.embedding_repo_interface,
        domain_service=embedding_domain_service,
    )

    batch_embed_use_case = providers.Factory(
        BatchEmbedUseCase,
        embedding_service=external_services.sentence_transformer_service,
        embedding_repository=repositories.embedding_repo_interface,
        domain_service=embedding_domain_service,
    )

    search_embeddings_use_case = providers.Factory(
        SearchEmbeddingsUseCase,
        embedding_service=external_services.sentence_transformer_service,
        embedding_repository=repositories.embedding_repo_interface,
        domain_service=embedding_domain_service,
    )

    setup_embedding_infrastructure_use_case = providers.Factory(
        SetupEmbeddingInfrastructureUseCase,
        embedding_service=external_services.sentence_transformer_service,
        weaviate_client=external_services.weaviate_client,
        embedding_config=config.realtime_embedding_config,
        weaviate_config=config.weaviate_config,
    )

    retrieval_service = providers.Factory(
        RetrievalService,
        embedding_service=external_services.sentence_transformer_service,
        search_repository=repositories.search_repo_interface,
    )

    semantic_search_use_case = providers.Factory(
        SemanticSearchUseCase,
        retrieval_service=retrieval_service,
    )

    categorizer = providers.Singleton(SklearnDocumentCategorizer)

    audit_data_sources_use_case = providers.Factory(
        AuditDataSourcesUseCase,
        data_source_repository=repositories.data_source_repo_interface,
    )

    document_external_apis_use_case = providers.Factory(
        DocumentExternalApisUseCase,
        data_source_repository=repositories.data_source_repo_interface,
    )

    categorize_sources_use_case = providers.Factory(
        CategorizeSourcesUseCase,
        data_source_repository=repositories.data_source_repo_interface,
        categorizer=categorizer,
    )

    setup_object_storage_use_case = providers.Factory(
        SetupObjectStorageUseCase,
    )

    setup_airflow_use_case = providers.Factory(
        SetupAirflowUseCase,
    )

    integrate_tika_use_case = providers.Factory(
        IntegrateTikaUseCase,
    )

    connect_databases_use_case = providers.Factory(
        ConnectDatabasesUseCase,
    )

    setup_web_scraping_use_case = providers.Factory(
        SetupWebScrapingUseCase,
    )

    # Pipeline service providers
    def _pipeline_repositories(
        document_repo: DocumentRepository,
        data_source_repo: DataSourceRepository,
        embedding_repo: EmbeddingRepository,
    ) -> "RepositoryBundle":
        from app.pipeline.workflow import RepositoryBundle

        return RepositoryBundle(
            document_repository=document_repo,
            data_source_repository=data_source_repo,
            embedding_repository=embedding_repo,
        )

    pipeline_repositories = providers.Factory(
        _pipeline_repositories,
        document_repo=repositories.document_repo_interface,
        data_source_repo=repositories.data_source_repo_interface,
        embedding_repo=repositories.embedding_repo_interface,
    )

    def _pipeline_services(
        embedding_service: SentenceTransformerService,
        embedding_model,
        weaviate_client: WeaviateClient,
        setup_use_case: SetupEmbeddingInfrastructureUseCase,
    ) -> "PipelineServices":
        from app.pipeline.workflow import PipelineServices

        return PipelineServices(
            embedding_service=embedding_service,
            embedding_model=embedding_model,
            weaviate_client=weaviate_client,
            setup_use_case=setup_use_case,
        )

    pipeline_services = providers.Factory(
        _pipeline_services,
        embedding_service=external_services.sentence_transformer_service,
        embedding_model=external_services.llama_index_embedding,
        weaviate_client=external_services.weaviate_client,
        setup_use_case=setup_embedding_infrastructure_use_case,
    )

    def _pipeline_workflow(
        repositories_bundle,
        services_bundle,
        verbose: bool,
    ) -> "RAGWorkflow":
        from app.pipeline.workflow import RAGWorkflow

        return RAGWorkflow(
            repositories=repositories_bundle,
            services=services_bundle,
            session=None,
            verbose=verbose,
        )

    pipeline_workflow = providers.Factory(
        _pipeline_workflow,
        repositories_bundle=pipeline_repositories,
        services_bundle=pipeline_services,
        verbose=config.debug_enabled,
    )

    # Service initialization providers
    async def _init_llm_service(service: OllamaService) -> OllamaService:
        await service.initialize()
        return service

    init_llm_service = providers.Resource(_init_llm_service, service=llm_service)

    async def _init_embedding_service(service: SentenceTransformerService) -> SentenceTransformerService:
        await service.initialize()
        return service

    init_embedding_service = providers.Resource(
        _init_embedding_service,
        service=external_services.sentence_transformer_service,
    )

    # Health check providers
    async def _llm_service_health_check(service: OllamaService) -> bool:
        try:
            return await service.health_check()
        except Exception:
            return False

    llm_service_health_check = providers.Factory(_llm_service_health_check, service=llm_service)

    async def _embedding_service_health_check(service: SentenceTransformerService) -> bool:
        try:
            return service.is_loaded
        except Exception:
            return False

    embedding_service_health_check = providers.Factory(
        _embedding_service_health_check,
        service=external_services.sentence_transformer_service,
    )

    async def _services_health_check(llm_healthy: bool, embedding_healthy: bool) -> dict:
        return {
            "overall": llm_healthy and embedding_healthy,
            "llm_service": llm_healthy,
            "embedding_service": embedding_healthy,
        }

    services_health_check = providers.Factory(
        _services_health_check,
        llm_healthy=llm_service_health_check,
        embedding_healthy=embedding_service_health_check,
    )

    # Environment-specific service providers
    def _active_llm_service(production_service: OllamaService, is_development: bool) -> LLMService:
        return production_service

    active_llm_service = providers.Factory(
        _active_llm_service,
        production_service=llm_service,
        is_development=config.is_development,
    )

    def _active_embedding_service(
        production_service: SentenceTransformerService,
        is_development: bool,
    ) -> SentenceTransformerService:
        return production_service

    active_embedding_service = providers.Factory(
        _active_embedding_service,
        production_service=external_services.sentence_transformer_service,
        is_development=config.is_development,
    )

    # Performance monitoring providers
    def _service_metrics(
        llm_service: OllamaService,
        embedding_service: SentenceTransformerService,
    ) -> dict:
        return {
            "llm_service": {
                "request_count": getattr(llm_service, "request_count", 0),
                "average_response_time": getattr(llm_service, "average_response_time", 0.0),
                "error_count": getattr(llm_service, "error_count", 0),
            },
            "embedding_service": {
                "embeddings_generated": getattr(embedding_service, "embeddings_generated", 0),
                "average_embedding_time": getattr(embedding_service, "average_embedding_time", 0.0),
                "cache_hit_rate": getattr(embedding_service, "cache_hit_rate", 0.0),
            },
        }

    service_metrics = providers.Factory(
        _service_metrics,
        llm_service=llm_service,
        embedding_service=external_services.sentence_transformer_service,
    )

    # Configuration-driven service providers
    def _llm_service_config(settings: Settings) -> dict:
        return {
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "timeout": settings.llm_timeout,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

    llm_service_config = providers.Factory(_llm_service_config, settings=config.settings)

    def _embedding_service_config(settings: Settings) -> dict:
        return {
            "model_name": settings.sentence_transformer_model,
            "dimension": settings.sentence_transformer_dimension,
            "device": settings.sentence_transformer_device,
            "batch_size": settings.sentence_transformer_batch_size,
            "cache_strategy": settings.sentence_transformer_cache_strategy,
            "cache_size_mb": settings.sentence_transformer_cache_size_mb,
        }

    embedding_service_config = providers.Factory(_embedding_service_config, settings=config.settings)
