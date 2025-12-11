"""External Service Providers.

This module provides external service integration providers for the DI container.
It handles Weaviate clients, embedding models, LLM services, and other external integrations.
"""

from dependency_injector import containers, providers
from langchain_core.retrievers import BaseRetriever

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.infrastructure.llm.services.llama_index_retriever import create_llama_index_weaviate_retriever
from app.infrastructure.llm.services.ollama_service import OllamaService


class ExternalServiceProviders(containers.DeclarativeContainer):
    """External service providers for the FitVise application.

    Provides access to external services like Weaviate, embedding models,
    and LLM services with proper lifecycle management and configuration.
    """

    # Configuration dependencies
    config = providers.Dependency()

    # Weaviate client provider
    weaviate_client = providers.Singleton(
        WeaviateClient,
        config=config.weaviate_config,
    )

    # Local development Weaviate client
    local_weaviate_client = providers.Singleton(
        WeaviateClient,
        config=config.local_weaviate_config,
    )

    # Sentence transformer service for real-time operations
    sentence_transformer_service = providers.Singleton(
        SentenceTransformerService,
        config=config.realtime_embedding_config,
    )

    # LlamaIndex embedding model
    llama_index_embedding = providers.Singleton(
        lambda config: __import__("llama_index.embeddings.huggingface", fromlist=["HuggingFaceEmbedding"]).HuggingFaceEmbedding(
            model_name=config.embedding_config.model_name,
            trust_remote_code=True,  # Required for Alibaba-NLP models
        ),
        config=config,
    )

    # Ollama LLM service
    ollama_service = providers.Singleton(
        OllamaService,
        base_url=config.settings.llm_base_url,
        model=config.settings.llm_model,
        timeout=config.settings.llm_timeout,
        temperature=config.settings.llm_temperature,
        max_tokens=config.settings.llm_max_tokens,
    )

    # Resource initialization providers
    async def _init_weaviate_client(client: WeaviateClient) -> WeaviateClient:
        if not client.is_connected:
            await client.connect()
        return client

    init_weaviate_client = providers.Resource(_init_weaviate_client, client=weaviate_client)

    async def _init_sentence_transformer(service: SentenceTransformerService) -> SentenceTransformerService:
        await service.initialize()
        return service

    init_sentence_transformer = providers.Resource(
        _init_sentence_transformer,
        service=sentence_transformer_service,
    )

    async def _init_ollama_service(service: OllamaService) -> OllamaService:
        await service.initialize()
        return service

    init_ollama_service = providers.Resource(_init_ollama_service, service=ollama_service)

    # Health check providers
    async def _weaviate_health_check(client: WeaviateClient) -> bool:
        try:
            return await client.health_check()
        except Exception:
            return False

    weaviate_health_check = providers.Factory(_weaviate_health_check, client=weaviate_client)

    async def _embedding_service_health_check(service: SentenceTransformerService) -> bool:
        try:
            return service.is_loaded
        except Exception:
            return False

    embedding_service_health_check = providers.Factory(
        _embedding_service_health_check,
        service=sentence_transformer_service,
    )

    async def _ollama_health_check(service: OllamaService) -> bool:
        try:
            return await service.health_check()
        except Exception:
            return False

    ollama_health_check = providers.Factory(_ollama_health_check, service=ollama_service)

    async def _external_services_health(
        weaviate_healthy: bool,
        embedding_healthy: bool,
        ollama_healthy: bool,
    ) -> dict:
        return {
            "overall": weaviate_healthy and embedding_healthy and ollama_healthy,
            "weaviate": weaviate_healthy,
            "embedding_service": embedding_healthy,
            "ollama": ollama_healthy,
        }

    external_services_health = providers.Factory(
        _external_services_health,
        weaviate_healthy=weaviate_health_check,
        embedding_healthy=embedding_service_health_check,
        ollama_healthy=ollama_health_check,
    )

    def _llama_index_retriever(
        client: WeaviateClient,
        settings,
    ) -> BaseRetriever:
        if not client.is_connected:
            raise ValueError(
                "WeaviateClient must be connected before creating retriever. "
                "Initialize external resources before requesting retriever."
            )
        return create_llama_index_weaviate_retriever(
            weaviate_client=client,
            top_k=getattr(settings, "rag_retrieval_top_k", 5),
            similarity_threshold=getattr(settings, "rag_retrieval_similarity_threshold", 0.7),
        )

    llama_index_retriever = providers.Factory(
        _llama_index_retriever,
        client=weaviate_client,
        settings=config.settings,
    )

    # Environment-specific service providers
    def _active_weaviate_client(
        client: WeaviateClient,
        local_client: WeaviateClient,
        is_development: bool,
    ) -> WeaviateClient:
        return local_client if is_development else client

    active_weaviate_client = providers.Factory(
        _active_weaviate_client,
        client=weaviate_client,
        local_client=local_weaviate_client,
        is_development=config.is_development,
    )

    def _active_embedding_config(
        realtime_config: EmbeddingModelConfig,
        default_config: EmbeddingModelConfig,
        is_production: bool,
    ) -> EmbeddingModelConfig:
        return realtime_config if is_production else default_config

    active_embedding_config = providers.Factory(
        _active_embedding_config,
        realtime_config=config.realtime_embedding_config,
        default_config=config.embedding_config,
        is_production=config.is_production,
    )
