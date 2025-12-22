from dataclasses import dataclass
from dependency_injector import containers, providers

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.core.settings import Settings
from app.infrastructure.database.database import get_async_session
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.infrastructure.persistence.repositories.in_memory_data_source_repository import (
    InMemoryDataSourceRepository,
)
from app.infrastructure.persistence.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.persistence.repositories.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from app.infrastructure.persistence.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)
from app.config.ml_models.embedding_model_configs import (
    CacheStrategy,
    DeviceType,
    EmbeddingModelConfig,
)
from app.infrastructure.llm.services.llama_index_retriever import LlamaIndexRetriever
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)

from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from llama_index.core.retrievers import BaseRetriever as LlamaIndexBaseRetriever
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore


def _get_config_value(configs, key: str, default=None):
    """Fetch config values from providers.Configuration or plain dict(Settings)."""
    if isinstance(configs, providers.Configuration):
        provider = getattr(configs, key)
        return provider()
    if hasattr(configs, key):
        attr = getattr(configs, key)
        return attr() if callable(attr) else attr
    if isinstance(configs, dict):
        return configs.get(key, default)
    return default


def build_sentence_transformer_service(configs) -> SentenceTransformerService:
    """Factory for SentenceTransformerService that works with Provider or dict configs."""
    return SentenceTransformerService(
        config=EmbeddingModelConfig(
            model_name=_get_config_value(configs, "embedding_model_name"),
            model_dimension=_get_config_value(
                configs, "embedding_model_dimension"
            ),
            device=DeviceType(_get_config_value(configs, "embedding_device")),
            batch_size=_get_config_value(configs, "embedding_batch_size"),
            max_seq_length=_get_config_value(configs, "embedding_max_seq_length"),
            normalize_embeddings=_get_config_value(
                configs, "embedding_normalize_embeddings"
            ),
            cache_strategy=CacheStrategy(
                _get_config_value(configs, "embedding_cache_strategy")
            ),
            num_workers=_get_config_value(configs, "embedding_num_workers"),
            show_progress=_get_config_value(configs, "embedding_show_progress"),
            model_kwargs={
                "trust_remote_code": _get_config_value(
                    configs, "embedding_trust_remote_code"
                )
            },
        )
    )


def build_weaviate_client(configs) -> WeaviateClient:
    """Factory for WeaviateClient that works with Provider or dict configs."""
    return WeaviateClient(
        config=WeaviateConfig(
            host=_get_config_value(configs, "weaviate_host"),
            port=_get_config_value(configs, "weaviate_port"),
            scheme=_get_config_value(configs, "weaviate_scheme"),
            auth_type=_get_config_value(configs, "weaviate_auth_type"),
            api_key=_get_config_value(configs, "weaviate_api_key"),
            timeout=_get_config_value(configs, "weaviate_timeout"),
            connection_timeout=_get_config_value(configs, "weaviate_connection_timeout"),
            read_timeout=_get_config_value(configs, "weaviate_read_timeout"),
            startup_period=_get_config_value(configs, "weaviate_startup_period"),
            max_retries=_get_config_value(configs, "weaviate_max_retries"),
            retry_delay=_get_config_value(configs, "weaviate_retry_delay"),
            consistency_level=_get_config_value(configs, "weaviate_consistency_level"),
            additional_headers=_get_config_value(configs, "weaviate_additional_headers"),
            grpc_secure=_get_config_value(configs, "weaviate_grpc_secure"),
        )
    )


@dataclass(frozen=True)
class WeaviateLlamaIndexResources:
    client: WeaviateClient
    vector_store: WeaviateVectorStore
    index: VectorStoreIndex


async def build_llama_index_resources(
    weaviate_client: WeaviateClient, configs: providers.Configuration
) -> WeaviateLlamaIndexResources:
    """
    Builds the vector store + index wrapper.
    NOTE:
      - This assumes embeddings are configured globally via LlamaIndex Settings
        (e.g. LlamaSettings.embed_model).
      - If you need custom embed/llm per app, set LlamaSettings before building index.
    """

    # ensure weaviate is connected
    if not weaviate_client.is_connected:
        await weaviate_client.connect()

    vector_store = WeaviateVectorStore(
        weaviate_client=weaviate_client._client,
        index_name=configs.llama_index_index_name(),
    )
    embed_model = HuggingFaceEmbedding(
        model_name=configs.embedding_model_name(),
        trust_remote_code=configs.embedding_trust_remote_code(),
    )

    # Create index from existing vector store (no re-indexing)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return WeaviateLlamaIndexResources(
        client=weaviate_client,
        vector_store=vector_store,
        index=index,
    )


def build_llama_index_retriever(
    resources: WeaviateLlamaIndexResources, configs: providers.Configuration
) -> LangChainBaseRetriever:
    # Use LlamaIndex's native as_retriever() method
    llama_retriever: LlamaIndexBaseRetriever = resources.index.as_retriever(
        similarity_top_k=configs.llama_index_top_k()
    )

    # Wrap in LangChain-compatible retriever
    return LlamaIndexRetriever(
        llama_retriever=llama_retriever,
        config={
            "similarity_threshold": configs.llama_index_similarity_threshold(),
            "top_k": configs.llama_index_top_k(),
            "index_name": configs.llama_index_index_name(),
            "text_key": configs.llama_index_text_key(),
        },
    )


class InfraContainer(containers.DeclarativeContainer):
    """
    Infrastructure Layer Container

    Contains all infrastructure concerns:
    - Database connections
    - External services (Weaviate, ML models, LLM services)
    - Persistence repositories
    """

    # SETTINGS
    configs = providers.Configuration()

    # DATABASE
    db_session: providers.Resource[AsyncSession] = providers.Resource(get_async_session)

    # EXTERNAL SERVICES
    weaviate_client = providers.Singleton(build_weaviate_client, configs=configs)

    # EMBEDDING SERVICES
    embedding = providers.Factory(
        HuggingFaceEmbedding,
        model_name=configs.embedding_model_name,
        trust_remote_code=configs.embedding_trust_remote_code,
    )

    sentence_transformer_service = providers.Singleton(
        build_sentence_transformer_service,
        configs=configs,
    )

    # LLAMA INDEX RESOURCES
    llama_index_resources = providers.Resource(
        build_llama_index_resources,
        weaviate_client=weaviate_client,
        configs=configs,
    )

    llama_index_retriever = providers.Factory(
        build_llama_index_retriever,
        resources=llama_index_resources,
        configs=configs,
    )

    # PERSISTENCE REPOSITORIES
    document_repository = providers.Selector(
        configs.database_type,
        default=providers.Singleton(InMemoryDocumentRepository),
        aiosqlite=providers.Singleton(SQLAlchemyDocumentRepository, session=db_session),
        asyncpg=providers.Singleton(SQLAlchemyDocumentRepository, session=db_session),
    )
    data_source_repository = providers.Singleton(InMemoryDataSourceRepository)
    embedding_repository = providers.Singleton(WeaviateEmbeddingRepository, client=weaviate_client)
