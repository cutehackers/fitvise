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
from app.config.ml_models.embedding_model_configs import CacheStrategy, DeviceType, EmbeddingModelConfig
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


def build_sentence_transformer_service(configs: providers.Configuration) -> SentenceTransformerService:
    return SentenceTransformerService(
        config=EmbeddingModelConfig(
            model_name=configs.embedding_model_name(),
            model_dimension=configs.embedding_model_dimension(),
            device=DeviceType(configs.embedding_device()),
            batch_size=configs.embedding_batch_size(),
            max_seq_length=configs.embedding_max_seq_length(),
            normalize_embeddings=configs.embedding_normalize_embeddings(),
            cache_strategy=CacheStrategy(configs.embedding_cache_strategy()),
            num_workers=configs.embedding_num_workers(),
            show_progress=configs.embedding_show_progress(),
            model_kwargs={"trust_remote_code": configs.embedding_trust_remote_code()},
        )
    )


def build_weaviate_client(configs: providers.Configuration) -> WeaviateClient:
    return WeaviateClient(
        config=WeaviateConfig(
            host=configs.weaviate_host(),
            port=configs.weaviate_port(),
            scheme=configs.weaviate_scheme(),
            auth_type=configs.weaviate_auth_type(),
            api_key=configs.weaviate_api_key(),
            timeout=configs.weaviate_timeout(),
            connection_timeout=configs.weaviate_connection_timeout(),
            read_timeout=configs.weaviate_read_timeout(),
            startup_period=configs.weaviate_startup_period(),
            max_retries=configs.weaviate_max_retries(),
            retry_delay=configs.weaviate_retry_delay(),
            consistency_level=configs.weaviate_consistency_level(),
            additional_headers=configs.weaviate_additional_headers(),
            grpc_secure=configs.weaviate_grpc_secure(),
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


class AppContainer(containers.DeclarativeContainer):
    """
    Dependency Injector Container for Fitvise
    """

    # SETTINGS
    configs = providers.Configuration()
    settings: providers.Singleton[Settings] = providers.Singleton(Settings)
    configs.from_pydantic(settings.provided)

    # INFRASTRUCTURE

    ## database
    db_session: providers.Resource[AsyncSession] = providers.Resource(get_async_session)

    ## embedding
    embedding = providers.Factory(
        HuggingFaceEmbedding,
        model_name=configs.embedding_model_name,
        trust_remote_code=configs.embedding_trust_remote_code,
    )

    ## embedding service
    sentence_transformer_service = providers.Singleton(
        build_sentence_transformer_service,
        configs=configs,
    )

    ## external services
    weaviate_client = providers.Singleton(build_weaviate_client, configs=configs)

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

    ## persistence
    document_repository = providers.Selector(
        configs.database_type,
        default=providers.Singleton(InMemoryDocumentRepository),
        aiosqlite=providers.Singleton(SQLAlchemyDocumentRepository, session=db_session),
        asyncpg=providers.Singleton(SQLAlchemyDocumentRepository, session=db_session),
    )
    data_source_repository = providers.Singleton(InMemoryDataSourceRepository)
    embedding_repository = providers.Singleton(WeaviateEmbeddingRepository, client=weaviate_client)
