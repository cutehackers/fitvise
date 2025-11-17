"""Dependency injection for Fitvise API endpoints."""

import logging
from functools import lru_cache

from app.application.use_cases.llm_infrastructure.setup_ollama_rag import (
    SetupOllamaRagUseCase,
)
from app.core.settings import settings
from app.infrastructure.adapters.weaviate_langchain_retriever import (
    WeaviateLangChainRetriever,
)
from app.infrastructure.external_services.context_management.context_window_manager import (
    ContextWindow,
    ContextWindowManager,
)
from app.infrastructure.llm.dependencies import get_llm_service
from app.infrastructure.external_services.ml_services.llm_services.llm_health_monitor import (
    LlmHealthMonitor,
)
from app.infrastructure.repositories.weaviate_search_repository import (
    WeaviateSearchRepository,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)

logger = logging.getLogger(__name__)


@lru_cache()
def get_weaviate_client() -> WeaviateClient:
    """Get Weaviate client singleton.

    Returns:
        WeaviateClient instance
    """
    client = WeaviateClient()
    logger.info("WeaviateClient initialized")
    return client




@lru_cache()
def get_context_window_manager() -> ContextWindowManager:
    """Get context window manager singleton.

    Returns:
        ContextWindowManager configured with settings
    """
    config = ContextWindow(
        max_tokens=settings.llm_context_window,
        reserve_tokens=settings.llm_reserve_tokens,
        truncation_strategy=settings.context_truncation_strategy,
    )
    manager = ContextWindowManager(config)
    logger.info(
        "ContextWindowManager initialized: max_tokens=%d, strategy=%s",
        config.max_tokens,
        config.truncation_strategy,
    )
    return manager


@lru_cache()
def get_weaviate_retriever() -> WeaviateLangChainRetriever:
    """Get Weaviate retriever singleton.

    Returns:
        WeaviateLangChainRetriever for semantic search
    """
    weaviate_client = get_weaviate_client()
    search_repo = WeaviateSearchRepository(weaviate_client)

    retriever = WeaviateLangChainRetriever(
        search_repository=search_repo,
        top_k=settings.rag_retrieval_top_k,
        similarity_threshold=settings.rag_retrieval_similarity_threshold,
    )
    logger.info(
        "WeaviateLangChainRetriever initialized: top_k=%d, threshold=%.2f",
        settings.rag_retrieval_top_k,
        settings.rag_retrieval_similarity_threshold,
    )
    return retriever


@lru_cache()
def get_rag_use_case() -> SetupOllamaRagUseCase:
    """Get RAG use case singleton.

    Returns:
        SetupOllamaRagUseCase for RAG orchestration
    """
    llm_service = get_llm_service()
    retriever = get_weaviate_retriever()
    context_mgr = get_context_window_manager()

    rag_use_case = SetupOllamaRagUseCase(
        llm_service=llm_service, retriever=retriever, context_manager=context_mgr
    )
    logger.info("SetupOllamaRagUseCase initialized")
    return rag_use_case


@lru_cache()
def get_llm_health_monitor() -> LlmHealthMonitor:
    """Get LLM health monitor singleton.

    Returns:
        LlmHealthMonitor for health tracking
    """
    # Create a temporary wrapper for health monitoring
    from app.infrastructure.external_services.ml_services.llm_services.ollama_service import OllamaService
    from app.core.settings import Settings

    settings_instance = Settings()
    ollama_service = OllamaService(settings_instance)
    monitor = LlmHealthMonitor(ollama_service)
    logger.info("LlmHealthMonitor initialized")
    return monitor
