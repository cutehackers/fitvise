"""Dependency injection for Fitvise API endpoints."""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import BaseCallbackHandler

from app.core.settings import settings, Settings
from app.domain.services.context_window_manager import (
    ContextWindow,
    ContextWindowManager,
)
from app.infrastructure.external_services.external_services_container import (
    ExternalServicesContainer,
)
from app.infrastructure.external_services.ml_services.llm_services.llm_health_monitor import (
    LlmHealthMonitor,
)
from app.infrastructure.external_services.ml_services.llm_services.ollama_service import (
    OllamaService,
)
from app.infrastructure.llm.dependencies import get_llm_service, get_callback_handler
from app.domain.services.session_service import SessionService
from app.application.use_cases.chat.rag_chat_use_case import RagChatUseCase

logger = logging.getLogger(__name__)


@lru_cache()
def get_settings_instance() -> Settings:
    """Get cached settings instance.

    Returns:
        Application settings
    """
    return settings


@lru_cache()
def get_external_services_container(
    settings_instance: Annotated[Settings, Depends(get_settings_instance)]
) -> ExternalServicesContainer:
    """Get external services container singleton.

    Args:
        settings_instance: Application settings

    Returns:
        ExternalServicesContainer with all external services initialized
    """
    container = ExternalServicesContainer(settings_instance)
    logger.info("ExternalServicesContainer initialized")
    return container




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
def get_session_service() -> SessionService:
    """Get shared session service to maintain chat history across requests."""
    return SessionService()


async def get_llama_index_retriever(
    container: Annotated[ExternalServicesContainer, Depends(get_external_services_container)]
) -> BaseRetriever:
    """Get LlamaIndex retriever with connected Weaviate.

    Ensures Weaviate connection is established before creating retriever.

    Args:
        container: External services container

    Returns:
        LlamaIndex-backed retriever for semantic search

    Raises:
        ExternalServicesError: If Weaviate connection fails
    """
    # Ensure Weaviate is connected
    await container.ensure_weaviate_connected()

    # Get retriever from container
    retriever = container.llama_index_retriever
    logger.info(
        "LlamaIndex retriever obtained: top_k=%d, threshold=%.2f",
        settings.rag_retrieval_top_k,
        settings.rag_retrieval_similarity_threshold,
    )
    return retriever


@lru_cache()
def get_llm_health_monitor() -> LlmHealthMonitor:
    """Get LLM health monitor singleton.

    Returns:
        LlmHealthMonitor for health tracking
    """
    # Create a temporary wrapper for health monitoring
    settings_instance = Settings()
    ollama_service = OllamaService(settings_instance)
    monitor = LlmHealthMonitor(ollama_service)
    logger.info("LlmHealthMonitor initialized")
    return monitor


async def get_rag_chat_use_case(
    session_service: Annotated[SessionService, Depends(get_session_service)],
    container: Annotated[ExternalServicesContainer, Depends(get_external_services_container)],
    callback_handler: Annotated[BaseCallbackHandler | None, Depends(get_callback_handler)],
) -> RagChatUseCase:
    """Get RAG Chat use case with all dependencies.

    Args:
        session_service: Shared session service for maintaining conversation history
        container: External services container with retriever and context manager
        callback_handler: Optional LangChain callback handler for analytics

    Returns:
        RagChatUseCase for RAG-enabled chat with document retrieval

    Raises:
        ExternalServicesError: If Weaviate connection fails
    """
    # Ensure Weaviate is connected
    await container.ensure_weaviate_connected()

    llm_service = get_llm_service(settings, callback_handler)
    retriever = container.llama_index_retriever
    context_mgr = get_context_window_manager()

    rag_chat_use_case = RagChatUseCase(
        llm_service=llm_service,
        retriever=retriever,
        context_manager=context_mgr,
        session_service=session_service,
        callback_handler=callback_handler,
    )
    logger.info("RagChatUseCase initialized (replaces RagOrchestrator)")
    return rag_chat_use_case
