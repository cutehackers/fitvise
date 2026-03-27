"""FastAPI dependency injection for Fitvise API endpoints.

This module provides FastAPI dependency functions organized by architectural layers:
- INFRASTRUCTURE: Core infrastructure services and containers
- DOMAIN: Domain services and business logic
- APPLICATION: Application use cases and orchestrators
"""

import logging
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import BaseCallbackHandler

from app.di.containers.container import AppContainer
from app.domain.services.context_window_manager import (
    ContextWindow,
    ContextWindowManager,
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
from app.core.settings import Settings

logger = logging.getLogger(__name__)


# ========================================
# INFRASTRUCTURE
# ========================================

@lru_cache()
def get_settings(container: Any) -> Settings:
    """Get cached settings instance.

    Returns:
        Application settings
    """
    return container.settings()


@lru_cache()
def get_app_container() -> AppContainer:
    """Get main application container with all layered services.

    Returns:
        AppContainer with infra, domain, and application layers
    """
    container = AppContainer()
    logger.info("AppContainer initialized with layered architecture")
    return container


# ========================================
# DOMAIN
# ========================================

@lru_cache()
def get_context_window_manager(container: Any) -> ContextWindowManager:
    """Get context window manager singleton.

    Returns:
        ContextWindowManager configured with settings
    """
    settings = container.settings()
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
    """Get shared session service to maintain chat history across requests.

    Returns:
        SessionService for managing user sessions
    """
    return SessionService()


@lru_cache()
def get_llm_health_monitor(container: Any) -> LlmHealthMonitor:
    """Get LLM health monitor singleton.

    Returns:
        LlmHealthMonitor for health tracking
    """
    # Create a temporary wrapper for health monitoring
    settings_instance = container.settings()
    ollama_service = OllamaService(settings_instance)
    monitor = LlmHealthMonitor(ollama_service)
    logger.info("LlmHealthMonitor initialized")
    return monitor


# ========================================
# APPLICATION
# ========================================

async def get_llama_index_retriever(
    container: Annotated[Any, Depends(get_app_container)]
) -> BaseRetriever:
    """Get LlamaIndex retriever with connected Weaviate.

    Ensures Weaviate connection is established before creating retriever.

    Args:
        container: Application container with infra services

    Returns:
        LlamaIndex-backed retriever for semantic search
    """
    # Container handles Weaviate connection internally
    retriever = container.llama_index_retriever
    settings = container.settings()
    logger.info(
        "LlamaIndex retriever obtained: top_k=%d, threshold=%.2f",
        settings.rag_retrieval_top_k,
        settings.rag_retrieval_similarity_threshold,
    )
    return retriever


async def get_rag_chat_use_case(
    session_service: Annotated[SessionService, Depends(get_session_service)],
    container: Annotated[Any, Depends(get_app_container)],
    callback_handler: Annotated[BaseCallbackHandler | None, Depends(get_callback_handler)],
) -> RagChatUseCase:
    """Get RAG Chat use case with all dependencies.

    Args:
        session_service: Shared session service for maintaining conversation history
        container: Application container with all layered services
        callback_handler: Optional LangChain callback handler for analytics

    Returns:
        RagChatUseCase for RAG-enabled chat with document retrieval
    """
    # Container handles Weaviate connection internally
    settings = container.settings()
    llm_service = get_llm_service(settings, callback_handler)
    retriever = container.llama_index_retriever
    context_mgr = get_context_window_manager(container)

    rag_chat_use_case = RagChatUseCase(
        llm_service=llm_service,
        retriever=retriever,
        context_manager=context_mgr,
        session_service=session_service,
        callback_handler=callback_handler,
    )
    logger.info("RagChatUseCase initialized (replaces RagOrchestrator)")
    return rag_chat_use_case
