"""Dependency injection helpers for Fitvise API endpoints."""

import logging
from functools import lru_cache
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.retrievers import BaseRetriever

from app.di import container as di_container
from app.di.container import FitviseContainer
from app.core.settings import Settings
from app.domain.llm.interfaces.llm_service import LLMService
from app.domain.services.context_window_manager import (
    ContextWindow,
    ContextWindowManager,
)
from app.domain.services.session_service import SessionService
from app.infrastructure.external_services.vector_stores.weaviate_client import WeaviateClient
from app.infrastructure.llm.dependencies import build_callback_handler
from app.infrastructure.external_services.ml_services.llm_services.llm_health_monitor import (
    LlmHealthMonitor,
)
from app.application.use_cases.chat.rag_chat_use_case import RagChatUseCase

logger = logging.getLogger(__name__)


SettingsProvider = Provide[FitviseContainer.config.settings]
LlmServiceProvider = Provide[FitviseContainer.services.llm_service]
WeaviateClientProvider = Provide[FitviseContainer.external.weaviate_client]


@lru_cache()
@inject
def get_context_window_manager(
    settings: Settings = Depends(SettingsProvider),
) -> ContextWindowManager:
    """Create context window manager from DI-managed settings."""
    config = ContextWindow(
        max_tokens=settings.llm_context_window,
        reserve_tokens=settings.llm_reserve_tokens,
        truncation_strategy=settings.context_truncation_strategy,
    )
    logger.info(
        "ContextWindowManager initialized: max_tokens=%d, strategy=%s",
        config.max_tokens,
        config.truncation_strategy,
    )
    return ContextWindowManager(config)


@lru_cache()
def get_session_service() -> SessionService:
    """Get shared session service to maintain chat history across requests."""
    return SessionService()


@inject
def get_llm_health_monitor(
    llm_service: LLMService = Depends(LlmServiceProvider),
) -> LlmHealthMonitor:
    """Return health monitor bound to DI-managed LLM service."""
    return LlmHealthMonitor(llm_service)


@inject
def get_callback_handler(
    settings: Settings = Depends(SettingsProvider),
) -> BaseCallbackHandler | None:
    """Build LangChain callback handler using DI-managed settings."""
    return build_callback_handler(settings)


@inject
async def get_llama_index_retriever(
    weaviate_client: WeaviateClient = Depends(WeaviateClientProvider),
) -> BaseRetriever:
    """Get LlamaIndex retriever, ensuring Weaviate client is initialized via DI."""
    if not weaviate_client.is_connected:
        await di_container.external.init_weaviate_client()
    return di_container.external.llama_index_retriever()


@inject
async def get_rag_chat_use_case(
    session_service: Annotated[SessionService, Depends(get_session_service)],
    llm_service: LLMService = Depends(LlmServiceProvider),
    retriever: BaseRetriever = Depends(get_llama_index_retriever),
    callback_handler: Annotated[BaseCallbackHandler | None, Depends(get_callback_handler)] = None,
    settings: Settings = Depends(SettingsProvider),
) -> RagChatUseCase:
    """Construct RagChatUseCase using DI-managed services."""
    context_manager = get_context_window_manager(settings)

    return RagChatUseCase(
        llm_service=llm_service,
        retriever=retriever,
        context_manager=context_manager,
        session_service=session_service,
        callback_handler=callback_handler,
        turns_window=getattr(settings, "chat_turns_window", 10),
        max_session_age_hours=getattr(settings, "chat_max_session_age_hours", 24),
    )
