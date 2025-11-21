"""Dependency injection for Fitvise API endpoints."""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langchain_core.retrievers import BaseRetriever

from app.application.use_cases.llm_infrastructure.setup_ollama_rag import (
    SetupOllamaRagUseCase,
)
from app.core.settings import settings, Settings
from app.infrastructure.external_services.context_management.context_window_manager import (
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
from app.infrastructure.llm.dependencies import get_llm_service
from app.infrastructure.llm.services.rag_orchestrator import RagOrchestrator

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


async def get_rag_use_case(
    container: Annotated[ExternalServicesContainer, Depends(get_external_services_container)]
) -> SetupOllamaRagUseCase:
    """Get RAG use case singleton with all dependencies.

    Args:
        container: External services container

    Returns:
        SetupOllamaRagUseCase for RAG orchestration
    """
    # Ensure Weaviate is connected
    await container.ensure_weaviate_connected()

    llm_service = get_llm_service(settings)
    retriever = container.llama_index_retriever
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
    settings_instance = Settings()
    ollama_service = OllamaService(settings_instance)
    monitor = LlmHealthMonitor(ollama_service)
    logger.info("LlmHealthMonitor initialized")
    return monitor


async def get_rag_orchestrator(
    container: Annotated[ExternalServicesContainer, Depends(get_external_services_container)]
) -> RagOrchestrator:
    """Get RAG orchestrator with all dependencies.

    Args:
        container: External services container with retriever and context manager

    Returns:
        RagOrchestrator for RAG-enabled chat with document retrieval

    Raises:
        ExternalServicesError: If Weaviate connection fails
    """
    # Ensure Weaviate is connected
    await container.ensure_weaviate_connected()

    llm_service = get_llm_service(settings)
    retriever = container.llama_index_retriever
    context_mgr = get_context_window_manager()

    rag_orchestrator = RagOrchestrator(
        llm_service=llm_service,
        retriever=retriever,
        context_manager=context_mgr,
    )
    logger.info("RagOrchestrator initialized")
    return rag_orchestrator
