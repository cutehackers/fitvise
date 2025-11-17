"""Dependency injection for LLM services."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.settings import Settings, settings
from app.domain.llm.interfaces.chat_orchestrator import ChatOrchestrator
from app.domain.llm.interfaces.llm_service import LLMService
from app.infrastructure.external_services.ml_services.llm_services.ollama_service import OllamaService
from app.infrastructure.llm.services.langchain_orchestrator import LangChainOrchestrator


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return settings


@lru_cache()
def get_llm_service(settings: Annotated[Settings, Depends(get_settings)]) -> LLMService:
    """Get LLM service instance.

    Args:
        settings: Application settings

    Returns:
        LLM service implementation
    """
    return OllamaService(settings)


@lru_cache()
def get_chat_orchestrator(
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatOrchestrator:
    """Get chat orchestrator instance.

    Args:
        llm_service: LLM service
        settings: Application settings

    Returns:
        Chat orchestrator implementation
    """
    return LangChainOrchestrator(
        llm_service=llm_service,
        turns_window=getattr(settings, 'chat_turns_window', 10),
        max_session_age_hours=getattr(settings, 'chat_max_session_age_hours', 24),
    )
