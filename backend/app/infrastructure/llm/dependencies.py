"""Dependency injection for LLM services."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.settings import Settings, settings
from app.domain.llm.interfaces.chat_orchestrator import ChatOrchestrator
from app.domain.llm.interfaces.llm_provider import LLMProvider
from app.infrastructure.llm.providers.ollama_provider import OllamaProvider
from app.infrastructure.llm.services.langchain_orchestrator import LangChainOrchestrator


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return settings


@lru_cache()
def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    """Get LLM provider instance.

    Args:
        settings: Application settings

    Returns:
        LLM provider implementation
    """
    return OllamaProvider(settings)


@lru_cache()
def get_chat_orchestrator(
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatOrchestrator:
    """Get chat orchestrator instance.

    Args:
        llm_provider: LLM provider
        settings: Application settings

    Returns:
        Chat orchestrator implementation
    """
    return LangChainOrchestrator(
        llm_provider=llm_provider,
        turns_window=getattr(settings, 'chat_turns_window', 10),
        max_session_age_hours=getattr(settings, 'chat_max_session_age_hours', 24),
    )
