"""Dependency injection for LLM services."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.settings import Settings, settings
from app.domain.llm.interfaces.llm_service import LLMService
from app.domain.services.session_service import SessionService
from app.infrastructure.external_services.ml_services.llm_services.ollama_service import OllamaService
from app.application.use_cases.chat.chat_use_case import ChatUseCase


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
) -> ChatUseCase:
    """Get Chat use case instance (replaces LangChainOrchestrator).

    Args:
        llm_service: LLM service
        settings: Application settings

    Returns:
        ChatUseCase implementation for basic chat functionality
    """
    # Create session service with default configuration
    session_service = SessionService()

    chat_use_case = ChatUseCase(
        llm_service=llm_service,
        session_service=session_service,
        turns_window=getattr(settings, 'chat_turns_window', 10),
        max_session_age_hours=getattr(settings, 'chat_max_session_age_hours', 24),
    )
    return chat_use_case
