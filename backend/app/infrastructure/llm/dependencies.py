"""Dependency injection for LLM services."""

import os
import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langchain_core.callbacks import BaseCallbackHandler
from langfuse.langchain import CallbackHandler as LangfuseLangchainHandler

from app.core.settings import Settings, settings
from app.domain.llm.interfaces.llm_service import LLMService
from app.domain.services.session_service import SessionService
from app.infrastructure.external_services.ml_services.llm_services.ollama_service import OllamaService
from app.application.use_cases.chat.chat_use_case import ChatUseCase

logger = logging.getLogger(__name__)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return settings


@lru_cache()
def get_callback_handler(settings: Annotated[Settings, Depends(get_settings)]) -> BaseCallbackHandler | None:
    """Get LangChain callback handler for analytics.

    Uses standard LangFuse environment variables for configuration.
    If no environment variables are set, returns None to disable analytics.

    Args:
        settings: Application settings

    Returns:
        LangChain callback handler for analytics, None if disabled
    """
    try:
        # Check if LangFuse environment variables are set
        secret_key = settings.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        public_key = settings.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        host = settings.langfuse_host or os.getenv("LANGFUSE_HOST")

        if not secret_key or not public_key or not host:
            logger.info("LangFuse environment variables not found, analytics disabled")
            return None

        # LangFuse SDK expects credentials in environment variables
        if secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        if public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        if host:
            os.environ["LANGFUSE_HOST"] = host

        # Import and create callback handler
        # Note: LangfuseLangchainHandler does not accept credentials in __init__
        callback_handler = LangfuseLangchainHandler(update_trace=True)

        logger.info("LangFuse callback handler initialized successfully")
        return callback_handler

    except Exception as e:
        logger.warning(f"Failed to initialize LangFuse callback handler: {e}")
        return None


def get_llm_service(
    settings: Annotated[Settings, Depends(get_settings)],
    callback_handler: Annotated[BaseCallbackHandler | None, Depends(get_callback_handler)],
) -> LLMService:
    """Get LLM service instance.

    Args:
        settings: Application settings
        callback_handler: Optional LangChain callback handler for analytics

    Returns:
        LLM service implementation
    """
    return OllamaService(settings, callback_handler)


@lru_cache()
def get_chat_use_case(
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    callback_handler: Annotated[BaseCallbackHandler | None, Depends(get_callback_handler)],
) -> ChatUseCase:
    """Get Chat use case instance (replaces LangChainOrchestrator).

    Args:
        llm_service: LLM service
        settings: Application settings
        callback_handler: Optional LangChain callback handler for analytics

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
        callback_handler=callback_handler,
    )
    return chat_use_case
