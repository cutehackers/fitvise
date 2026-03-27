"""Factory helpers for the canonical BotAdvisor LLM path."""

from __future__ import annotations

from typing import Any

from botadvisor.app.core.config import get_settings
from botadvisor.app.llm.ollama import OllamaChatService


def create_llm_service(settings: Any):
    """Create the configured LLM service."""
    provider = getattr(settings, "llm_provider", "ollama")

    if provider == "ollama":
        return OllamaChatService(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unsupported llm provider: {provider}")


def get_llm_service(settings: Any | None = None):
    """Return the runtime LLM service using explicit or global settings."""
    return create_llm_service(settings or get_settings())
