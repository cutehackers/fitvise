"""LLM services package."""

from app.infrastructure.external_services.ml_services.llm_services.ollama_service import (
    OllamaService,
)

__all__ = [
    "OllamaService",
]
