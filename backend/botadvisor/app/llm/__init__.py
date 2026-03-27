"""Canonical LLM boundary for BotAdvisor."""

from botadvisor.app.llm.factory import create_llm_service, get_llm_service
from botadvisor.app.llm.ollama import OllamaChatService

__all__ = ["OllamaChatService", "create_llm_service", "get_llm_service"]
