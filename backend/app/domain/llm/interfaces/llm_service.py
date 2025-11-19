"""Abstract LLM service interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional

from app.domain.llm.entities.message import Message
from app.domain.llm.entities.model_spec import ModelSpec
from langchain_core.language_models.chat_models import BaseChatModel   


class LLMService(ABC):
    """Abstract interface for LLM services.

    This interface defines the contract for LLM services, supporting
    both streaming and non-streaming generation with a clean,
    provider-agnostic API.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response.

        Args:
            messages: List of messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Service-specific parameters

        Returns:
            Generated response text

        Raises:
            LLMServiceError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response.

        Args:
            messages: List of messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Service-specific parameters

        Yields:
            Generated text chunks as they are produced

        Raises:
            LLMServiceError: If streaming fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_model_spec(self) -> ModelSpec:
        """Get information about the current model.

        Returns:
            Model information including name, capabilities, etc.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this service provider."""
        pass

    @property
    @abstractmethod
    def llm_instance(self) -> BaseChatModel:
        """Get the underlying LLM instance for direct operations.

        Returns:
            The LLM client instance (e.g., ChatOllama, OpenAI client)
        """
        pass
