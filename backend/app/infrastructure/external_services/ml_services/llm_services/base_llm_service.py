"""Base LLM service interface following rag_phase3.md architecture."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LlmResponse:
    """LLM generation response.

    Attributes:
        content: Generated text content
        model: Model name used for generation
        tokens_usage: Token usage breakdown (prompt_tokens, completion_tokens, total_tokens)
        metadata: Additional response metadata
    """

    content: str
    model: str
    tokens_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmHealthStatus:
    """LLM service health status.

    Attributes:
        is_healthy: Whether the service is healthy
        model: Model name being monitored
        response_time_ms: Health check response time in milliseconds
        error: Error message if unhealthy
    """

    is_healthy: bool
    model: str
    response_time_ms: float
    error: Optional[str] = None


class BaseLlmService(ABC):
    """Base interface for LLM services.

    Provides abstract interface for LLM operations following the repository pattern.
    Supports multiple backends (Ollama, OpenAI, Anthropic, etc.).
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LlmResponse:
        """Generate a single non-streaming response.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Additional model-specific parameters

        Returns:
            LlmResponse with generated content and metadata

        Raises:
            LlmGenerationError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Additional model-specific parameters

        Yields:
            Generated text chunks as they are produced

        Raises:
            LlmGenerationError: If generation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> LlmHealthStatus:
        """Check LLM service health.

        Returns:
            LlmHealthStatus with health information
        """
        pass
