"""Ollama LLM service implementation."""

import logging
import time
from datetime import datetime
from typing import AsyncGenerator, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama.chat_models import ChatOllama

from app.core.settings import Settings
from app.domain.llm.entities.message import Message
from app.domain.llm.entities.model_spec import ModelSpec
from app.domain.llm.exceptions import LLMServiceError
from app.domain.llm.interfaces.llm_service import LLMService
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class OllamaService(LLMService):
    """Ollama implementation of LLM service.

    This service uses LangChain's ChatOllama to interact with Ollama
    instances, providing a clean interface for both streaming and
    non-streaming generation.
    """

    def __init__(self, settings: Settings, callback_handler: Optional[BaseCallbackHandler] = None):
        """Initialize Ollama service with settings.

        Args:
            settings: Application settings containing LLM configuration
            callback_handler: Optional LangChain callback handler for analytics
        """
        self._settings = settings
        self._model_name = settings.llm_model
        self._callback_handler = callback_handler

        # Initialize ChatOllama client with optional callbacks
        callbacks = [callback_handler] if callback_handler else None
        self._llm = ChatOllama(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            callbacks=callbacks,
        )

        # Model information
        self._model_spec = ModelSpec(
            name=settings.llm_model,
            provider="ollama",
            max_tokens=self._get_max_tokens_for_model(settings.llm_model),
            supports_streaming=True,
            supports_function_calling=False,
            temperature_range=(0.0, 2.0),
            metadata={
                "base_url": settings.llm_base_url,
                "temperature": settings.llm_temperature,
            }
        )

    async def generate(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response with automatic callback tracking.

        Args:
            messages: List of messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Ollama-specific parameters

        Returns:
            Generated response text

        Raises:
            LLMServiceError: If generation fails
        """
        try:
            # Convert messages to LangChain format and invoke
            # Callbacks handle all token tracking and timing automatically
            langchain_messages = self._convert_messages_to_langchain(messages)

            # Build generation parameters for runtime configuration
            options = {"temperature": temperature}
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            # Add any additional kwargs to options
            for key, value in kwargs.items():
                if key not in ["temperature", "num_predict"]:
                    options[key] = value

            # Create configured LLM with runtime parameters and callbacks
            configured_llm = self._llm.bind(options=options)

            # Generate response - callbacks handle tracking automatically
            response = await configured_llm.ainvoke(langchain_messages)

            logger.debug(
                "Ollama generation completed: model=%s, response_length=%d",
                self._model_name,
                len(response.content) if response.content else 0,
            )

            return response.content or ""

        except Exception as e:
            error_msg = f"Ollama generation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMServiceError(
                error_msg,
                provider="ollama",
                original_error=e,
            )

    
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
            temperature: Sampling temperature
            **kwargs: Additional Ollama-specific parameters

        Yields:
            Generated text chunks as they are produced

        Raises:
            LLMServiceError: If streaming fails
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages_to_langchain(messages)

            # Build generation parameters
            # Note: In newer versions of langchain-ollama, temperature should be passed
            # through the 'options' parameter to the Ollama client, not directly
            options = {"temperature": temperature}
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            # Add any additional kwargs to options
            for key, value in kwargs.items():
                if key not in ["temperature", "num_predict"]:
                    options[key] = value

            # Create configured LLM with runtime parameters
            configured_llm = self._llm.bind(options=options)

            # Stream response using configured instance
            async for chunk in configured_llm.astream(langchain_messages):
                yield chunk.content

            logger.debug(
                "Ollama streaming completed: model=%s",
                self._model_name,
            )

        except Exception as e:
            error_msg = f"Ollama streaming failed: {str(e)}"
            logger.error(error_msg)
            raise LLMServiceError(
                error_msg,
                provider="ollama",
                original_error=e,
            )

    async def health_check(self) -> bool:
        """Check if the Ollama service is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        start_time = time.time()

        try:
            # Simple health check with minimal prompt
            # Use options parameter for LangChain 1.0 compatibility
            configured_llm = self._llm.bind(options={"num_predict": 1})
            await configured_llm.ainvoke([HumanMessage(content="ping")])

            response_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                "Ollama health check passed: response_time=%.2fms",
                response_time_ms,
            )
            return True

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.warning(
                "Ollama health check failed: error=%s, response_time=%.2fms",
                str(e),
                response_time_ms,
            )
            return False

    def get_model_spec(self) -> ModelSpec:
        """Get information about the current model.

        Returns:
            Model information
        """
        return self._model_spec

    @property
    def provider_name(self) -> str:
        """Get the name of this service provider."""
        return "ollama"

    @property
    def llm_instance(self) -> BaseChatModel:
        """Get the underlying ChatOllama instance for direct operations.

        Returns:
            The ChatOllama client instance
        """
        return self._llm

    def _convert_messages_to_langchain(self, messages: list[Message]) -> list[BaseMessage]:
        """Convert domain messages to LangChain message format.

        Args:
            messages: List of domain messages

        Returns:
            List of LangChain messages
        """
        langchain_messages: list[BaseMessage] = []

        for message in messages:
            if message.role.value == "system":
                langchain_messages.append(SystemMessage(content=message.content))
            elif message.role.value == "user":
                langchain_messages.append(HumanMessage(content=message.content))
            elif message.role.value == "assistant":
                langchain_messages.append(AIMessage(content=message.content))
            else:
                # Default to human message for unknown roles
                logger.warning(
                    "Unknown message role '%s', treating as user message",
                    message.role.value,
                )
                langchain_messages.append(HumanMessage(content=message.content))

        return langchain_messages


    def _get_max_tokens_for_model(self, model_name: str) -> Optional[int]:
        """Get maximum token limit for a model.

        Args:
            model_name: Name of the model

        Returns:
            Maximum token limit if known, None otherwise
        """
        # Known token limits for common models
        token_limits = {
            "llama3.2:3b": 128000,
            "llama3.1:8b": 128000,
            "llama3.1:70b": 128000,
            "llama3:8b": 8192,
            "llama3:70b": 8192,
            "mistral:7b": 8192,
            "codellama:7b": 16384,
            "codellama:13b": 16384,
        }

        return token_limits.get(model_name.lower())

    