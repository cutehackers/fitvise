"""LangChain LLM adapter for custom LLM provider integration.

This module provides a proper LangChain-compatible adapter for our custom LLM provider,
following LangChain's recommended patterns for custom LLM integration.
"""

import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from pydantic import ConfigDict
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.domain.entities.message_role import MessageRole
from app.domain.llm.entities.message import Message
from app.domain.llm.interfaces.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class LangChainLLMAdapter(BaseChatModel):
    """LangChain-compatible adapter for our custom LLM provider.

    This adapter allows our custom LLM provider to work seamlessly with LangChain's
    ecosystem, including chains, agents, and other LangChain components.

    Attributes:
        llm_provider: The underlying LLM provider implementation
        model_name: Name of the model for LangChain tracking
    """

    llm_provider: LLMProvider
    """The underlying LLM provider instance."""

    model_name: str = "custom-llm"
    """Model name for LangChain tracking and callbacks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, llm_provider: LLMProvider, **kwargs: Any):
        """Initialize the LangChain LLM adapter.

        Args:
            llm_provider: The LLM provider instance to wrap
            **kwargs: Additional arguments passed to BaseChatModel
        """
        super().__init__(llm_provider=llm_provider, **kwargs)
        model_info = llm_provider.get_model_info()
        self.model_name = model_info.name

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for LangChain tracking."""
        return "custom-llm-provider"

    def _convert_messages_to_domain(self, messages: List[BaseMessage]) -> List[Message]:
        """Convert LangChain messages to domain messages.

        Args:
            messages: List of LangChain BaseMessage objects

        Returns:
            List of domain Message objects
        """
        domain_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                domain_messages.append(
                    Message(content=str(msg.content), role=MessageRole.USER)
                )
            elif isinstance(msg, AIMessage):
                domain_messages.append(
                    Message(content=str(msg.content), role=MessageRole.ASSISTANT)
                )
            elif isinstance(msg, SystemMessage):
                domain_messages.append(
                    Message(content=str(msg.content), role=MessageRole.SYSTEM)
                )
            elif isinstance(msg, ToolMessage):
                domain_messages.append(
                    Message(content=str(msg.content), role=MessageRole.TOOL)
                )
            else:
                logger.warning(
                    "Unknown LangChain message type: %s, treating as user message",
                    type(msg).__name__,
                )
                domain_messages.append(
                    Message(content=str(msg.content), role=MessageRole.USER)
                )
        return domain_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation (not supported for async provider).

        Args:
            messages: List of messages to generate from
            stop: Stop sequences (not currently used)
            run_manager: Callback manager for tracking
            **kwargs: Additional generation parameters

        Raises:
            NotImplementedError: This provider only supports async generation
        """
        raise NotImplementedError(
            "Synchronous generation not supported. Use async methods (agenerate, astream) instead."
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from messages asynchronously.

        Args:
            messages: List of messages to generate from
            stop: Stop sequences (not currently used)
            run_manager: Callback manager for tracking
            **kwargs: Additional generation parameters

        Returns:
            ChatResult with generated message and metadata
        """
        # Convert to domain messages
        domain_messages = self._convert_messages_to_domain(messages)

        # Generate response
        response_content = await self.llm_provider.generate(domain_messages, **kwargs)

        # Create LangChain message
        message = AIMessage(content=response_content)

        # Track with callback manager if provided
        if run_manager:
            await run_manager.on_llm_new_token(response_content)

        # Create generation result
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming (not supported for async provider).

        Args:
            messages: List of messages to stream from
            stop: Stop sequences (not currently used)
            run_manager: Callback manager for tracking
            **kwargs: Additional generation parameters

        Raises:
            NotImplementedError: This provider only supports async streaming
        """
        raise NotImplementedError(
            "Synchronous streaming not supported. Use async methods (astream) instead."
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream response from messages asynchronously.

        Args:
            messages: List of messages to stream from
            stop: Stop sequences (not currently used)
            run_manager: Callback manager for tracking
            **kwargs: Additional generation parameters

        Yields:
            ChatGenerationChunk objects with incremental content
        """
        # Convert to domain messages
        domain_messages = self._convert_messages_to_domain(messages)

        # Stream responses
        async for chunk in self.llm_provider.generate_stream(domain_messages, **kwargs):
            # Create chunk message (use AIMessageChunk for streaming)
            chunk_message = AIMessageChunk(content=chunk)

            # Track with callback manager if provided
            if run_manager:
                await run_manager.on_llm_new_token(chunk)

            # Yield generation chunk
            yield ChatGenerationChunk(message=chunk_message)
