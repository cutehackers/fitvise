"""Chat use case for basic chat functionality (Phase 3 refactoring).

This module implements the ChatUseCase that handles basic chat without RAG
by directly integrating with LangChain components and session management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Dict, Any, Optional

from langchain_core.messages import (
    BaseMessage,
    trim_messages,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler

from app.domain.llm.exceptions import ChatOrchestratorError, MessageValidationError
from app.domain.llm.interfaces.llm_service import LLMService
from app.domain.entities.message_role import MessageRole
from app.domain.services.session_service import SessionService
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


# Token limits for common models
MAX_TOKENS_TABLE = {
    "llama3.2:3b": 128000,
    "llama3.1:8b": 128000,
    "llama3.1:70b": 128000,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "mistral:7b": 8192,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
}

# Default max token length for models without specific limit
DEFAULT_MAX_TOKEN_LENGTH = 8192


def _get_current_timestamp() -> str:
    """Generate current timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ChatUseCase:
    """Chat use case for basic chat functionality.

    Handles chat conversations without document retrieval using direct
    LangChain integration and clean architecture principles.

    Responsibilities:
    - Session management through SessionService
    - LLM interaction through LLMService
    - LangChain integration for message history and trimming
    - Streaming response generation
    """

    def __init__(
        self,
        llm_service: LLMService,
        session_service: SessionService,
        turns_window: int = 10,
        max_session_age_hours: int = 24,
        callback_handler: Optional[BaseCallbackHandler] = None,
    ) -> None:
        """Initialize the chat use case.

        Args:
            llm_service: LLM service for generating responses
            session_service: Service for session management
            turns_window: Number of conversation turns to keep in memory
            max_session_age_hours: Maximum age before sessions expire
            callback_handler: Optional LangChain callback handler for observability
        """
        self._llm_service = llm_service
        self._session_service = session_service
        self._turns_window = turns_window
        self._max_session_age_hours = max_session_age_hours
        self._callback_handler = callback_handler
        self._callbacks = [callback_handler] if callback_handler else None

        # Basic chat prompt template
        self._prompt = ChatPromptTemplate([
            ("system", "You are a helpful AI fitness assistant. Provide accurate and helpful responses to fitness-related questions."),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])

        # Store LLM instance for direct access
        self._llm = llm_service.llm_instance

        self._trimmer = trim_messages(
            max_tokens=MAX_TOKENS_TABLE.get(
                llm_service.get_model_spec().name.lower(),
                DEFAULT_MAX_TOKEN_LENGTH
            ),
            token_counter=self._llm,
            strategy="last",
            start_on="human",
            include_system=True,
        )

        # Performance optimization: Only apply trimming for conversations with >20 messages
        self._trim_messages_threshold = 20

        logger.info(
            "ChatUseCase initialized with LLM service and session service"
        )

    def _should_apply_trimming(self, messages: List[BaseMessage]) -> bool:
        """Determine if message trimming should be applied.

        Args:
            messages: List of messages to evaluate

        Returns:
            True if trimming should be applied, False otherwise
        """
        return len(messages) > self._trim_messages_threshold

    def _get_smart_trimmer(self, messages: List[BaseMessage]) -> Any:
        """Get conditional trimmer for conversation length.

        Args:
            messages: List of messages to potentially trim

        Returns:
            Either the trimmer or identity function based on conversation length
        """
        if self._should_apply_trimming(messages):
            return self._trimmer
        return lambda x: x

    def _ensure_chat_request(self, request: ChatRequest) -> None:
        """Validate a chat request.

        Args:
            request: Chat request to validate

        Raises:
            MessageValidationError: If request is invalid
        """
        if not request.message:
            raise MessageValidationError("Message is required")

        if not request.message.content or not request.message.content.strip():
            raise MessageValidationError("Message content cannot be empty", field="content")

        if not request.session_id or not request.session_id.strip():
            raise MessageValidationError("Session ID is required", field="session_id")

    async def chat(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Process a chat message and generate streaming responses.

        Args:
            request: Chat request with message and session info

        Yields:
            Streaming chat responses

        Raises:
            MessageValidationError: If request validation fails
            ChatOrchestratorError: If chat processing fails
            Exception: For unexpected errors (wrapped as ChatOrchestratorError)
        """
        try:
            # Ensure a session exists (generate one for first-time chats)
            session_id, session_history = self._session_service.ensure_session(request.session_id)
            if request.session_id != session_id:
                request = request.model_copy(update={"session_id": session_id})

            # Validate request using the same pattern as original orchestrators
            self._ensure_chat_request(request)

            # Apply conditional trimming
            smart_trimmer = self._get_smart_trimmer(session_history.messages)

            if smart_trimmer is self._trimmer:
                # For long conversations: trim history before passing to prompt
                trimmed_history = smart_trimmer.invoke(session_history.messages)
                chain = (
                    RunnablePassthrough.assign(history=lambda x: trimmed_history)
                    | self._prompt 
                    | self._llm
                )
            else:
                # For short conversations: pass input directly to prompt
                # The prompt template handles history through MessagesPlaceholder
                chain = self._prompt | self._llm

            # Create runnable with history using SessionService's persistent history
            runnable_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: self._session_service.get_session_history(session_id),
                input_messages_key="input",
                history_messages_key="history",
            )

            # Process message and stream response
            config = {"configurable": {"session_id": session_id}}
            if self._callbacks:
                config["callbacks"] = self._callbacks

            async for chunk in runnable_with_history.astream(
                {"input": request.message.content},
                config=config,
            ):
                if isinstance(chunk, BaseMessage) and chunk.content and chunk.content.strip():
                    # Stream response chunk
                    yield ChatResponse(
                        model=self._llm_service.get_model_spec().name,
                        message=request.message.model_copy(
                            update={"role": MessageRole.ASSISTANT.value, "content": chunk.content}
                        ),
                        done=False,
                        session_id=session_id,
                        created_at=_get_current_timestamp(),
                    )

            # Send final response (empty message with done=True)
            yield ChatResponse(
                model=self._llm_service.get_model_spec().name,
                done=True,
                session_id=session_id,
                created_at=_get_current_timestamp(),
            )

        except (MessageValidationError, ChatOrchestratorError) as e:
            # Re-raise domain exceptions for the API layer to handle
            raise e
        except Exception as e:
            logger.error(
                "Error processing chat message for session %s: %s",
                session_id,
                str(e),
            )
            # Log detailed exception info for debugging
            logger.error(
                "Exception type: %s, Exception args: %s, Request message object: %s",
                type(e).__name__,
                str(e.args),
                f"message={request.message}, content='{request.message.content if request.message else None}'" if request.message else "request.message is None"
            )
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            # Wrap unexpected exceptions in ChatOrchestratorError
            raise ChatOrchestratorError(
                f"Failed to process chat message: {str(e)}",
                session_id=session_id,
                original_error=e,
            )

    async def clear_session(self, session_id: str) -> bool:
        """Clear a specific chat session.

        Args:
            session_id: The session ID to clear

        Returns:
            True if session was found and cleared, False otherwise
        """
        return self._session_service.clear_session(session_id)

    async def get_active_session_count(self) -> int:
        """Get the current number of active sessions.

        Returns:
            Number of active chat sessions
        """
        return self._session_service.get_active_session_count()

    async def clear_all_sessions(self) -> int:
        """Clear all chat sessions.

        Returns:
            Number of sessions that were cleared
        """
        return self._session_service.cleanup_expired_sessions()

    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session.

        Args:
            session_id: Session ID to get statistics for

        Returns:
            Dictionary with session statistics
        """
        return self._session_service.get_session_statistics(session_id)

    async def health_check(self) -> bool:
        """Check if the chat use case is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check LLM provider health
            llm_service_healthy = await self._llm_service.health_check()
            if not llm_service_healthy:
                logger.warning("ChatUseCase health check: LLM service unhealthy")
                return False

            session_count = self._session_service.get_active_session_count()
            logger.debug(
                "ChatUseCase health: LLM OK, %d active sessions",
                session_count,
            )

            return True

        except Exception as e:
            logger.error("ChatUseCase health check failed: %s", str(e))
            return False
