"""LangChain-based chat orchestrator implementation."""

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Dict, Any, Optional

from langchain_core.messages import (
    BaseMessage,
    trim_messages,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from operator import itemgetter

from app.domain.llm.entities.message import Message
from app.domain.llm.exceptions import ChatOrchestratorError, SessionNotFoundError, MessageValidationError
from app.domain.llm.interfaces.chat_orchestrator import ChatOrchestrator
from app.domain.llm.interfaces.llm_service import LLMService
from app.domain.entities.message_role import MessageRole
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
    """Generate current timestamp in ISO 8601 format.

    Returns:
        Current timestamp in ISO 8601 format (UTC)
    """
    return datetime.now(timezone.utc).isoformat()


class LangChainOrchestrator(ChatOrchestrator):
    """LangChain-based implementation of chat orchestration.

    This orchestrator handles chat sessions, conversation history,
    and response streaming using LangChain components.
    """

    def __init__(
        self,
        llm_service: LLMService,
        turns_window: int = 10,
        max_session_age_hours: int = 24,
    ):
        """Initialize the chat orchestrator.

        Args:
            llm_service: LLM service for generating responses
            turns_window: Number of conversation turns to keep in memory (default: 10)
            max_session_age_hours: Maximum age before sessions expire (default: 24)
        """
        self._llm_service = llm_service
        self._turns_window = turns_window
        self._max_session_age_hours = max_session_age_hours

        # Session store using LangChain's InMemoryChatMessageHistory
        self._session_store: Dict[str, BaseChatMessageHistory] = {}

        # Chat prompt template
        self._prompt = ChatPromptTemplate([
            ("system", "You are a helpful and versatile AI assistant. Answer user questions thoroughly and thoughtfully."),
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
            token_counter=self._create_token_counter(),
            strategy="last",
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
        )

        # Performance optimization: Only apply trimming for conversations with more than 20 messages
        # This avoids expensive token counting and processing for short conversations
        self._trim_messages_threshold = 20

        # Use ChatOllama instance directly (no adapter needed)
        # The OllamaService's llm_instance returns a LangChain-compatible ChatOllama

        # Create optimized prompt template with built-in history handling
        # This eliminates the need for complex RunnablePassthrough.assign
        self._prompt = ChatPromptTemplate([
            ("system", "You are a helpful and versatile AI assistant. Answer user questions thoroughly and thoughtfully."),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])

    def _should_apply_trimming(self, messages: List[BaseMessage]) -> bool:
        """
        Determine if message trimming should be applied based on conversation length.

        Performance optimization: Skip trimming for short conversations to avoid
        expensive token counting and processing overhead.

        Args:
            messages: List of messages to evaluate

        Returns:
            True if trimming should be applied, False otherwise
        """
        # Only apply trimming for conversations with more than the threshold number of messages
        return len(messages) > self._trim_messages_threshold

    def _get_smart_trimmer(self, messages: List[BaseMessage]) -> Any:
        """
        Get conditional trimmer that skips processing for short conversations.

        Args:
            messages: List of messages to potentially trim

        Returns:
            Either the trimmer or identity function (no-op) based on conversation length
        """
        if self._should_apply_trimming(messages):
            return self._trimmer
        # Return identity function for short conversations - no trimming needed
        return lambda x: x

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
            ChatOrchestratorError: If message processing fails
        """
        try:
            # Validate request
            self._ensure_chat_request(request)

            # Get session history (creates new session if doesn't exist)
            session_history = self.get_session_history(request.session_id)

            # Get current session history and apply conditional trimming
            smart_trimmer = self._get_smart_trimmer(session_history.messages)

            # Create simplified chain directly
            # This eliminates the complex RunnablePassthrough.assign overhead
            # The prompt template now handles history internally through MessagesPlaceholder
            if smart_trimmer is self._trimmer:
                # Apply trimming for long conversations before passing to prompt
                trimmed_history = smart_trimmer(session_history.messages)
                chain = (
                    lambda x: {"input": x["input"], "history": trimmed_history}
                ) | self._prompt | self._llm
            else:
                # Direct input for short conversations - no expensive preprocessing
                chain = (
                    lambda x: x
                ) | self._prompt | self._llm

            # Create runnable with history
            # Use get_session_history method instead of dict access to ensure session exists
            runnable_with_history = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )

            # Process message and stream response
            config = {"configurable": {"session_id": request.session_id}}
            full_response = ""

            async for chunk in runnable_with_history.astream(
                {"input": request.message.content},
                config=config,
            ):
                if isinstance(chunk, BaseMessage) and chunk.content:
                    full_response += chunk.content

                    # Stream response chunk
                    yield ChatResponse(
                        model=self._llm_service.get_model_spec().name,
                        message=request.message.model_copy(
                            update={"role": MessageRole.ASSISTANT.value, "content": chunk.content}
                        ),
                        done=False,
                        created_at=_get_current_timestamp(),
                    )

            # Send final response
            yield ChatResponse(
                model=self._llm_service.get_model_spec().name,
                done=True,
                created_at=_get_current_timestamp(),
            )

        except Exception as e:
            logger.error(
                "Error processing chat message for session %s: %s",
                request.session_id,
                str(e),
            )
            raise ChatOrchestratorError(
                f"Failed to process chat message: {str(e)}",
                session_id=request.session_id,
                original_error=e,
            )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieve the chat message history for a given session.

        If the session does not exist in the session store, a new InMemoryChatMessageHistory
        instance is created and associated with the session ID.

        Note: This method is synchronous because it only manages in-memory state
        and doesn't perform any I/O operations. This allows it to be used directly
        by LangChain's RunnableWithMessageHistory.

        Args:
            session_id: The unique identifier for the chat session.

        Returns:
            BaseChatMessageHistory: The chat message history associated with the session.

        Raises:
            ValueError: If session_id is invalid or empty.
        """
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty or None")

        session_id = session_id.strip()

        if session_id not in self._session_store:
            try:
                self._session_store[session_id] = InMemoryChatMessageHistory()
                logger.info("New chat session: %s", session_id)
            except Exception as e:
                logger.error("Failed to initialize session %s: %s", session_id, str(e))
                raise ValueError(f"Failed to initialize chat session: {str(e)}") from e

        return self._session_store[session_id]

    async def clear_session(self, session_id: str) -> bool:
        """Clear a specific chat session from memory.

        Args:
            session_id: The session ID to clear

        Returns:
            True if session was found and cleared, False if session didn't exist
        """
        if session_id in self._session_store:
            del self._session_store[session_id]
            logger.info("Cleared chat session: %s", session_id)
            return True
        return False

    async def get_active_session_count(self) -> int:
        """Get the current number of active sessions.

        Returns:
            Number of active chat sessions
        """
        return len(self._session_store)

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

    async def clear_all_sessions(self) -> int:
        """Clear all chat sessions from memory.

        Returns:
            int: Number of sessions that were cleared
        """
        session_count = len(self._session_store)
        self._session_store.clear()
        logger.info("Cleared %d chat sessions", session_count)
        return session_count

    def get_turns_window(self) -> int:
        """Get the configured turns window for conversation history.

        Returns:
            Number of conversation turns kept in memory
        """
        return self._turns_window
        
    async def health_check(self) -> bool:
        """Check if the chat orchestrator is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check LLM provider health
            llm_service_healthy = await self._llm_service.health_check()
            if not llm_service_healthy:
                return False

            # Check session storage (simple count check)
            session_count = len(self._session_store)
            logger.debug("Chat orchestrator health: %d active sessions", session_count)

            return True

        except Exception as e:
            logger.error("Chat orchestrator health check failed: %s", str(e))
            return False

    def _create_token_counter(self):
        """Create a token counter function using stored LLM instance directly."""
        def count_tokens(messages: List[BaseMessage]) -> int:
            # Use self._llm directly for cleaner dependency access
            if hasattr(self._llm, 'get_num_tokens_from_messages'):
                return self._llm.get_num_tokens_from_messages(messages)
            elif hasattr(self._llm, 'get_num_tokens'):
                # Fallback for individual message token counting
                return sum(self._llm.get_num_tokens(msg) for msg in messages if hasattr(msg, 'content'))
            else:
                # Final fallback to simple character-based estimation
                total_chars = sum(len(msg.content) for msg in messages if hasattr(msg, 'content'))
                return total_chars // 4

        return count_tokens
    