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
from app.domain.llm.interfaces.llm_provider import LLMProvider
from app.domain.entities.message_role import MessageRole
from app.schemas.chat import ChatRequest, ChatResponse
from app.infrastructure.llm.adapters import LangChainLLMAdapter

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
        llm_provider: LLMProvider,
        turns_window: int = 10,
        max_session_age_hours: int = 24,
    ):
        """Initialize the chat orchestrator.

        Args:
            llm_provider: LLM provider for generating responses
            turns_window: Number of conversation turns to keep in memory (default: 10)
            max_session_age_hours: Maximum age before sessions expire (default: 24)
        """
        self._llm_provider = llm_provider
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

        # Message trimmer
        self._trimmer = trim_messages(
            max_tokens=MAX_TOKENS_TABLE.get(
                llm_provider.get_model_info().name.lower(),
                8192
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

        # Create LangChain-compatible LLM adapter
        self._llm_adapter = LangChainLLMAdapter(llm_provider=llm_provider)

        # Create chain with memory
        # self.trimmer outputs a list (trimmed messages). ChatPromptTemplate expects a dict.
        # So we transform the list into a dict with "history" as the key.
        # The chain will use the trimmed history as input.
        # The prompt expects a "history" key, which will be filled with the trimmed messages.
        # The LLM will then generate a response based on the trimmed history.
        self._chain = (
            RunnablePassthrough.assign(
                history=itemgetter("history") | self._trimmer
            ) | self._prompt | self._llm_adapter
        )

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

            # Create runnable with history
            # Use get_session_history method instead of dict access to ensure session exists
            runnable_with_history = RunnableWithMessageHistory(
                self._chain,
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
                        model=self._llm_provider.get_model_info().name,
                        message=request.message.model_copy(
                            update={"role": MessageRole.ASSISTANT.value, "content": chunk.content}
                        ),
                        done=False,
                        created_at=_get_current_timestamp(),
                    )

            # Send final response
            yield ChatResponse(
                model=self._llm_provider.get_model_info().name,
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
            provider_healthy = await self._llm_provider.health_check()
            if not provider_healthy:
                return False

            # Check session storage (simple count check)
            session_count = len(self._session_store)
            logger.debug("Chat orchestrator health: %d active sessions", session_count)

            return True

        except Exception as e:
            logger.error("Chat orchestrator health check failed: %s", str(e))
            return False

    def _create_token_counter(self):
        """Create a token counter function for the trimmer."""
        def count_tokens(messages: List[BaseMessage]) -> int:
            # Simple estimation - in production, use actual tokenizer
            total_chars = sum(len(msg.content) for msg in messages if hasattr(msg, 'content'))
            return total_chars // 4  # Rough estimate: 1 token ≈ 4 characters

        return count_tokens
    