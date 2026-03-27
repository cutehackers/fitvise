"""Abstract chat orchestrator interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from app.schemas.chat import ChatRequest, ChatResponse


class ChatOrchestrator(ABC):
    """Abstract interface for chat orchestration.

    This interface handles the high-level chat logic including
    session management, conversation history, and response streaming.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get the conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Chat message history for the session

        Raises:
            ValueError: If session_id is invalid
        """
        pass

    @abstractmethod
    async def clear_session(self, session_id: str) -> bool:
        """Clear a chat session.

        Args:
            session_id: Session to clear

        Returns:
            True if session was cleared, False if it didn't exist
        """
        pass

    @abstractmethod
    async def get_active_session_count(self) -> int:
        """Get the number of active chat sessions.

        Returns:
            Number of active sessions
        """
        pass

    @abstractmethod
    async def clear_all_sessions(self) -> int:
        """Clear all chat sessions.

        Returns:
            Number of sessions that were cleared
        """
        pass

    @abstractmethod
    def get_turns_window(self) -> int:
        """Get the configured turns window.

        Returns:
            Number of conversation turns kept in memory
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the chat orchestrator is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass