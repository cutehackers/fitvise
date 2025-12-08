"""Session service for managing chat sessions (Phase 3 refactoring).

This module defines the SessionService domain service that manages chat sessions
using LangChain's BaseChatMessageHistory for consistent session management
across all chat implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

# Import LangChain for history management
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


@dataclass
class SessionConfig:
    """Configuration for session management operations.

    Attributes:
        max_session_age_hours: Maximum age before sessions expire
        max_messages_per_session: Maximum messages to keep in session
        turns_window: Number of conversation turns to keep in memory
        auto_cleanup_expired: Whether to automatically cleanup expired sessions
    """

    max_session_age_hours: int = 24
    max_messages_per_session: int = 100
    turns_window: int = 10
    auto_cleanup_expired: bool = True


class SessionService:
    """Domain service for managing chat sessions.

    Provides session lifecycle management using LangChain's BaseChatMessageHistory
    for consistent session management across all chat implementations.

    Responsibilities:
    - Create and manage LangChain chat sessions
    - Track conversation history through LangChain's history interface
    - Handle session cleanup and statistics
    - Provide session querying capabilities

    Examples:
        >>> service = SessionService()
        >>> history = service.get_session_history("session_123")
        >>> service.clear_session("session_123")
        >>> count = service.get_active_session_count()
    """

    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        """Initialize session service.

        Args:
            config: Optional session management configuration
        """
        self._config = config or SessionConfig()

        # LangChain history management (matching LangChainOrchestrator pattern)
        self._session_store: Dict[str, BaseChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get session history (matching LangChainOrchestrator interface exactly).

        Args:
            session_id: The session ID to get history for

        Returns:
            BaseChatMessageHistory for the session

        Raises:
            ValueError: If session_id is invalid
        """
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty or None")

        session_id = session_id.strip()

        # Return existing history if available
        if session_id in self._session_store:
            return self._session_store[session_id]

        # Create new LangChain history
        self._session_store[session_id] = InMemoryChatMessageHistory()
        return self._session_store[session_id]

    def add_user_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message to the session history.

        Args:
            session_id: The session ID to add message to
            content: Message content
            metadata: Optional message metadata (not used in LangChain history)
        """
        history = self.get_session_history(session_id)
        history.add_user_message(HumanMessage(content=content))
        self._trim_history(history)

    def add_assistant_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an assistant message to the session history.

        Args:
            session_id: The session ID to add message to
            content: Message content
            metadata: Optional message metadata (not used in LangChain history)
        """
        history = self.get_session_history(session_id)
        history.add_ai_message(AIMessage(content=content))
        self._trim_history(history)

    def add_system_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a system message to the session history.

        Args:
            session_id: The session ID to add message to
            content: Message content
            metadata: Optional message metadata (not used in LangChain history)
        """
        history = self.get_session_history(session_id)
        history.add_message(SystemMessage(content=content))
        self._trim_history(history)

    def clear_session(self, session_id: str) -> bool:
        """Clear session history for a specific session.

        Args:
            session_id: The session ID to clear

        Returns:
            True if session was found and cleared, False otherwise
        """
        if session_id in self._session_store:
            del self._session_store[session_id]
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely (alias for clear_session).

        Args:
            session_id: The session ID to delete

        Returns:
            True if session was found and deleted, False otherwise
        """
        return self.clear_session(session_id)

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions (simplified - no expiration tracking).

        Returns:
            Always returns 0 as we don't track session expiration
        """
        # Simplified: We don't track session expiration in LangChain history mode
        # Sessions remain in memory until explicitly cleared
        return 0

    def get_active_session_count(self) -> int:
        """Get the current number of active sessions.

        Returns:
            Number of active chat sessions
        """
        return len(self._session_store)

    def get_total_session_count(self) -> int:
        """Get the total number of sessions.

        Returns:
            Total number of sessions (same as active count)
        """
        return len(self._session_store)

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session.

        Args:
            session_id: The session ID to get statistics for

        Returns:
            Dictionary with session statistics
        """
        if session_id not in self._session_store:
            return {"error": "Session not found"}

        history = self._session_store[session_id]
        messages = history.messages

        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        assistant_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "system_message_count": len(system_messages),
            "langchain_history": True,
        }

    def _trim_history(self, history: BaseChatMessageHistory) -> None:
        """Trim session history according to configured limits.

        Keeps the most recent turns (user/assistant) while preserving system messages.
        """
        messages = list(history.messages)

        # First, enforce turns_window on non-system messages
        if self._config.turns_window > 0:
            max_non_system = self._config.turns_window * 2
            non_system_kept = 0
            trimmed: List = []

            # Walk from newest to oldest to keep the most recent turns
            for msg in reversed(messages):
                if isinstance(msg, SystemMessage):
                    trimmed.append(msg)
                    continue
                if non_system_kept < max_non_system:
                    trimmed.append(msg)
                    non_system_kept += 1
            trimmed.reverse()
            messages = trimmed

        # Then, enforce absolute max_messages_per_session (prefer dropping non-system first)
        max_msgs = self._config.max_messages_per_session
        if max_msgs > 0 and len(messages) > max_msgs:
            drop = len(messages) - max_msgs
            trimmed_messages: List = []
            for msg in messages:
                if drop > 0 and not isinstance(msg, SystemMessage):
                    drop -= 1
                    continue
                trimmed_messages.append(msg)
            # If we still need to drop (all were system), drop from the front
            if drop > 0:
                trimmed_messages = trimmed_messages[drop:]
            messages = trimmed_messages

        # Replace underlying history if changed
        history.messages = messages

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global session statistics.

        Returns:
            Dictionary with global session statistics
        """
        total_sessions = len(self._session_store)
        total_messages = sum(len(history.messages) for history in self._session_store.values())

        return {
            "total_sessions": total_sessions,
            "active_sessions": total_sessions,  # All sessions are considered active
            "expired_sessions": 0,
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0,
            "langchain_history": True,
            "auto_cleanup_enabled": self._config.auto_cleanup_expired,
        }
