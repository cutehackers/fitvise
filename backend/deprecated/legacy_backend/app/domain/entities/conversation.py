"""Conversation entity for chat management (Phase 3 refactoring).

This module defines the Conversation entity that manages chat conversations
with context, sessions, and retrieval information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.retrieval_context import RetrievalContext
from app.domain.llm.entities.session import ChatSession
from app.domain.llm.entities.message import Message
from app.domain.entities.message_role import MessageRole


@dataclass
class Conversation:
    """Domain entity representing a chat conversation.

    Manages conversation state, context history, and retrieval information
    for chat interactions with document context.

    Attributes:
        conversation_id: Unique identifier for this conversation
        session: The underlying chat session with messages
        context_history: History of retrieval contexts used in this conversation
        current_context: Current active retrieval context
        metadata: Additional conversation metadata
        created_at: When the conversation was created
        last_activity: Last activity timestamp
        is_active: Whether the conversation is currently active
    """

    conversation_id: UUID
    session: ChatSession
    context_history: List[RetrievalContext] = field(default_factory=list)
    current_context: Optional[RetrievalContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def __post_init__(self) -> None:
        """Validate conversation after initialization."""
        if not self.session:
            raise ValueError("Chat session cannot be None")

    @classmethod
    def create_new(cls, metadata: Optional[Dict[str, Any]] = None) -> Conversation:
        """Create a new conversation with a fresh session.

        Args:
            metadata: Optional conversation metadata

        Returns:
            New Conversation instance
        """
        session = ChatSession()
        return Conversation(
            conversation_id=UUID(),
            session=session,
            metadata=metadata or {},
        )

    @classmethod
    def from_session(cls, session: ChatSession, metadata: Optional[Dict[str, Any]] = None) -> Conversation:
        """Create conversation from existing chat session.

        Args:
            session: Existing chat session
            metadata: Optional conversation metadata

        Returns:
            New Conversation instance
        """
        return Conversation(
            conversation_id=UUID(),
            session=session,
            metadata=metadata or {},
            created_at=session.created_at,
            last_activity=session.last_activity,
        )

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a user message to the conversation.

        Args:
            content: Message content
            metadata: Optional message metadata

        Returns:
            Created user message
        """
        message = Message(
            content=content,
            role=MessageRole.USER,
            metadata=metadata,
        )
        self.session.add_message(message)
        self._update_activity()
        return message

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add an assistant message to the conversation.

        Args:
            content: Message content
            metadata: Optional message metadata

        Returns:
            Created assistant message
        """
        message = Message(
            content=content,
            role=MessageRole.ASSISTANT,
            metadata=metadata,
        )
        self.session.add_message(message)
        self._update_activity()
        return message

    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a system message to the conversation.

        Args:
            content: Message content
            metadata: Optional message metadata

        Returns:
            Created system message
        """
        message = Message(
            content=content,
            role=MessageRole.SYSTEM,
            metadata=metadata,
        )
        self.session.add_message(message)
        self._update_activity()
        return message

    def set_current_context(self, context: RetrievalContext) -> None:
        """Set the current retrieval context.

        Args:
            context: The retrieval context to set as current
        """
        self.current_context = context
        self.context_history.append(context)
        self._update_activity()

    def clear_current_context(self) -> None:
        """Clear the current retrieval context."""
        self.current_context = None
        self._update_activity()

    def get_recent_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get recent messages from the conversation.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        return self.session.get_recent_messages(limit)

    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation.

        Returns:
            Last message or None if no messages
        """
        messages = self.session.messages
        return messages[-1] if messages else None

    def get_last_user_message(self) -> Optional[Message]:
        """Get the last user message in the conversation.

        Returns:
            Last user message or None if no user messages
        """
        for message in reversed(self.session.messages):
            if message.role == MessageRole.USER:
                return message
        return None

    def message_count(self) -> int:
        """Get the total number of messages in the conversation.

        Returns:
            Total message count
        """
        return len(self.session.messages)

    def user_message_count(self) -> int:
        """Get the number of user messages.

        Returns:
            User message count
        """
        return sum(1 for msg in self.session.messages if msg.role == MessageRole.USER)

    def assistant_message_count(self) -> int:
        """Get the number of assistant messages.

        Returns:
            Assistant message count
        """
        return sum(1 for msg in self.session.messages if msg.role == MessageRole.ASSISTANT)

    def has_context(self) -> bool:
        """Check if conversation has current context.

        Returns:
            True if current context exists
        """
        return self.current_context is not None

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context.

        Returns:
            Context summary dictionary
        """
        if not self.current_context:
            return {"has_context": False}

        return {
            "has_context": True,
            "context_summary": self.current_context.get_summary(),
            "document_count": len(self.current_context.document_references),
        }

    def deactivate(self) -> None:
        """Deactivate the conversation."""
        self.is_active = False
        self._update_activity()

    def activate(self) -> None:
        """Activate the conversation."""
        self.is_active = True
        self._update_activity()

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if the conversation is expired.

        Args:
            max_age_hours: Maximum age in hours before considering expired

        Returns:
            True if expired, False otherwise
        """
        return self.session.is_expired(max_age_hours)

    def _update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get comprehensive conversation summary.

        Returns:
            Dictionary with conversation statistics
        """
        return {
            "conversation_id": str(self.conversation_id),
            "session_id": self.session.session_id,
            "message_count": self.message_count(),
            "user_message_count": self.user_message_count(),
            "assistant_message_count": self.assistant_message_count(),
            "context_history_count": len(self.context_history),
            "has_current_context": self.has_context(),
            "context_summary": self.get_context_summary(),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "age_hours": (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Convert Conversation to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "conversation_id": str(self.conversation_id),
            "session": self.session.to_dict(),
            "context_history": [ctx.as_dict() for ctx in self.context_history],
            "current_context": self.current_context.as_dict() if self.current_context else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "summary": self.get_conversation_summary(),
        }

    def __str__(self) -> str:
        """String representation of conversation."""
        return (
            f"Conversation(id={self.conversation_id}, "
            f"messages={self.message_count()}, "
            f"contexts={len(self.context_history)}, "
            f"active={self.is_active})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Conversation(conversation_id={self.conversation_id}, "
            f"session={self.session}, "
            f"context_history={len(self.context_history)}, "
            f"current_context={'has' if self.current_context else 'no'}, "
            f"is_active={self.is_active})"
        )