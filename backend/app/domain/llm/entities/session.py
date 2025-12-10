"""Chat session entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid

from app.domain.llm.entities.message import Message


@dataclass
class ChatSession:
    """Represents a chat session with conversation history."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    def add_message(self, message: Message) -> None:
        """Add a message to the session.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.last_activity = datetime.now(timezone.utc)

    def get_recent_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get recent messages from the session.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:] if limit > 0 else []

    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if the session is expired.

        Args:
            max_age_hours: Maximum age in hours before considering expired

        Returns:
            True if expired, False otherwise
        """
        age_hours = (datetime.now(timezone.utc) - self.last_activity).total_seconds() / 3600
        return age_hours > max_age_hours

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary representation."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """Create session from dictionary representation."""
        # Parse datetime strings
        created_at = datetime.fromisoformat(data["created_at"])
        last_activity = datetime.fromisoformat(data["last_activity"])

        # Parse messages
        messages = [Message.from_dict(msg_data) for msg_data in data["messages"]]

        return cls(
            session_id=data["session_id"],
            messages=messages,
            created_at=created_at,
            last_activity=last_activity,
            metadata=data.get("metadata"),
        )