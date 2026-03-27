"""Message entities for LLM interactions."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

from app.domain.entities.message_role import MessageRole


@dataclass
class Message:
    """Represents a message in a conversation."""

    content: str
    role: MessageRole
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")

        if not isinstance(self.role, MessageRole):
            raise ValueError("Message role must be a valid MessageRole enum")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "content": self.content,
            "role": self.role.value,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary representation."""
        return cls(
            content=data["content"],
            role=MessageRole(data["role"]),
            metadata=data.get("metadata"),
        )

    def to_langchain_format(self) -> Dict[str, str]:
        """Convert to LangChain message format."""
        return {
            "content": self.content,
            "role": self.role.value,
        }