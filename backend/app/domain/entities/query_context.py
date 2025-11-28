"""Query context domain entity.

This module contains the QueryContext entity that encapsulates
the context and metadata for retrieval queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from app.domain.exceptions.retrieval_exceptions import QueryValidationError


class QueryPriority(Enum):
    """Priority levels for retrieval queries."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_string(cls, priority_str: str) -> QueryPriority:
        """Create priority from string value."""
        priority_map = {
            "low": cls.LOW,
            "normal": cls.NORMAL,
            "high": cls.HIGH,
            "critical": cls.CRITICAL,
        }
        return priority_map.get(priority_str.lower(), cls.NORMAL)


@dataclass
class QueryContext:
    """Context entity for retrieval queries.

    This entity contains all contextual information about a retrieval query
    including user session, request metadata, and execution preferences.
    """

    query_id: UUID = field(default_factory=uuid4)
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    query_text: str = ""
    priority: QueryPriority = QueryPriority.NORMAL
    request_source: str = "api"
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_context: Dict[str, Any] = field(default_factory=dict)
    execution_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_override: Optional[int] = None
    max_tokens_override: Optional[int] = None
    is_cached: bool = False

    def __post_init__(self):
        """Validate context after initialization."""
        if self.query_text and not isinstance(self.query_text, str):
            raise QueryValidationError("Query text must be a string")

        if self.query_text and len(self.query_text.strip()) == 0:
            raise QueryValidationError("Query text cannot be empty")

        self._validate_request_metadata()
        self._validate_execution_preferences()

    def _validate_request_metadata(self) -> None:
        """Validate request metadata structure."""
        # Check for required metadata fields
        if self.request_source not in ["api", "cli", "batch", "internal"]:
            raise QueryValidationError(
                f"Invalid request source: {self.request_source}"
            )

        # Validate IP address if present
        if "client_ip" in self.request_metadata:
            ip = self.request_metadata["client_ip"]
            if not isinstance(ip, str) or not ip:
                raise QueryValidationError("Client IP must be a non-empty string")

    def _validate_execution_preferences(self) -> None:
        """Validate execution preferences."""
        # Validate timeout override
        if self.timeout_override is not None:
            if not isinstance(self.timeout_override, int) or self.timeout_override <= 0:
                raise QueryValidationError(
                    f"Timeout override must be a positive integer, got {self.timeout_override}"
                )

            if self.timeout_override > 600:  # 10 minutes max
                raise QueryValidationError(
                    f"Timeout override too large: {self.timeout_override} > 600 seconds"
                )

        # Validate max tokens override
        if self.max_tokens_override is not None:
            if not isinstance(self.max_tokens_override, int) or self.max_tokens_override <= 0:
                raise QueryValidationError(
                    f"Max tokens override must be a positive integer, got {self.max_tokens_override}"
                )

            if self.max_tokens_override > 32000:  # Reasonable limit
                raise QueryValidationError(
                    f"Max tokens override too large: {self.max_tokens_override} > 32000"
                )

    @property
    def is_high_priority(self) -> bool:
        """Check if this is a high-priority query."""
        return self.priority in {QueryPriority.HIGH, QueryPriority.CRITICAL}

    @property
    def is_critical_priority(self) -> bool:
        """Check if this is a critical-priority query."""
        return self.priority == QueryPriority.CRITICAL

    @property
    def has_user_context(self) -> bool:
        """Check if this context has user information."""
        return self.user_id is not None

    @property
    def has_session_context(self) -> bool:
        """Check if this context has session information."""
        return self.session_id is not None

    @property
    def query_length(self) -> int:
        """Get the length of the query text."""
        return len(self.query_text)

    @property
    def is_complex_query(self) -> bool:
        """Determine if this is a complex query based on length and content."""
        # Simple heuristic for query complexity
        return (
            self.query_length > 100 or
            self.query_text.count('?') > 1 or
            'and' in self.query_text.lower() or
            'or' in self.query_text.lower() or
            'not' in self.query_text.lower()
        )

    def get_client_ip(self) -> Optional[str]:
        """Get the client IP address from request metadata."""
        return self.request_metadata.get("client_ip")

    def get_user_agent(self) -> Optional[str]:
        """Get the user agent from request metadata."""
        return self.request_metadata.get("user_agent")

    def get_request_id(self) -> Optional[str]:
        """Get the request ID from request metadata."""
        return self.request_metadata.get("request_id")

    def get_correlation_id(self) -> Optional[str]:
        """Get the correlation ID from request metadata."""
        return self.request_metadata.get("correlation_id")

    def set_user_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        self.user_preferences[key] = value

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.user_preferences.get(key, default)

    def set_session_context(self, key: str, value: Any) -> None:
        """Set session context information."""
        self.session_context[key] = value

    def get_session_context(self, key: str, default: Any = None) -> Any:
        """Get session context information."""
        return self.session_context.get(key, default)

    def get_effective_timeout(self, default_timeout: int) -> int:
        """Get the effective timeout for this query."""
        if self.timeout_override is not None:
            return self.timeout_override

        # Apply priority-based timeout adjustments
        if self.is_critical_priority:
            return default_timeout * 2  # Double timeout for critical queries
        elif self.is_high_priority:
            return int(default_timeout * 1.5)  # 50% more timeout for high priority

        return default_timeout

    def get_effective_max_tokens(self, default_max_tokens: int) -> int:
        """Get the effective max tokens for this query."""
        if self.max_tokens_override is not None:
            return self.max_tokens_override

        # Apply user preferences for token limits
        user_max_tokens = self.get_user_preference("max_tokens")
        if user_max_tokens and isinstance(user_max_tokens, int):
            return min(user_max_tokens, default_max_tokens)

        return default_max_tokens

    def should_use_cache(self) -> bool:
        """Determine if caching should be used for this query."""
        # Don't cache critical queries to ensure freshness
        if self.is_critical_priority:
            return False

        # Check user preference
        cache_preference = self.get_user_preference("use_cache", True)
        if not cache_preference:
            return False

        # Don't cache very recent queries
        time_since_creation = datetime.utcnow() - self.created_at
        if time_since_creation.total_seconds() < 60:  # Less than 1 minute
            return False

        return True

    def get_cache_key(self) -> str:
        """Generate a cache key for this query context."""
        key_parts = [
            str(self.query_id),
            self.query_text.lower().strip(),
            str(self.priority.value),
        ]

        if self.has_user_context:
            key_parts.append(f"user_{self.user_id}")

        if self.has_session_context:
            key_parts.append(f"session_{self.session_id}")

        # Include relevant user preferences that affect results
        relevant_preferences = ["language", "region", "content_filter"]
        for pref in relevant_preferences:
            if pref in self.user_preferences:
                key_parts.append(f"{pref}_{self.user_preferences[pref]}")

        return "|".join(key_parts)

    def get_execution_tags(self) -> Dict[str, str]:
        """Get execution tags for monitoring and logging."""
        tags = {
            "priority": self.priority.name,
            "source": self.request_source,
            "has_user": str(self.has_user_context),
            "has_session": str(self.has_session_context),
            "complex_query": str(self.is_complex_query),
        }

        if self.has_user_context:
            tags["user_id"] = self.user_id

        if self.get_request_id():
            tags["request_id"] = self.get_request_id()

        return tags

    def should_apply_reranking(self) -> bool:
        """Determine if reranking should be applied based on context."""
        # Always apply reranking for high-priority queries
        if self.is_high_priority:
            return True

        # Check user preference
        return self.get_user_preference("enable_reranking", True)

    def should_apply_filters(self) -> bool:
        """Determine if content filters should be applied."""
        # Always apply filters for API requests
        if self.request_source == "api":
            return True

        # Check user preference
        return self.get_user_preference("apply_filters", True)

    def get_metadata_filters(self) -> Dict[str, Any]:
        """Get metadata filters based on context."""
        filters = {}

        # Add user-specific filters
        if self.has_user_context:
            user_filter_level = self.get_user_preference("filter_level", "standard")
            if user_filter_level != "none":
                filters["user_filter_level"] = user_filter_level

        # Add session-based filters
        if self.has_session_context:
            allowed_sources = self.get_session_context("allowed_sources")
            if allowed_sources:
                filters["source_types"] = allowed_sources

        # Add language filters
        language = self.get_user_preference("language")
        if language:
            filters["language"] = language

        # Add content type preferences
        content_types = self.get_user_preference("preferred_content_types")
        if content_types:
            filters["content_types"] = content_types

        return filters

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary representation."""
        return {
            "query_id": str(self.query_id),
            "user_id": self.user_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "query_text": self.query_text,
            "priority": self.priority.name,
            "request_source": self.request_source,
            "request_metadata": self.request_metadata,
            "user_preferences": self.user_preferences,
            "session_context": self.session_context,
            "execution_preferences": self.execution_preferences,
            "created_at": self.created_at.isoformat(),
            "timeout_override": self.timeout_override,
            "max_tokens_override": self.max_tokens_override,
            "is_cached": self.is_cached,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the context."""
        return {
            "query_id": str(self.query_id),
            "has_user": self.has_user_context,
            "has_session": self.has_session_context,
            "priority": self.priority.name,
            "query_length": self.query_length,
            "is_complex": self.is_complex_query,
            "request_source": self.request_source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryContext:
        """Create context from dictionary representation."""
        # Handle UUID conversion
        if "query_id" in data and isinstance(data["query_id"], str):
            data["query_id"] = UUID(data["query_id"])

        if "session_id" in data and data["session_id"] and isinstance(data["session_id"], str):
            data["session_id"] = UUID(data["session_id"])

        # Handle datetime conversion
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Handle priority conversion
        if "priority" in data and isinstance(data["priority"], str):
            data["priority"] = QueryPriority[data["priority"]]

        return cls(**data)