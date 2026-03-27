"""Exception types for data source configuration and connectivity."""
from __future__ import annotations

from typing import Any, Dict, Optional


class SourceError(Exception):
    """Base class for data source related errors."""

    def __init__(
        self,
        message: str,
        *,
        source: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.source = source
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "source": self.source,
            "detail": self.detail,
        }


class SourceConfigurationError(SourceError):
    """Raised when a data source configuration is invalid or incomplete."""


class SourceAuthenticationError(SourceError):
    """Raised when authentication against a data source fails."""


class SourceNotReachableError(SourceError):
    """Raised when the pipeline cannot connect to the data source."""


class SourceRateLimitError(SourceError):
    """Raised when requests to a source exceed allowed rate limits."""
