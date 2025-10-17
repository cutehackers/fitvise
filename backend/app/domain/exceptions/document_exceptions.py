"""Exception types for document discovery and validation steps."""
from __future__ import annotations

from typing import Any, Dict, Optional


class DocumentError(Exception):
    """Base class for document-related errors in the pipeline."""

    def __init__(
        self,
        message: str,
        *,
        document: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.document = document
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the error for logging or API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "document": self.document,
            "detail": self.detail,
        }


class DocumentPathNotFoundError(DocumentError):
    """Raised when a configured document path cannot be located."""


class DocumentFormatNotSupportedError(DocumentError):
    """Raised when a document format is not supported by the processors."""


class DocumentReadError(DocumentError):
    """Raised when the system cannot read the document from disk or memory."""


class DocumentValidationError(DocumentError):
    """Raised when a document fails validation prior to processing."""
