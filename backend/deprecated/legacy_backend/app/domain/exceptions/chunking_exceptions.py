"""Chunking-specific exception hierarchy (Task 2.1.1)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class ChunkingError(Exception):
    """Base class for chunk generation errors."""

    def __init__(self, message: str, *, document_id: Optional[str] = None, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.document_id = document_id
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "document_id": self.document_id,
            "detail": self.detail,
        }


class ChunkingDependencyError(ChunkingError):
    """Raised when required chunking dependencies are missing."""


class ChunkGenerationError(ChunkingError):
    """Raised when chunk generation fails for a document."""


class ChunkValidationError(ChunkingError):
    """Raised when chunk metadata fails validation rules."""


class ChunkPersistenceError(ChunkingError):
    """Raised when persisting chunk outputs fails."""

