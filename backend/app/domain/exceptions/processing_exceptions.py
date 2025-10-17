"""Exception types for pipeline processing stages."""
from __future__ import annotations

from typing import Any, Dict, Optional


class ProcessingError(Exception):
    """Base class for errors occurring during document processing."""

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        source: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.source = source
        self.detail = detail

    def to_record(self) -> Dict[str, Any]:
        """Return a serializable representation of the error."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "stage": self.stage,
            "source": self.source,
            "detail": self.detail,
        }


class PipelineConfigurationError(ProcessingError):
    """Raised when pipeline configuration is invalid or incomplete."""


class ContentExtractionError(ProcessingError):
    """Raised when a document's raw content cannot be extracted."""


class NormalizationError(ProcessingError):
    """Raised when normalization or cleaning fails."""


class MetadataEnrichmentError(ProcessingError):
    """Raised when metadata extraction or enrichment fails."""


class QualityValidationError(ProcessingError):
    """Raised when quality validation encounters an error."""


class StorageWriteError(ProcessingError):
    """Raised when the processed payload cannot be written to storage."""
