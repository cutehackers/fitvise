"""
Domain-level exception hierarchy for the RAG ingestion pipeline.

These exceptions are referenced across the pipeline orchestration layer
to provide structured error reporting for document discovery, source
integration, and processing stages.
"""
from __future__ import annotations

from .document_exceptions import (
    DocumentError,
    DocumentPathNotFoundError,
    DocumentFormatNotSupportedError,
    DocumentReadError,
    DocumentValidationError,
)
from .source_exceptions import (
    SourceError,
    SourceConfigurationError,
    SourceAuthenticationError,
    SourceNotReachableError,
    SourceRateLimitError,
)
from .processing_exceptions import (
    ProcessingError,
    ContentExtractionError,
    NormalizationError,
    MetadataEnrichmentError,
    QualityValidationError,
    StorageWriteError,
    PipelineConfigurationError,
)
from .chunking_exceptions import (
    ChunkingError,
    ChunkingDependencyError,
    ChunkGenerationError,
    ChunkValidationError,
    ChunkPersistenceError,
)

__all__ = [
    # Document
    "DocumentError",
    "DocumentPathNotFoundError",
    "DocumentFormatNotSupportedError",
    "DocumentReadError",
    "DocumentValidationError",
    # Source
    "SourceError",
    "SourceConfigurationError",
    "SourceAuthenticationError",
    "SourceNotReachableError",
    "SourceRateLimitError",
    # Processing
    "ProcessingError",
    "ContentExtractionError",
    "NormalizationError",
    "MetadataEnrichmentError",
    "QualityValidationError",
    "StorageWriteError",
    "PipelineConfigurationError",
    # Chunking
    "ChunkingError",
    "ChunkingDependencyError",
    "ChunkGenerationError",
    "ChunkValidationError",
    "ChunkPersistenceError",
]
