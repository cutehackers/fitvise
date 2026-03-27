"""Use cases for advanced document processing (Task 1.3.*).

This package provides:
- Task 1.3.1: PDF processing to structured markdown with table extraction
- Task 1.3.2: spaCy-based text cleaning pipeline with fallbacks
- Task 1.3.3: Metadata extraction (keywords, entities, dates, authors)
- Task 1.3.4: Data quality validation with Great Expectations fallback

All external dependencies are optional. When unavailable at runtime, each
use case provides safe, minimal fallbacks so the system remains usable.
"""

from .process_pdfs import (
    ProcessPdfsRequest,
    ProcessPdfsResponse,
    PdfDocumentResult,
    ProcessPdfsUseCase,
)
from .normalize_text import (
    NormalizeTextRequest,
    NormalizeTextResponse,
    NormalizeTextUseCase,
)
from .extract_metadata import (
    ExtractMetadataRequest,
    ExtractMetadataResponse,
    ExtractMetadataUseCase,
)
from .validate_quality import (
    ValidateQualityRequest,
    ValidateQualityResponse,
    ValidateQualityUseCase,
)

__all__ = [
    "ProcessPdfsRequest",
    "ProcessPdfsResponse",
    "PdfDocumentResult",
    "ProcessPdfsUseCase",
    "NormalizeTextRequest",
    "NormalizeTextResponse",
    "NormalizeTextUseCase",
    "ExtractMetadataRequest",
    "ExtractMetadataResponse",
    "ExtractMetadataUseCase",
    "ValidateQualityRequest",
    "ValidateQualityResponse",
    "ValidateQualityUseCase",
]
