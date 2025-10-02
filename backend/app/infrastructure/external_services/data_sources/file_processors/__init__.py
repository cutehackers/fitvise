"""File processing services for data ingestion."""
from .tika_processor import (
    TikaClientConfig,
    TikaExtractionResult,
    TikaIntegrationService,
)
from .base_processor import (
    FileProcessingResult,
    CleanTextOptions,
    CleanTextResult,
    MetadataExtractionResult,
    PdfProcessorBase,
    TextCleanerBase,
    MetadataExtractorBase,
)
from .docling_processor import DoclingPdfProcessor
from .spacy_processor import SpacyTextProcessor

__all__ = [
    "TikaClientConfig",
    "TikaExtractionResult",
    "TikaIntegrationService",
    "FileProcessingResult",
    "CleanTextOptions",
    "CleanTextResult",
    "MetadataExtractionResult",
    "PdfProcessorBase",
    "TextCleanerBase",
    "MetadataExtractorBase",
    "DoclingPdfProcessor",
    "SpacyTextProcessor",
]
