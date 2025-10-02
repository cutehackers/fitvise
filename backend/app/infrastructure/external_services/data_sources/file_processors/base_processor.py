"""Base processor interfaces and result schemas for document pipeline (Task 1.3).

These definitions provide a stable contract for infrastructure processors like
Docling (PDF) and spaCy (text cleaning/NER), used by higher-level use cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass
class FileProcessingResult:
    """Standardized output for file processing steps."""

    success: bool
    text: str = ""
    markdown: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: Optional[str] = None
    language: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "markdown": self.markdown,
            "tables": self.tables,
            "metadata": self.metadata,
            "content_type": self.content_type,
            "language": self.language,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class CleanTextOptions:
    lowercase: bool = False
    correct_typos: bool = False
    lemmatize: bool = True
    extract_entities: bool = True


@dataclass
class CleanTextResult:
    success: bool
    original: str
    cleaned: str
    tokens: List[str]
    lemmas: List[str]
    entities: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "original": self.original,
            "cleaned": self.cleaned,
            "tokens": self.tokens,
            "lemmas": self.lemmas,
            "entities": self.entities,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class MetadataExtractionResult:
    success: bool
    keywords: List[str]
    entities: List[Dict[str, str]]
    dates: List[str]
    authors: List[str]
    language: Optional[str]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "keywords": self.keywords,
            "entities": self.entities,
            "dates": self.dates,
            "authors": self.authors,
            "language": self.language,
            "warnings": self.warnings,
            "error": self.error,
        }


class PdfProcessorBase(Protocol):
    """PDF processing contract."""

    SUPPORTED_SUFFIXES: Tuple[str, ...]

    def process_pdf_from_path(self, path: Path) -> FileProcessingResult:
        ...

    def process_pdf_from_bytes(self, data: bytes, file_name: str = "document.pdf") -> FileProcessingResult:
        ...


class TextCleanerBase(Protocol):
    """Text cleaning/NER contract."""

    def clean_text(self, text: str, options: Optional[CleanTextOptions] = None) -> CleanTextResult:
        ...


class MetadataExtractorBase(Protocol):
    """Metadata extraction contract."""

    def extract(self, texts: Sequence[str], top_k_keywords: int = 10) -> List[MetadataExtractionResult]:
        ...

