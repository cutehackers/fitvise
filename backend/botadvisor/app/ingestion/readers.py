"""Reader selection helpers for canonical ingestion workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DOCLING_AVAILABLE = False
PDFReader = None
OfficeReader = None
TextReader = None

try:
    from docling.readers import PDFReader, OfficeReader, TextReader

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


def simple_text_reader(content: bytes) -> str:
    """Decode text content using pragmatic fallbacks."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return content.decode("latin-1")
        except Exception:
            return content.decode("utf-8", errors="replace")


def get_reader_for_file(file_path: Path, logger: Any) -> Any | None:
    """Return the configured reader for the given file."""
    extension = file_path.suffix.lower()

    if not DOCLING_AVAILABLE:
        if extension in (".txt", ".md", ".html", ".htm"):
            return simple_text_reader
        logger.warning("Docling not available, only text files supported", extra={"file": str(file_path)})
        return None

    if extension == ".pdf":
        return PDFReader()
    if extension in (".doc", ".docx"):
        return OfficeReader()
    if extension in (".txt", ".md", ".html", ".htm"):
        return TextReader()

    logger.warning("Unsupported file type", extra={"file": str(file_path), "extension": extension})
    return None


def read_text_with_reader(reader: Any, content: bytes) -> str:
    """Read text content with either Docling readers or fallback callables."""
    if hasattr(reader, "read"):
        return reader.read(content)
    return reader(content)
