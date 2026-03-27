"""File metadata rules for canonical ingestion workflows."""

from __future__ import annotations

from pathlib import Path


MIME_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".xml": "application/xml",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
}


def detect_mime_type(file_path: Path) -> str:
    """Return the canonical MIME type for a filesystem document."""
    return MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")


def extract_source_metadata(file_path: Path, platform: str) -> dict[str, str]:
    """Extract metadata fields used to build canonical document entities."""
    return {
        "source_id": str(file_path),
        "platform": platform,
        "source_url": f"file://{file_path.absolute()}",
        "file_name": file_path.name,
        "file_extension": file_path.suffix,
        "file_size": str(file_path.stat().st_size),
    }
