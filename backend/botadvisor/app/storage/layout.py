"""Shared artifact naming and layout helpers for storage backends."""

from __future__ import annotations


MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/html": ".html",
    "application/json": ".json",
    "application/xml": ".xml",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "application/zip": ".zip",
}


def build_checksum_prefix(checksum: str) -> str:
    """Build the shared two-level checksum prefix used by storage backends."""
    return f"{checksum[:2]}/{checksum[2:4]}"


def extension_for_mime_type(mime_type: str) -> str:
    """Return the file extension for the provided MIME type."""
    return MIME_TO_EXTENSION.get(mime_type, "")


def build_artifact_name(*, checksum: str, document_id: str, mime_type: str) -> str:
    """Build the storage artifact name used by local and object storage."""
    extension = extension_for_mime_type(mime_type)
    artifact_name = f"{checksum}_{document_id}"
    if extension:
        artifact_name += extension
    return artifact_name
