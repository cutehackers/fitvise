"""
Local Storage Backend

Filesystem-based storage backend with checksum-driven deduplication
for raw artifact persistence during document ingestion.
"""

from pathlib import Path
from typing import Optional

from app.core.entity.document import Document
from app.core.entity.storage_artifact import StorageArtifact

class LocalStorage:
    """
    Local filesystem storage backend with checksum-based deduplication.

    Persists raw document artifacts and normalized outputs to a configurable
    local directory structure. Uses SHA-256 checksums to prevent duplicate
    writes and enable idempotent ingestion.
    """

    def __init__(self, base_path: str = "./data/artifacts"):
        """
        Initialize local storage backend.

        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, content: bytes, document: Document) -> StorageArtifact:
        """
        Save content to local storage with checksum-based deduplication.

        Args:
            content: Raw content bytes to store
            document: Document metadata

        Returns:
            StorageArtifact with URI and verification metadata
        """
        checksum = document.checksum
        storage_path = self._get_storage_path(document, checksum)

        # Check if content already exists
        if self.exists(checksum):
            return StorageArtifact(
                uri=str(storage_path),
                checksum=checksum,
                size_bytes=document.size_bytes,
                already_existed=True
            )

        # Ensure directory structure exists
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content to filesystem
        with storage_path.open("wb") as f:
            f.write(content)

        return StorageArtifact(
            uri=str(storage_path),
            checksum=checksum,
            size_bytes=document.size_bytes,
            already_existed=False
        )

    def exists(self, checksum: str) -> bool:
        """
        Check if content with given checksum already exists in storage.

        Args:
            checksum: SHA-256 checksum to check

        Returns:
            True if content exists, False otherwise
        """
        # Look for any file with this checksum in the storage
        checksum_dir = self.base_path / checksum[:2] / checksum[2:4]
        if not checksum_dir.exists():
            return False

        # Check if any file in the checksum directory matches
        for file_path in checksum_dir.iterdir():
            if file_path.name.startswith(checksum):
                return True

        return False

    def _get_storage_path(self, document: Document, checksum: str) -> Path:
        """
        Generate storage path for document content.

        Uses a hierarchical structure: base_path/checksum_prefix/document_id.ext

        Args:
            document: Document metadata
            checksum: Content checksum

        Returns:
            Full path where content should be stored
        """
        # Create hierarchical path based on checksum
        checksum_dir = self.base_path / checksum[:2] / checksum[2:4]
        filename = f"{checksum}_{document.id}"

        # Add appropriate file extension based on mime type
        extension = self._get_extension_from_mime(document.mime_type)
        if extension:
            filename += extension

        return checksum_dir / filename

    def _get_extension_from_mime(self, mime_type: str) -> str:
        """
        Get file extension from MIME type.

        Args:
            mime_type: MIME type string

        Returns:
            File extension (including dot) or empty string
        """
        mime_to_ext = {
            "application/pdf": ".pdf",
            "text/plain": ".txt",
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

        return mime_to_ext.get(mime_type, "")

    def get_artifact_path(self, checksum: str, document_id: str) -> Optional[Path]:
        """
        Get the path to a stored artifact by checksum and document ID.

        Args:
            checksum: Content checksum
            document_id: Document ID

        Returns:
            Path to artifact if found, None otherwise
        """
        checksum_dir = self.base_path / checksum[:2] / checksum[2:4]
        if not checksum_dir.exists():
            return None

        # Look for file matching the pattern
        pattern = f"{checksum}_{document_id}*"
        for file_path in checksum_dir.glob(pattern):
            return file_path

        return None
