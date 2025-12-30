"""
MinIO Storage Backend

MinIO-based storage backend with checksum-driven deduplication
for raw artifact persistence during document ingestion.
"""

import hashlib
import os
from typing import Optional
from urllib.parse import urljoin

from minio import Minio
from minio.error import S3Error

from app.core.entity.document import Document
from app.core.entity.storage_artifact import StorageArtifact

class MinIOStorage:
    """
    MinIO storage backend with checksum-based deduplication.

    Persists raw document artifacts and normalized outputs to MinIO object storage.
    Uses SHA-256 checksums to prevent duplicate writes and enable idempotent ingestion.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str = "botvisor-artifacts",
        secure: bool = True
    ):
        """
        Initialize MinIO storage backend.

        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name for artifact storage
            secure: Use HTTPS if True
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.secure = secure

        # Initialize MinIO client
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def save(self, content: bytes, document: Document) -> StorageArtifact:
        """
        Save content to MinIO storage with checksum-based deduplication.

        Args:
            content: Raw content bytes to store
            document: Document metadata

        Returns:
            StorageArtifact with URI and verification metadata
        """
        checksum = document.checksum
        object_name = self._get_object_name(document, checksum)

        # Check if content already exists
        if self.exists(checksum):
            # Generate presigned URL for existing object
            presigned_url = self._generate_presigned_url(object_name)
            return StorageArtifact(
                uri=presigned_url,
                checksum=checksum,
                size_bytes=document.size_bytes,
                already_existed=True
            )

        # Upload content to MinIO
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=content,
            length=len(content),
            content_type=document.mime_type
        )

        # Generate presigned URL for the uploaded object
        presigned_url = self._generate_presigned_url(object_name)

        return StorageArtifact(
            uri=presigned_url,
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
        # List objects with prefix matching the checksum pattern
        object_prefix = f"{checksum[:2]}/{checksum[2:4]}/{checksum}_"

        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=object_prefix,
                recursive=True
            )

            # If we find any objects with this prefix, content exists
            return any(True for _ in objects)
        except S3Error:
            return False

    def _get_object_name(self, document: Document, checksum: str) -> str:
        """
        Generate object name for document content in MinIO.

        Uses a hierarchical structure: checksum_prefix/document_id.ext

        Args:
            document: Document metadata
            checksum: Content checksum

        Returns:
            Object name for MinIO storage
        """
        # Create hierarchical path based on checksum
        checksum_prefix = f"{checksum[:2]}/{checksum[2:4]}"
        filename = f"{checksum}_{document.id}"

        # Add appropriate file extension based on mime type
        extension = self._get_extension_from_mime(document.mime_type)
        if extension:
            filename += extension

        return f"{checksum_prefix}/{filename}"

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

    def _generate_presigned_url(self, object_name: str, expires: int = 3600) -> str:
        """
        Generate presigned URL for accessing stored object.

        Args:
            object_name: Object name in MinIO
            expires: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        try:
            presigned_url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=expires
            )
            return presigned_url
        except S3Error as e:
            # Fallback to direct object URL if presigned fails
            return f"http{'s' if self.secure else ''}://{self.endpoint}/{self.bucket_name}/{object_name}"

    def get_object_url(self, checksum: str, document_id: str) -> Optional[str]:
        """
        Get URL to a stored object by checksum and document ID.

        Args:
            checksum: Content checksum
            document_id: Document ID

        Returns:
            URL to object if found, None otherwise
        """
        object_prefix = f"{checksum[:2]}/{checksum[2:4]}/{checksum}_{document_id}"

        try:
            objects = list(self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=object_prefix,
                recursive=True
            ))

            if objects:
                return self._generate_presigned_url(objects[0].object_name)
            return None
        except S3Error:
            return None
