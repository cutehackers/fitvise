"""MinIO object storage client wrapper with local fallback (Task 1.4.1).

This module provides a thin wrapper that tries to use the `minio` client
when available. If missing or configuration is incomplete, it falls back to
`LocalObjectStorage` so the pipeline remains usable in development.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from pathlib import Path

try:  # Optional dependency
    from minio import Minio  # type: ignore
    from minio.error import S3Error  # type: ignore
except Exception:  # pragma: no cover - keep runtime resilient
    Minio = None  # type: ignore
    S3Error = Exception  # type: ignore

from .local_storage import LocalObjectStorage, PutObjectResult


@dataclass
class ObjectStorageConfig:
    provider: str = "local"  # "minio" or "local"
    endpoint: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    secure: bool = False
    base_dir: Optional[Path] = None  # local-only


class ObjectStorageClient:
    """Unified object storage client that can back onto MinIO or local FS."""

    def __init__(self, config: Optional[ObjectStorageConfig] = None) -> None:
        self.config = config or ObjectStorageConfig()
        self._minio = None
        self._local = None

        if self.config.provider == "minio" and Minio is not None and self.config.endpoint:
            try:
                self._minio = Minio(
                    self.config.endpoint,
                    access_key=self.config.access_key or "minioadmin",
                    secret_key=self.config.secret_key or "minioadmin",
                    secure=self.config.secure,
                )
            except Exception:  # fallback to local if init fails
                self._minio = None

        if self._minio is None:
            self._local = LocalObjectStorage(base_dir=self.config.base_dir)

    # --------------------------- bucket operations ---------------------------
    def ensure_bucket(self, bucket: str) -> None:
        if self._minio is not None:
            found = self._minio.bucket_exists(bucket)
            if not found:
                self._minio.make_bucket(bucket)
        else:
            self._local.ensure_bucket(bucket)  # type: ignore[union-attr]

    def bucket_exists(self, bucket: str) -> bool:
        if self._minio is not None:
            return bool(self._minio.bucket_exists(bucket))
        return self._local.bucket_exists(bucket)  # type: ignore[union-attr]

    async def list_buckets(self):
        """List all buckets."""
        if self._minio is not None:
            buckets = self._minio.list_buckets()
            return buckets
        else:
            # For local storage, return buckets as directories in base_dir
            if self._local and hasattr(self._local, 'base_dir') and self._local.base_dir:
                base_path = Path(self._local.base_dir)
                if base_path.exists():
                    # Create a simple bucket-like object for directories
                    class LocalBucket:
                        def __init__(self, name):
                            self.name = name

                    return [LocalBucket(d.name) for d in base_path.iterdir() if d.is_dir()]
            return []

    async def create_bucket(self, bucket_name: str) -> None:
        """Create a new bucket."""
        if self._minio is not None:
            if not self._minio.bucket_exists(bucket_name):
                self._minio.make_bucket(bucket_name)
        else:
            # For local storage, create directory
            if self._local and hasattr(self._local, 'base_dir') and self._local.base_dir:
                bucket_path = Path(self._local.base_dir) / bucket_name
                bucket_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------- object operations --------------------------
    async def put_object(
        self,
        bucket_name: str,
        object_key: str,
        data: bytes,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> PutObjectResult:
        if self._minio is not None:
            # Upload object
            from io import BytesIO

            length = len(data)
            self._minio.put_object(
                bucket_name,
                object_key,
                BytesIO(data),
                length=length,
                content_type=content_type,
                metadata=metadata or {},
            )
            # Apply tags if available in client
            try:
                if tags:
                    self._minio.set_object_tags(bucket_name, object_key, tags)
            except Exception:
                pass
            # For consistent return type, also write a local metadata shadow only if local configured
            # Here we just return a lightweight result
            return PutObjectResult(
                bucket=bucket_name,
                key=object_key,
                size=length,
                content_type=content_type,
                metadata_path=Path("/dev/null"),
                object_path=Path(f"s3://{bucket_name}/{object_key}"),
            )

        # Local fallback
        return self._local.put_object(  # type: ignore[union-attr]
            bucket_name,
            object_key,
            data,
            content_type=content_type,
            metadata=metadata,
            tags=tags,
        )

    async def get_object(self, bucket: str, key: str):
        """Get an object from storage."""
        if self._minio is not None:
            from io import BytesIO

            response = self._minio.get_object(bucket, key)
            data = response.read()
            response.close()
            response.release_conn()

            # Create a simple result object with data attribute
            class GetObjectResult:
                def __init__(self, data: bytes):
                    self.data = data

            return GetObjectResult(data)
        else:
            # Local fallback
            if self._local and hasattr(self._local, 'get_object'):
                result = self._local.get_object(bucket, key)
                # Local storage returns a tuple (data, metadata)
                if isinstance(result, tuple) and len(result) >= 1:
                    data = result[0]
                else:
                    data = result

                class GetObjectResult:
                    def __init__(self, data: bytes):
                        self.data = data

                return GetObjectResult(data)
            else:
                # Simple local file read
                if self._local and hasattr(self._local, 'base_dir') and self._local.base_dir:
                    file_path = Path(self._local.base_dir) / bucket / key
                    if file_path.exists():
                        class GetObjectResult:
                            def __init__(self, data: bytes):
                                self.data = data

                        with open(file_path, 'rb') as f:
                            return GetObjectResult(f.read())
                raise FileNotFoundError(f"Object {key} not found in bucket {bucket}")

    async def delete_object(self, bucket: str, key: str) -> None:
        """Delete an object from storage."""
        if self._minio is not None:
            self._minio.remove_object(bucket, key)
        else:
            # Local fallback
            if self._local and hasattr(self._local, 'delete_object'):
                self._local.delete_object(bucket, key)
            else:
                # Simple local file deletion
                if self._local and hasattr(self._local, 'base_dir') and self._local.base_dir:
                    file_path = Path(self._local.base_dir) / bucket / key
                    if file_path.exists():
                        file_path.unlink()

