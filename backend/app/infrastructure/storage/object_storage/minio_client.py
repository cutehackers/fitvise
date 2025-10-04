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

    # ---------------------------- object operations --------------------------
    def put_object(
        self,
        bucket: str,
        key: str,
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
                bucket,
                key,
                BytesIO(data),
                length=length,
                content_type=content_type,
                metadata=metadata or {},
            )
            # Apply tags if available in client
            try:
                if tags:
                    self._minio.set_object_tags(bucket, key, tags)
            except Exception:
                pass
            # For consistent return type, also write a local metadata shadow only if local configured
            # Here we just return a lightweight result
            return PutObjectResult(
                bucket=bucket,
                key=key,
                size=length,
                content_type=content_type,
                metadata_path=Path("/dev/null"),
                object_path=Path(f"s3://{bucket}/{key}"),
            )

        # Local fallback
        return self._local.put_object(  # type: ignore[union-attr]
            bucket,
            key,
            data,
            content_type=content_type,
            metadata=metadata,
            tags=tags,
        )

