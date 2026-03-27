"""Canonical storage backend selection for the BotAdvisor runtime."""

from __future__ import annotations

from typing import Any

from botadvisor.app.core.config import get_settings
from botadvisor.app.storage.local_storage import LocalStorage
from botadvisor.app.storage.minio_client import MinIOStorage


def create_storage_backend(settings: Any) -> LocalStorage | MinIOStorage:
    """Create the configured artifact storage backend."""
    backend = getattr(settings, "storage_backend", "local")

    if backend == "local":
        return LocalStorage(base_path=settings.storage_local_path)

    if backend == "minio":
        return MinIOStorage(
            endpoint=settings.storage_minio_endpoint,
            access_key=settings.storage_minio_access_key,
            secret_key=settings.storage_minio_secret_key,
            bucket_name=settings.storage_minio_bucket,
            secure=settings.storage_minio_secure,
        )

    raise ValueError(f"Unsupported storage backend: {backend}")


def get_storage_backend(settings: Any | None = None) -> LocalStorage | MinIOStorage:
    """Return the runtime storage backend using explicit or global settings."""
    return create_storage_backend(settings or get_settings())
