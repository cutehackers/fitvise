from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_create_storage_backend_returns_local_storage_by_default():
    from botadvisor.app.storage.factory import create_storage_backend
    from botadvisor.app.storage.local_storage import LocalStorage

    settings = SimpleNamespace(
        storage_backend="local",
        storage_local_path="/tmp/botadvisor-artifacts",
        storage_minio_endpoint=None,
        storage_minio_access_key=None,
        storage_minio_secret_key=None,
        storage_minio_bucket="botadvisor-artifacts",
        storage_minio_secure=False,
    )

    backend = create_storage_backend(settings)

    assert isinstance(backend, LocalStorage)
    assert str(backend.base_path) == "/tmp/botadvisor-artifacts"


def test_create_storage_backend_returns_minio_storage_when_configured():
    from botadvisor.app.storage.factory import create_storage_backend

    settings = SimpleNamespace(
        storage_backend="minio",
        storage_local_path="/tmp/unused",
        storage_minio_endpoint="minio.internal:9000",
        storage_minio_access_key="access",
        storage_minio_secret_key="secret",
        storage_minio_bucket="botadvisor-artifacts",
        storage_minio_secure=True,
    )

    backend = create_storage_backend(settings)

    assert backend.endpoint == "minio.internal:9000"
    assert backend.bucket_name == "botadvisor-artifacts"
    assert backend.secure is True


def test_create_storage_backend_rejects_unknown_backend():
    from botadvisor.app.storage.factory import create_storage_backend

    settings = SimpleNamespace(
        storage_backend="unknown",
        storage_local_path="/tmp/botadvisor-artifacts",
        storage_minio_endpoint=None,
        storage_minio_access_key=None,
        storage_minio_secret_key=None,
        storage_minio_bucket="botadvisor-artifacts",
        storage_minio_secure=False,
    )

    with pytest.raises(ValueError, match="Unsupported storage backend"):
        create_storage_backend(settings)
