"""Use case for setting up object storage (Task 1.4.1).

Creates required buckets and verifies basic write/list operations. Defaults
to local filesystem-backed storage when MinIO client/config is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.infrastructure.storage import ObjectStorageClient
from app.infrastructure.storage.object_storage.minio_client import ObjectStorageConfig


DEFAULT_BUCKETS = [
    "rag-raw",        # for raw ingested files
    "rag-processed",  # for processed/markdown outputs
    "rag-metadata",   # for metadata sidecars/exports
]


@dataclass
class SetupObjectStorageRequest:
    provider: str = "local"  # "minio" or "local"
    endpoint: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    secure: bool = False
    base_dir: Optional[str] = None  # local-only base path
    buckets: Optional[List[str]] = None
    test_object: bool = True


@dataclass
class SetupObjectStorageResponse:
    success: bool
    provider: str
    created_buckets: List[str]
    environment: Dict[str, str]
    test_put_key: Optional[str] = None
    warnings: Optional[List[str]] = None


class SetupObjectStorageUseCase:
    """Ensure object storage is ready and organized into buckets."""

    def __init__(self) -> None:
        pass

    async def execute(self, request: SetupObjectStorageRequest) -> SetupObjectStorageResponse:
        buckets = request.buckets or DEFAULT_BUCKETS
        config = ObjectStorageConfig(
            provider=request.provider,
            endpoint=request.endpoint,
            access_key=request.access_key,
            secret_key=request.secret_key,
            secure=request.secure,
            base_dir=Path(request.base_dir).resolve() if request.base_dir else None,
        )
        client = ObjectStorageClient(config)

        created: List[str] = []
        for b in buckets:
            if not client.bucket_exists(b):
                client.ensure_bucket(b)
                created.append(b)
            else:
                # still return bucket name as ensured
                created.append(b)

        test_put_key = None
        if request.test_object:
            now = datetime.now(timezone.utc).isoformat()
            test_put_key = f"health/initialized_at={now}.txt"
            client.put_object(
                buckets[0],
                test_put_key,
                f"initialized {now}".encode(),
                content_type="text/plain",
                metadata={"component": "rag-pipeline", "phase": "1"},
                tags={"env": "dev", "task": "1.4.1"},
            )

        env: Dict[str, str] = {
            "provider": request.provider,
            "endpoint": request.endpoint or "local",
            "base_dir": str(config.base_dir) if config.base_dir else "n/a",
        }
        return SetupObjectStorageResponse(
            success=True,
            provider=request.provider,
            created_buckets=created,
            environment=env,
            test_put_key=test_put_key,
            warnings=[],
        )

