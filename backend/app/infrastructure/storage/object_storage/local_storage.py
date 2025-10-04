"""Local filesystem-backed object storage adapter.

This adapter simulates S3/MinIO buckets using directories on disk.
Objects are stored as files, with a sidecar JSON for metadata/tags.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class PutObjectResult:
    bucket: str
    key: str
    size: int
    content_type: Optional[str]
    metadata_path: Path
    object_path: Path


class LocalObjectStorage:
    """Filesystem-based object storage for development/testing."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir or (Path.cwd() / "storage"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- bucket operations ---------------------------
    def ensure_bucket(self, bucket: str) -> Path:
        path = self.base_dir / bucket
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_buckets(self) -> List[str]:
        return [p.name for p in self.base_dir.iterdir() if p.is_dir()]

    def bucket_exists(self, bucket: str) -> bool:
        return (self.base_dir / bucket).is_dir()

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
        bucket_dir = self.ensure_bucket(bucket)
        object_path = bucket_dir / key
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(data)

        meta = {
            "content_type": content_type,
            "metadata": metadata or {},
            "tags": tags or {},
            "size": len(data),
        }
        meta_path = object_path.with_suffix(object_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return PutObjectResult(
            bucket=bucket,
            key=key,
            size=len(data),
            content_type=content_type,
            metadata_path=meta_path,
            object_path=object_path,
        )

    def get_object(self, bucket: str, key: str) -> Tuple[bytes, Dict[str, str]]:
        object_path = self.base_dir / bucket / key
        data = object_path.read_bytes()
        meta_path = object_path.with_suffix(object_path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return data, meta

    def list_objects(self, bucket: str, prefix: Optional[str] = None) -> List[str]:
        bucket_dir = self.base_dir / bucket
        if not bucket_dir.exists():
            return []
        keys: List[str] = []
        for path in bucket_dir.rglob("*"):
            if path.is_file() and not path.name.endswith(".meta.json"):
                relative = path.relative_to(bucket_dir).as_posix()
                if prefix is None or relative.startswith(prefix):
                    keys.append(relative)
        return keys

    def object_exists(self, bucket: str, key: str) -> bool:
        return (self.base_dir / bucket / key).is_file()

