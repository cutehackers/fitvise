"""Storage adapters for object storage (Task 1.4.1).

Provides S3-compatible client wrappers, including a LocalObjectStorage
fallback that mimics bucket/object behavior on the filesystem.
"""

from .object_storage.minio_client import ObjectStorageClient
from .object_storage.local_storage import LocalObjectStorage

__all__ = [
    "ObjectStorageClient",
    "LocalObjectStorage",
]
