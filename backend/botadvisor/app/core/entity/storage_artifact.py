"""
Storage Artifact Entity

Result of storing content via storage backend.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class StorageArtifact:
    """
    Result of storing content via storage backend.

    Provides the URI where content was stored and verification
    metadata for deduplication checks.
    """

    uri: str
    checksum: str
    size_bytes: int
    already_existed: bool = False
