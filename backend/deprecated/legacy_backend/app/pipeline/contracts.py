from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DocumentSource:
    source_id: Optional[str]
    uri: Optional[str] = None
    path: Optional[str] = None
    content_bytes: Optional[bytes] = None
    content_type: Optional[str] = None  # e.g. "application/pdf", "text/markdown"
    origin: str = "file"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # do not expose raw bytes in dict view
        if d.get("content_bytes") is not None:
            d["content_bytes"] = f"<bytes:{len(self.content_bytes or b'')}>"
        return d


@dataclass
class SourceContent:
    text: str
    markdown: Optional[str]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    content_type: Optional[str]
    language: Optional[str]
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StandardSourceContent:
    normalized: str
    tokens: List[str]
    lemmas: List[str]
    entities: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetadataAnnotation:
    keywords: List[str]
    entities: List[Dict[str, Any]]
    dates: List[str]
    authors: List[str]
    language: Optional[str]
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    overall_score: float
    quality_level: str
    metrics: Dict[str, Any]
    validations: List[Dict[str, Any]]
    ge_report: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StorageObject:
    bucket: str
    key: str
    size: int
    metadata: Dict[str, str]
    tags: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    processed: int
    skipped: int
    failed: int
    stored: List[StorageObject]
    errors: List[Dict[str, Any]]
    counters: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "stored": [s.as_dict() for s in self.stored],
            "errors": self.errors,
            "counters": self.counters,
        }
