from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
import json
from typing import Any, Dict

try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False


class DocumentOption(BaseModel):
    path: Path = Field(..., description="Input directory to scan")
    recurse: bool = Field(True, description="Recurse into subdirectories")
    include: List[str] = Field(default_factory=lambda: ["*.pdf"], description="Glob patterns to include")


class StorageOptions(BaseModel):
    provider: Literal["local", "minio"] = "local"
    base_dir: Optional[Path] = Field(None, description="Local base directory for buckets (local)")
    endpoint: Optional[str] = Field(None, description="MinIO endpoint host:port (minio)")
    bucket: str = Field("rag-processed", description="Target bucket name")
    metadata_base: Path = Field(Path("rag-metadata"), description="Local path for manifests/errors")


class ScheduleOptions(BaseModel):
    mode: Literal["manual", "cron", "airflow"] = "manual"
    cron: Optional[str] = Field("0 3 1 * *", description="Run monthly at 03:00 on day 1")


class ProcessorOptions(BaseModel):
    pdf: Literal["docling"] = "docling"
    non_pdf: Literal["tika", "none"] = "tika"
    normalize: Literal["spacy"] = "spacy"


class LimitOptions(BaseModel):
    since: Optional[date] = None
    max_files: Optional[int] = None


class DedupeOptions(BaseModel):
    strategy: Literal["content_hash", "none"] = "content_hash"
    manifest_path: Path = Field(Path("rag-metadata/runs"), description="Where to write per-run manifests")


class PipelineSpec(BaseModel):
    inputs: DocumentOption
    storage: StorageOptions = Field(default_factory=StorageOptions)
    schedule: ScheduleOptions = Field(default_factory=ScheduleOptions)
    processors: ProcessorOptions = Field(default_factory=ProcessorOptions)
    limits: LimitOptions = Field(default_factory=LimitOptions)
    dedupe: DedupeOptions = Field(default_factory=DedupeOptions)

    @model_validator(mode="after")
    def _validate_paths(self):  # type: ignore[override]
        if self.storage.provider == "local" and self.storage.base_dir is None:
            # Default local base dir to project-relative ./data
            self.storage.base_dir = Path("data").resolve()
        return self

    # --------------------------- helpers ---------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineSpec":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        text = p.read_text(encoding="utf-8")
        data: Dict[str, Any]
        if _YAML_AVAILABLE:
            try:
                data = yaml.safe_load(text) or {}
                if not isinstance(data, dict):
                    raise ValueError("Top-level YAML must be a mapping")
            except Exception:
                # fall back to json
                data = json.loads(text)
        else:
            data = json.loads(text)
        return cls.model_validate(data)
