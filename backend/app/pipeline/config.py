from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
import json

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

    @model_validator(mode="after")
    def _expand_path(self):  # type: ignore[override]
        self.path = self.path.expanduser()
        return self

class StorageOptions(BaseModel):
    provider: Literal["local", "minio"] = "local"
    base_dir: Optional[Path] = Field(None, description="Local base directory for buckets (local)")
    endpoint: Optional[str] = Field(None, description="MinIO endpoint host:port (minio)")
    access_key: Optional[str] = Field(None, description="MinIO access key")
    secret_key: Optional[str] = Field(None, description="MinIO secret key")
    secure: bool = Field(False, description="Use HTTPS for MinIO connections")
    bucket: str = Field("rag-processed", description="Target bucket name")
    metadata_base: Path = Field(Path("rag-metadata"), description="Local path for manifests/errors")


class ScheduleOptions(BaseModel):
    mode: Literal["manual", "cron", "airflow"] = "manual"
    cron: Optional[str] = Field("0 3 1 * *", description="Run monthly at 03:00 on day 1")


class ProcessorOptions(BaseModel):
    pdf: Literal["docling"] = "docling"
    non_pdf: Literal["tika", "none"] = "tika"
    normalize: Literal["spacy"] = "spacy"


class ChunkingOptions(BaseModel):
    enabled: bool = True
    preset: Optional[str] = Field(None, description="Chunking preset name (balanced, short_form, long_form)")
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Override chunker config values")
    replace_existing_chunks: bool = True
    metadata_overrides: Dict[str, Any] = Field(default_factory=dict)


class LimitOptions(BaseModel):
    since: Optional[date] = None
    max_files: Optional[int] = None


class DedupeOptions(BaseModel):
    strategy: Literal["content_hash", "none"] = "content_hash"
    manifest_path: Path = Field(Path("rag-metadata/runs"), description="Where to write per-run manifests")


class AuditOptions(BaseModel):
    enabled: bool = False
    scan_paths: List[str] = Field(default_factory=list)
    database_configs: List[Dict[str, Any]] = Field(default_factory=list)
    api_endpoints: List[str] = Field(default_factory=list)
    max_scan_depth: int = 5
    min_file_count: int = 5
    export_csv: Optional[str] = None
    export_json: Optional[str] = None
    save_to_repository: bool = True


class CategorizeOptions(BaseModel):
    enabled: bool = False
    train_model: bool = False
    use_synthetic_data: bool = True
    synthetic_data_size: int = 100
    categorize_sources: bool = True
    source_ids: Optional[List[str]] = None
    min_confidence: float = 0.6
    model_type: str = "logistic_regression"
    save_model: bool = True
    model_path: Optional[str] = None


class DatabaseConnectorOptions(BaseModel):
    name: str
    connector_type: Literal["postgres", "postgresql", "mysql", "mariadb", "mongo", "mongodb"]
    driver: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    db_schema: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    use_ssl: bool = False
    sample_collection: Optional[str] = None
    sample_limit: int = Field(5, ge=1, le=100)
    fetch_samples: bool = True


class WebSourceOptions(BaseModel):
    start_urls: List[str]
    allowed_domains: Optional[List[str]] = None
    max_depth: int = 2
    max_pages: int = 50
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    follow_css_selectors: Optional[List[str]] = None
    follow_xpath: Optional[List[str]] = None
    authentication: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None
    follow_external_links: bool = False


class ApiDocumentationOptions(BaseModel):
    enabled: bool = False
    api_endpoints: List[str] = Field(default_factory=list)
    api_discovery_urls: List[str] = Field(default_factory=list)
    include_common_apis: bool = True
    validate_endpoints: bool = True
    rate_limit_test: bool = False
    timeout_seconds: int = 10
    export_documentation: Optional[str] = None
    save_to_repository: bool = True


class SourcesOptions(BaseModel):
    audit: AuditOptions = Field(default_factory=AuditOptions)
    categorize: CategorizeOptions = Field(default_factory=CategorizeOptions)
    document_apis: ApiDocumentationOptions = Field(default_factory=ApiDocumentationOptions)
    databases: List[DatabaseConnectorOptions] = Field(default_factory=list)
    web: List[WebSourceOptions] = Field(default_factory=list)


class PipelineSpec(BaseModel):
    documents: DocumentOption
    storage: StorageOptions = Field(default_factory=StorageOptions)
    schedule: ScheduleOptions = Field(default_factory=ScheduleOptions)
    processors: ProcessorOptions = Field(default_factory=ProcessorOptions)
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    limits: LimitOptions = Field(default_factory=LimitOptions)
    dedupe: DedupeOptions = Field(default_factory=DedupeOptions)
    sources: SourcesOptions = Field(default_factory=SourcesOptions)

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
