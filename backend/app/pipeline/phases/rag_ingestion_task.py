"""Document Ingestion Phase.

This phase handles document discovery, processing, normalization, chunking,
and storage to prepare documents for embedding generation.

All implementation logic is contained within this module - no external orchestrator needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.pipeline.config import PipelineSpec
from app.domain.exceptions import (
    ContentExtractionError,
    DocumentError,
    DocumentPathNotFoundError,
    MetadataEnrichmentError,
    NormalizationError,
    ProcessingError,
    QualityValidationError,
    SourceConfigurationError,
    SourceNotReachableError,
    StorageWriteError,
    ChunkingError,
)
from app.domain.entities.document import Document
from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentFormat
from app.application.use_cases.chunking import (
    SemanticChunkingRequest,
    SemanticChunkingUseCase,
)
from app.pipeline.contracts import (
    DocumentSource,
    MetadataAnnotation,
    QualityReport,
    RunSummary,
    SourceContent,
    StandardSourceContent,
    StorageObject,
)

from app.application.use_cases.document_processing import (
    ProcessPdfsUseCase,
    ProcessPdfsRequest,
    NormalizeTextUseCase,
    NormalizeTextRequest,
    ExtractMetadataUseCase,
    ExtractMetadataRequest,
    ValidateQualityUseCase,
    ValidateQualityRequest,
)
from app.application.use_cases.data_ingestion import (
    IntegrateTikaUseCase,
    IntegrateTikaRequest,
    ConnectDatabasesUseCase,
    ConnectDatabasesRequest,
)
from app.application.use_cases.data_ingestion.connect_databases import DatabaseConnectorSpec
from app.application.use_cases.data_ingestion.setup_web_scraping import (
    SetupWebScrapingUseCase,
    SetupWebScrapingRequest,
)
from app.application.use_cases.knowledge_sources.audit_data_sources import (
    AuditDataSourcesRequest,
    AuditDataSourcesUseCase,
)
from app.application.use_cases.knowledge_sources.categorize_sources import (
    CategorizeSourcesRequest,
    CategorizeSourcesUseCase,
)
from app.application.use_cases.knowledge_sources.document_external_apis import (
    DocumentExternalApisRequest,
    DocumentExternalApisUseCase,
)
from app.application.use_cases.storage_management import (
    SetupObjectStorageRequest,
    SetupObjectStorageUseCase,
)
from app.infrastructure.external_services.data_sources.database_connectors import DatabaseConnectionConfig
from app.infrastructure.repositories.in_memory_data_source_repository import InMemoryDataSourceRepository
from app.infrastructure.storage.object_storage.minio_client import ObjectStorageClient, ObjectStorageConfig
from app.infrastructure.external_services.ml_services.categorization.sklearn_categorizer import SklearnDocumentCategorizer
from app.infrastructure.external_services import ExternalServicesContainer
from app.config.ml_models import get_chunking_config
from app.pipeline.chunking_config_resolver import resolve_chunking_configuration, requires_embedding_model
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.core.settings import get_settings
from app.pipeline.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class UseCaseBundle:
    """Bundle of use cases required for ingestion pipeline execution."""
    repository: InMemoryDataSourceRepository
    document_repository: DocumentRepository
    storage: SetupObjectStorageUseCase
    process_pdfs: ProcessPdfsUseCase
    normalize_text: NormalizeTextUseCase
    extract_metadata: ExtractMetadataUseCase
    validate_quality: ValidateQualityUseCase
    tika: IntegrateTikaUseCase
    connect_databases: ConnectDatabasesUseCase
    web_scraping: SetupWebScrapingUseCase
    audit_sources: AuditDataSourcesUseCase
    categorize_sources: CategorizeSourcesUseCase
    document_apis: DocumentExternalApisUseCase
    chunk_documents: SemanticChunkingUseCase


@dataclass
class RagIngestionTaskReport:
    """Report for document ingestion and processing task execution.

    Wraps RunSummary with detailed timing metadata for all pipeline steps.
    Tracks execution time for: setup, discovery, processing stages, and chunking.
    """

    success: bool = False
    execution_time_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    phase_result: Optional[RunSummary] = None
    total_errors: int = 0
    total_warnings: int = 0

    # Pipeline step timing breakdown (all times in seconds)
    timing_breakdown: Dict[str, float] = None

    def __post_init__(self):
        """Initialize timing breakdown if not provided."""
        if self.timing_breakdown is None:
            self.timing_breakdown = {}

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with detailed pipeline timing breakdown."""
        result_dict = {}
        if self.phase_result and hasattr(self.phase_result, 'as_dict'):
            result_dict = self.phase_result.as_dict()
        elif self.phase_result:
            result_dict = self.phase_result.__dict__

        # Format timing breakdown with rounded values
        formatted_timing = {
            step: round(duration, 3) for step, duration in (self.timing_breakdown or {}).items()
        }

        return {
            "task_name": "Document Ingestion",
            "success": self.success,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "pipeline_timing": {
                "total_seconds": round(self.execution_time_seconds, 2),
                "breakdown": formatted_timing,
                "summary": self._compute_timing_summary(formatted_timing),
            },
            "phase_result": result_dict,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings
        }

    def _compute_timing_summary(self, timing_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """Compute summary statistics for pipeline timing."""
        if not timing_breakdown:
            return {"steps": 0, "fastest": None, "slowest": None, "average": None}

        values = list(timing_breakdown.values())
        return {
            "steps": len(timing_breakdown),
            "fastest": min(values),
            "slowest": max(values),
            "average": sum(values) / len(values),
        }

    def as_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.as_dict(), indent=indent, default=str)

    # Convenience properties for easy access to key metrics
    @property
    def discovered(self) -> int:
        """Number of documents discovered."""
        if self.phase_result and self.phase_result.counters:
            return self.phase_result.counters.get('discovered', 0)
        return 0

    @property
    def processed(self) -> int:
        """Number of documents processed."""
        if self.phase_result:
            return self.phase_result.processed
        return 0

    @property
    def skipped(self) -> int:
        """Number of documents skipped."""
        if self.phase_result:
            return self.phase_result.skipped
        return 0

    @property
    def failed(self) -> int:
        """Number of documents that failed processing."""
        if self.phase_result:
            return self.phase_result.failed
        return 0

    @property
    def chunking_summary(self) -> Dict[str, Any]:
        """Chunking results summary."""
        if self.phase_result and self.phase_result.counters:
            return self.phase_result.counters.get('chunking', {})
        return {}

    @property
    def counters(self) -> Dict[str, Any]:
        """Direct access to counters for backward compatibility."""
        if self.phase_result:
            return self.phase_result.counters or {}
        return {}

    @property
    def errors(self) -> List[Dict[str, object]]:
        """Direct access to errors for backward compatibility."""
        if self.phase_result:
            return self.phase_result.errors or []
        return []


class RagIngestionTask:
    """Task 2: Document Ingestion and Processing.

    Handles document discovery, processing, normalization, chunking,
    and storage to prepare documents for embedding.

    All implementation logic is self-contained within this class - no external
    orchestrator dependencies.
    """

    def __init__(
        self,
        external_services: ExternalServicesContainer,
        document_repository: DocumentRepository,
        data_source_repository: Optional[DataSourceRepository] = None,
        verbose: bool = False,
    ):
        """Initialize the ingestion phase.

        Args:
            document_repository: Shared document repository instance (required).
                                Should be provided by the workflow orchestrator
                                to ensure data continuity across pipeline phases.
            external_services: External services container with embedding model and other services.
                        Ensures single service initialization.
            data_source_repository: Optional shared data source repository
            verbose: Enable verbose logging
        """
        self.document_repository = document_repository
        self.external_services = external_services
        self.data_source_repository = data_source_repository
        self.verbose = verbose
        self.tracker = PerformanceTracker()

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    # ==================== Helper Methods ====================

    def _build_use_cases(self, spec: PipelineSpec) -> UseCaseBundle:
        """Instantiate repositories and use cases required for pipeline run.

        Uses injected document repository for data continuity.

        Args:
            spec: Pipeline specification needed to configure chunking use case
        """
        repository = self.data_source_repository or InMemoryDataSourceRepository()

        return UseCaseBundle(
            repository=repository,
            document_repository=self.document_repository,  # Use injected repository!
            storage=SetupObjectStorageUseCase(),
            process_pdfs=ProcessPdfsUseCase(),
            normalize_text=NormalizeTextUseCase(),
            extract_metadata=ExtractMetadataUseCase(),
            validate_quality=ValidateQualityUseCase(),
            tika=IntegrateTikaUseCase(),
            connect_databases=ConnectDatabasesUseCase(),
            web_scraping=SetupWebScrapingUseCase(),
            audit_sources=AuditDataSourcesUseCase(repository),
            categorize_sources=CategorizeSourcesUseCase(repository, SklearnDocumentCategorizer()),
            document_apis=DocumentExternalApisUseCase(repository),
            # Only create SemanticChunkingUseCase with embedding model if semantic chunking is enabled
            # This prevents unnecessary loading of the embedding model when using sentence-based chunking
            chunk_documents=self._create_chunking_use_case(spec),
        )

    def _create_chunking_use_case(self, spec: PipelineSpec) -> SemanticChunkingUseCase:
        """Create chunking use case based on configuration.

        Only creates with embedding model if semantic chunking is enabled,
        preventing unnecessary model loading when using sentence-based chunking.
        """
        # Check if semantic chunking is enabled (respects YAML overrides)
        enable_semantic = requires_embedding_model(spec)

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Creating chunking use case with enable_semantic={enable_semantic}")

        if enable_semantic:
            # Semantic chunking requires embedding model
            return SemanticChunkingUseCase(
                document_repository=self.document_repository,
                embedding_model=self.external_services.embedding_model,
            )
        else:
            # Sentence-based chunking doesn't need embedding model
            return SemanticChunkingUseCase(
                document_repository=self.document_repository,
                embedding_model=None,  # No embedding model for sentence-based chunking
            )

    def _guess_content_type(self, path: Path) -> Optional[str]:
        """Map a file extension to a best-effort MIME type."""
        suf = path.suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".csv": "text/csv",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }.get(suf)

    def _infer_document_format(self, content_type: Optional[str], path: Optional[str]) -> DocumentFormat:
        """Determine the domain DocumentFormat from content-type metadata or file path."""
        if content_type:
            mapping = {
                "application/pdf": DocumentFormat.PDF,
                "text/markdown": DocumentFormat.MD,
                "text/plain": DocumentFormat.TXT,
                "text/html": DocumentFormat.HTML,
                "application/json": DocumentFormat.JSON,
                "text/csv": DocumentFormat.CSV,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.DOCX,
            }
            mapped = mapping.get(content_type)
            if mapped:
                return mapped

        if path:
            ext_mapping = {
                ".pdf": DocumentFormat.PDF,
                ".md": DocumentFormat.MD,
                ".txt": DocumentFormat.TXT,
                ".html": DocumentFormat.HTML,
                ".htm": DocumentFormat.HTML,
                ".json": DocumentFormat.JSON,
                ".csv": DocumentFormat.CSV,
                ".docx": DocumentFormat.DOCX,
                ".doc": DocumentFormat.DOCX,
            }
            ext = Path(path).suffix.lower()
            mapped = ext_mapping.get(ext)
            if mapped:
                return mapped

        return DocumentFormat.TXT

    def _build_document_entity(
        self,
        doc: DocumentSource,
        normalized: StandardSourceContent,
        enriched: MetadataAnnotation,
        quality: QualityReport,
        content: SourceContent,
        object_key: str,
        payload_size: int,
        run_id: str,
    ) -> Document:
        """Construct the domain `Document` with metadata and structured content payloads."""
        file_path = doc.path or doc.uri or f"object://{object_key}"
        if doc.path:
            file_name = Path(doc.path).name
        elif doc.uri:
            file_name = doc.uri.rstrip("/").split("/")[-1] or object_key.split("/")[-1]
        else:
            file_name = object_key.split("/")[-1]

        doc_format = self._infer_document_format(content.content_type, doc.path or doc.uri)

        metadata = DocumentMetadata(
            file_name=file_name,
            file_path=file_path,
            file_size=payload_size,
            format=doc_format,
            language=content.language or enriched.language,
            keywords=enriched.keywords or [],
            word_count=len(normalized.tokens) if normalized.tokens else None,
            custom_fields={
                "object_key": object_key,
                "origin": doc.origin,
                "quality_level": quality.quality_level,
                "overall_score": f"{quality.overall_score:.4f}",
                "run_id": run_id,
            },
        )

        document = Document(
            source_id=uuid4(),
            metadata=metadata,
            content=normalized.normalized,
        )
        document.complete_processing(
            extracted_text=normalized.normalized,
            structured_content={
                "entities": enriched.entities,
                "tables": content.tables,
                "metadata": content.metadata,
            },
        )
        return document

    def _content_hash(self, payload: bytes) -> str:
        """Generate a SHA256 digest used for deterministic storage keys."""
        return hashlib.sha256(payload).hexdigest()

    def _storage_client(self, spec: PipelineSpec) -> ObjectStorageClient:
        """Instantiate an object storage client using pipeline configuration values."""
        return ObjectStorageClient(
            ObjectStorageConfig(
                provider=spec.storage.provider,
                endpoint=spec.storage.endpoint,
                access_key=spec.storage.access_key,
                secret_key=spec.storage.secret_key,
                secure=spec.storage.secure,
                base_dir=spec.storage.base_dir,
            )
        )

    # ==================== Discovery Methods ====================

    def _discover_file_documents(self, spec: PipelineSpec) -> List[DocumentSource]:
        """Scan the configured filesystem path(s) and wrap matching files as sources."""
        root = spec.documents.path

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Starting file discovery at path: {root}")
            logger.debug(f"🔧 INGESTION VERBOSE: File exists check: {root.exists()}")

        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Created directory: {root}")

        if not root.exists():
            raise DocumentPathNotFoundError(
                f"Input path does not exist: {root}",
                document=str(root),
            )

        patterns = spec.documents.include or ["*.pdf"]
        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: File patterns to match: {patterns}")
            logger.debug(f"🔧 INGESTION VERBOSE: Recursive search: {spec.documents.recurse}")

        paths: List[Path] = []
        for pattern in patterns:
            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Searching for pattern: {pattern}")

            if spec.documents.recurse:
                matched_paths = list(root.rglob(pattern))
                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: Recursive search found {len(matched_paths)} matches for pattern '{pattern}'")
                paths.extend(matched_paths)
            else:
                matched_paths = list(root.glob(pattern))
                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: Non-recursive search found {len(matched_paths)} matches for pattern '{pattern}'")
                paths.extend(matched_paths)

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Total files found before filtering: {len(paths)}")

        documents: List[DocumentSource] = []
        for i, path in enumerate(paths):
            if path.is_file():
                content_type = self._guess_content_type(path)
                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: File {i+1}: {path} (type: {content_type}, size: {path.stat().st_size} bytes)")

                documents.append(
                    DocumentSource(
                        source_id=None,
                        uri=str(path),
                        path=str(path),
                        content_bytes=None,
                        content_type=content_type,
                        origin="file",
                    )
                )
            elif self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Skipped non-file: {path}")

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: File discovery completed: {len(documents)} documents found")

        return documents

    async def _discover_database_documents(
        self,
        spec: PipelineSpec,
        use_cases: UseCaseBundle,
        errors: Optional[List[Dict[str, object]]] = None,
    ) -> List[DocumentSource]:
        """Run each configured database connector and capture sample exports as sources."""
        documents: List[DocumentSource] = []

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Starting database discovery with {len(spec.sources.databases)} database configurations")

        for i, option in enumerate(spec.sources.databases):
            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Processing database {i+1}/{len(spec.sources.databases)}: {option.name}")
                logger.debug(f"🔧 INGESTION VERBOSE: Database config - driver: {option.driver}, host: {option.host}, port: {option.port}")
                logger.debug(f"🔧 INGESTION VERBOSE: Database config - connector_type: {option.connector_type}, fetch_samples: {option.fetch_samples}")

            config = DatabaseConnectionConfig(
                name=option.name,
                driver=option.driver,
                host=option.host,
                port=option.port,
                username=option.username,
                password=option.password,
                database=option.database,
                schema=option.db_schema,
                params=option.params,
                use_ssl=option.use_ssl,
            )
            spec_obj = DatabaseConnectorSpec(
                connector_type=option.connector_type,
                config=config,
                sample_collection=option.sample_collection,
                sample_limit=option.sample_limit,
            )

            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Attempting database connection to {option.name}")

            try:
                response = await use_cases.connect_databases.execute(
                    ConnectDatabasesRequest(connectors=[spec_obj], fetch_samples=option.fetch_samples)
                )

                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: Database {option.name} connected successfully")
                    logger.debug(f"🔧 INGESTION VERBOSE: Database {option.name} returned {len(response.results)} results")

            except Exception as exc:
                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: Database {option.name} connection failed: {exc}")
                logger.warning("Database connector '%s' failed: %s", option.name, exc)
                if errors is not None:
                    err = SourceNotReachableError(
                        f"Failed to connect to database source '{option.name}'",
                        source=option.name,
                        detail=str(exc),
                    )
                    errors.append(err.to_dict())
                continue
            for result in response.results:
                sample = result.sample
                if sample and sample.rows:
                    payload = sample.as_dict()
                    if result.collections is not None:
                        payload["collections"] = result.collections
                    serialized = json.dumps(payload, indent=2, default=str)
                    documents.append(
                        DocumentSource(
                            source_id=None,
                            uri=f"db://{option.name}",
                            path=None,
                            content_bytes=serialized.encode("utf-8"),
                            content_type="application/json",
                            origin="database",
                        )
                    )
                elif result.error:
                    logger.warning("Database connector '%s' failed: %s", option.name, result.error)
                    if errors is not None:
                        err = SourceNotReachableError(
                            f"Database connector '{option.name}' reported an error",
                            source=option.name,
                            detail=str(result.error),
                        )
                        errors.append(err.to_dict())
        return documents

    async def _discover_web_documents(
        self,
        spec: PipelineSpec,
        use_cases: UseCaseBundle,
        errors: Optional[List[Dict[str, object]]] = None,
    ) -> List[DocumentSource]:
        """Execute web scraping setups and translate crawled pages into document sources."""
        documents: List[DocumentSource] = []
        for option in spec.sources.web:
            request = SetupWebScrapingRequest(
                start_urls=option.start_urls,
                allowed_domains=option.allowed_domains,
                max_depth=option.max_depth,
                max_pages=option.max_pages,
                include_patterns=option.include_patterns,
                exclude_patterns=option.exclude_patterns,
                follow_css_selectors=option.follow_css_selectors,
                follow_xpath=option.follow_xpath,
                authentication=option.authentication,
                headers=option.headers,
                cookies=option.cookies,
                follow_external_links=option.follow_external_links,
            )
            try:
                response = await use_cases.web_scraping.execute(request)
            except Exception as exc:
                logger.warning("Web scraping setup failed for start_urls=%s: %s", option.start_urls, exc)
                if errors is not None:
                    err = SourceNotReachableError(
                        "Failed to initialize web scraping source",
                        source=", ".join(option.start_urls),
                        detail=str(exc),
                    )
                    errors.append(err.to_dict())
                continue
            for page in response.result.pages:
                if not page.content:
                    continue
                documents.append(
                    DocumentSource(
                        source_id=None,
                        uri=page.url,
                        path=None,
                        content_bytes=page.content.encode("utf-8"),
                        content_type="text/html",
                        origin="web",
                    )
                )
        return documents

    async def _document_external_apis(
        self,
        spec: PipelineSpec,
        use_cases: UseCaseBundle,
        errors: Optional[List[Dict[str, object]]] = None,
    ) -> List[DocumentSource]:
        """Document external APIs and produce JSON payloads representing each API surface."""
        options = spec.sources.document_apis
        if not options.enabled:
            return []
        request = DocumentExternalApisRequest(
            api_endpoints=options.api_endpoints or None,
            api_discovery_urls=options.api_discovery_urls or None,
            include_common_apis=options.include_common_apis,
            validate_endpoints=options.validate_endpoints,
            rate_limit_test=options.rate_limit_test,
            timeout_seconds=options.timeout_seconds,
            save_to_repository=options.save_to_repository,
            export_documentation=options.export_documentation,
        )
        try:
            response = await use_cases.document_apis.execute(request)
        except Exception as exc:
            logger.warning("External API documentation failed: %s", exc)
            if errors is not None:
                err = SourceConfigurationError(
                    "Failed to document external APIs",
                    source="document_apis",
                    detail=str(exc),
                )
                errors.append(err.to_dict())
            return []
        documents: List[DocumentSource] = []
        for api in response.documented_apis:
            payload = {
                "name": api.name,
                "base_url": api.base_url,
                "description": api.description,
                "version": api.version,
                "auth_methods": api.auth_methods,
                "rate_limits": api.rate_limits,
                "documentation_url": api.documentation_url,
                "openapi_spec_url": api.openapi_spec_url,
                "requires_key": api.requires_key,
                "health_check_url": api.health_check_url,
                "status": api.status,
                "response_time_ms": api.response_time_ms,
            }
            documents.append(
                DocumentSource(
                    source_id=None,
                    uri=f"api://{api.name}",
                    path=None,
                    content_bytes=json.dumps(payload, indent=2).encode("utf-8"),
                    content_type="application/json",
                    origin="api",
                )
            )
        return documents

    # ==================== Infrastructure Methods ====================

    async def _maybe_audit_sources(self, spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
        """Optionally scan configured sources and persist audit artifacts."""
        options = spec.sources.audit
        if not options.enabled:
            return
        request = AuditDataSourcesRequest(
            scan_paths=options.scan_paths or None,
            database_configs=options.database_configs or None,
            api_endpoints=options.api_endpoints or None,
            max_scan_depth=options.max_scan_depth,
            min_file_count=options.min_file_count,
            export_csv_path=options.export_csv,
            export_json_path=options.export_json,
            save_to_repository=options.save_to_repository,
        )
        await use_cases.audit_sources.execute(request)

    async def _maybe_categorize_sources(self, spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
        """Optionally train or run the document source categorization workflow."""
        options = spec.sources.categorize
        if not options.enabled:
            return
        request = CategorizeSourcesRequest(
            train_model=options.train_model,
            use_synthetic_data=options.use_synthetic_data,
            synthetic_data_size=options.synthetic_data_size,
            categorize_sources=options.categorize_sources,
            source_ids=options.source_ids,
            min_confidence=options.min_confidence,
            model_type=options.model_type,
            save_model=options.save_model,
            model_path=options.model_path,
        )
        await use_cases.categorize_sources.execute(request)

    async def _prepare_storage(self, spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
        """Provision object storage buckets and credentials prior to ingestion."""
        storage_base = str(spec.storage.base_dir) if spec.storage.base_dir else None
        request = SetupObjectStorageRequest(
            provider=spec.storage.provider,
            endpoint=spec.storage.endpoint,
            access_key=spec.storage.access_key,
            secret_key=spec.storage.secret_key,
            secure=spec.storage.secure,
            base_dir=storage_base,
            buckets=[spec.storage.bucket],
            test_object=False,
        )
        await use_cases.storage.execute(request)

    async def _init_pipeline(self, spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
        """Initialize pipeline dependencies and run infrastructure setup steps."""
        await self._prepare_storage(spec, use_cases)
        await self._maybe_audit_sources(spec, use_cases)
        await self._maybe_categorize_sources(spec, use_cases)

    # ==================== Processing Methods ====================

    async def _extract_content(self, doc: DocumentSource, spec: PipelineSpec, use_cases: UseCaseBundle) -> SourceContent:
        """Pull raw text/markdown from a document source using Docling for PDFs or Tika otherwise."""
        try:
            content_type = doc.content_type
            if doc.path:
                path = Path(doc.path)
                content_type = content_type or self._guess_content_type(path)
                if content_type == "application/pdf":
                    response = await use_cases.process_pdfs.execute(ProcessPdfsRequest(file_paths=[str(path)]))
                    record = response.documents[0] if response.documents else None
                    if record is None:
                        return SourceContent(
                            text="",
                            markdown="",
                            tables=[],
                            metadata={},
                            content_type=content_type,
                            language=None,
                        )
                    return SourceContent(
                        text=record.markdown or record.text or "",
                        markdown=record.markdown,
                        tables=record.tables,
                        metadata={"processor": "docling", "source": str(path)},
                        content_type=content_type,
                        language=None,
                        warnings=record.warnings,
                    )
                # Non-PDF path -> Tika
                response = await use_cases.tika.execute(IntegrateTikaRequest(file_paths=[str(path)]))
                record = response.documents[0] if response.documents else None
                if record is None:
                    return SourceContent(
                        text="",
                        markdown="",
                        tables=[],
                        metadata={},
                        content_type=content_type,
                        language=None,
                    )
                result = record.result
                return SourceContent(
                    text=result.text or "",
                    markdown=None,
                    tables=[],
                    metadata=result.metadata or {},
                    content_type=result.content_type or content_type,
                    language=result.language,
                    warnings=result.warnings,
                )

            # Inline bytes
            filename = doc.uri or "document"
            if doc.content_type == "application/pdf":
                response = await use_cases.process_pdfs.execute(
                    ProcessPdfsRequest(raw_pdfs=[(doc.content_bytes or b"", filename + ".pdf")])
                )
                record = response.documents[0] if response.documents else None
                if record is None:
                    return SourceContent(
                        text="",
                        markdown="",
                        tables=[],
                        metadata={},
                        content_type=doc.content_type,
                        language=None,
                    )
                return SourceContent(
                    text=record.markdown or record.text or "",
                    markdown=record.markdown,
                    tables=record.tables,
                    metadata={"processor": "docling", "source": filename},
                    content_type=doc.content_type,
                    language=None,
                    warnings=record.warnings,
                )

            response = await use_cases.tika.execute(
                IntegrateTikaRequest(raw_documents=[(doc.content_bytes or b"", filename)])
            )
            record = response.documents[0] if response.documents else None
            if record is None:
                return SourceContent(
                    text="",
                    markdown="",
                    tables=[],
                    metadata={},
                    content_type=doc.content_type,
                    language=None,
                )
            result = record.result
            return SourceContent(
                text=result.text or "",
                markdown=None,
                tables=[],
                metadata=result.metadata or {},
                content_type=result.content_type or doc.content_type,
                language=result.language,
                warnings=result.warnings,
            )
        except ContentExtractionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            source_label = doc.uri or doc.path or "inline"
            raise ContentExtractionError(
                f"Failed to extract content for {source_label}",
                stage="extract_content",
                source=source_label,
                detail=str(exc),
            ) from exc

    async def _normalize_content(self, text: str, use_cases: UseCaseBundle) -> StandardSourceContent:
        """Apply text normalization to produce consistent tokens, lemmas, and entity stubs."""
        try:
            response = await use_cases.normalize_text.execute(
                NormalizeTextRequest(texts=[text], lowercase=False, correct_typos=False)
            )
            record = response.results[0] if response.results else None
            if record is None:
                return StandardSourceContent(normalized=text, tokens=[], lemmas=[], entities=[], warnings=[])
            return StandardSourceContent(
                normalized=record.normalized,
                tokens=record.tokens,
                lemmas=record.lemmas,
                entities=record.entities,
                warnings=[],
            )
        except NormalizationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise NormalizationError(
                "Failed to normalize content",
                stage="normalize_content",
                detail=str(exc),
            ) from exc

    async def _enrich_content(self, text: str, use_cases: UseCaseBundle) -> MetadataAnnotation:
        """Extract metadata annotations such as keywords, entities, and authors from text."""
        try:
            response = await use_cases.extract_metadata.execute(ExtractMetadataRequest(texts=[text], top_k_keywords=10))
            record = response.results[0] if response.results else None
            if record is None:
                return MetadataAnnotation(keywords=[], entities=[], dates=[], authors=[], language=None)
            return MetadataAnnotation(
                keywords=record.keywords,
                entities=record.entities,
                dates=record.dates,
                authors=record.authors,
                language=record.language,
                warnings=[],
            )
        except MetadataEnrichmentError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MetadataEnrichmentError(
                "Failed to extract metadata annotations",
                stage="metadata_enrichment",
                detail=str(exc),
            ) from exc

    async def _validate_content(self, text: str, use_cases: UseCaseBundle) -> QualityReport:
        """Score normalized text against quality heuristics and validation rules."""
        try:
            response = await use_cases.validate_quality.execute(ValidateQualityRequest(texts=[text]))
            record = response.reports[0] if response.reports else None
            if record is None:
                return QualityReport(
                    overall_score=0.0,
                    quality_level="unknown",
                    metrics={},
                    validations=[],
                    ge_report=None,
                )
            return QualityReport(
                overall_score=record.overall_score,
                quality_level=record.quality_level,
                metrics=record.metrics,
                validations=record.validations,
                ge_report=record.ge_report,
            )
        except QualityValidationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise QualityValidationError(
                "Failed to validate content quality",
                stage="quality_validation",
                detail=str(exc),
            ) from exc

    # ==================== Pipeline Orchestration Methods ====================

    async def _discover_documents(
        self,
        spec: PipelineSpec,
        use_cases: UseCaseBundle,
    ) -> tuple[List[DocumentSource], List[Dict[str, object]]]:
        """Aggregate documents from configured sources while recording discovery issues."""
        documents: List[DocumentSource] = []
        errors: List[Dict[str, object]] = []

        try:
            documents.extend(self._discover_file_documents(spec))
        except DocumentError as exc:
            logger.warning("Document discovery error: %s", exc)
            errors.append(exc.to_dict())
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected file discovery error")
            errors.append(
                {
                    "error_type": "UnexpectedDocumentDiscoveryError",
                    "message": str(exc),
                    "detail": repr(exc),
                }
            )

        try:
            documents.extend(await self._discover_database_documents(spec, use_cases, errors))
        except SourceNotReachableError as exc:
            logger.warning("Database discovery error: %s", exc)
            errors.append(exc.to_dict())
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected database discovery error")
            err = SourceNotReachableError(
                "Unexpected database discovery failure",
                source="databases",
                detail=str(exc),
            )
            errors.append(err.to_dict())

        try:
            documents.extend(await self._discover_web_documents(spec, use_cases, errors))
        except SourceNotReachableError as exc:
            logger.warning("Web discovery error: %s", exc)
            errors.append(exc.to_dict())
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected web discovery error")
            err = SourceNotReachableError(
                "Unexpected web discovery failure",
                source="web",
                detail=str(exc),
            )
            errors.append(err.to_dict())

        try:
            documents.extend(await self._document_external_apis(spec, use_cases, errors))
        except SourceConfigurationError as exc:
            logger.warning("API discovery error: %s", exc)
            errors.append(exc.to_dict())
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected API discovery error")
            err = SourceConfigurationError(
                "Unexpected API documentation failure",
                source="document_apis",
                detail=str(exc),
            )
            errors.append(err.to_dict())

        if spec.limits.max_files is not None:
            documents = documents[: spec.limits.max_files]

        return documents, errors

    async def _process_documents(
        self,
        documents: List[DocumentSource],
        spec: PipelineSpec,
        use_cases: UseCaseBundle,
        client: ObjectStorageClient,
        run_id: str,
        errors: List[Dict[str, object]],
    ) -> tuple[List[StorageObject], Dict[str, int], int, List[UUID]]:
        """Execute extraction, normalization, enrichment, validation, and persistence.

        Tracks timing for each processing stage:
        - extract_content: Content extraction using Docling/Tika
        - normalize_content: Text normalization and tokenization
        - enrich_metadata: Keyword and entity extraction
        - validate_quality: Quality scoring
        - store_document: Storage and repository persistence
        """
        stored_objects: List[StorageObject] = []
        origin_counts: Dict[str, int] = {}
        skipped = 0
        processed_document_ids: List[UUID] = []

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Starting processing of {len(documents)} documents")

        for i, doc in enumerate(documents):
            try:
                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: Processing document {i+1}/{len(documents)}: {doc.uri or doc.path or 'inline'}")
                    logger.debug(f"🔧 INGESTION VERBOSE: Document origin: {doc.origin}, content_type: {doc.content_type}")

                # Stage 1: Extract Content
                async with self.tracker.with_tracker("extract_content"):
                    content = await self._extract_content(doc, spec, use_cases)

                base_text = content.markdown or content.text or ""

                if self.verbose:
                    extract_stats = self.tracker.get_stats("extract_content")
                    if extract_stats:
                        logger.debug(f"🔧 INGESTION VERBOSE: Extracted content length: {len(base_text)} characters (took {extract_stats.get('average', 0):.3f}s)")

                if not base_text:
                    if self.verbose:
                        logger.debug(f"🔧 INGESTION VERBOSE: Skipping document {i+1} - no content extracted")
                    skipped += 1
                    continue

                # Stage 2: Normalize Content
                async with self.tracker.with_tracker("normalize_content"):
                    normalized = await self._normalize_content(base_text, use_cases)

                if self.verbose:
                    normalize_stats = self.tracker.get_stats("normalize_content")
                    if normalize_stats:
                        logger.debug(f"🔧 INGESTION VERBOSE: Normalization complete - tokens: {len(normalized.tokens) if normalized.tokens else 0} (took {normalize_stats.get('average', 0):.3f}s)")

                # Stage 3: Enrich Metadata
                async with self.tracker.with_tracker("enrich_metadata"):
                    enriched = await self._enrich_content(normalized.normalized, use_cases)

                if self.verbose:
                    enrich_stats = self.tracker.get_stats("enrich_metadata")
                    if enrich_stats:
                        logger.debug(f"🔧 INGESTION VERBOSE: Metadata enrichment complete - keywords: {len(enriched.keywords) if enriched.keywords else 0}, entities: {len(enriched.entities) if enriched.entities else 0} (took {enrich_stats.get('average', 0):.3f}s)")

                # Stage 4: Validate Quality
                async with self.tracker.with_tracker("validate_quality"):
                    quality = await self._validate_content(normalized.normalized, use_cases)

                if self.verbose:
                    validate_stats = self.tracker.get_stats("validate_quality")
                    if validate_stats:
                        logger.debug(f"🔧 INGESTION VERBOSE: Quality validation complete - score: {quality.overall_score:.4f}, level: {quality.quality_level} (took {validate_stats.get('average', 0):.3f}s)")

                now = datetime.now(timezone.utc)
                dated = now.strftime("%Y/%m/%d")
                payload = normalized.normalized.encode("utf-8")
                object_key = f"{dated}/{self._content_hash(payload)[:16]}.md"

                tags = {
                    "validated": "true" if quality.overall_score >= 0.5 else "false",
                    "origin": doc.origin,
                }
                metadata = {
                    "source": doc.uri or doc.path or "unknown",
                    "keywords": ",".join(enriched.keywords[:10]) if enriched.keywords else "",
                    "overall_score": str(quality.overall_score),
                    "quality_level": quality.quality_level,
                    "run_id": run_id,
                }

                # Stage 5: Store Document (MinIO + Repository)
                async with self.tracker.with_tracker("store_document"):
                    try:
                        result = client.put_object(
                            spec.storage.bucket,
                            object_key,
                            payload,
                            content_type="text/markdown",
                            metadata=metadata,
                            tags=tags,
                        )
                    except Exception as exc:
                        raise StorageWriteError(
                            "Failed to write processed document to storage",
                            stage="storage",
                            source=doc.uri or doc.path,
                            detail=str(exc),
                        ) from exc

                    stored_objects.append(
                        StorageObject(
                            bucket=result.bucket,
                            key=result.key,
                            size=result.size,
                            metadata={k: str(v) for k, v in metadata.items()},
                            tags=tags,
                        )
                    )

                    document_entity = self._build_document_entity(
                        doc=doc,
                        normalized=normalized,
                        enriched=enriched,
                        quality=quality,
                        content=content,
                        object_key=object_key,
                        payload_size=result.size,
                        run_id=run_id,
                    )
                    await use_cases.document_repository.save(document_entity)
                    processed_document_ids.append(document_entity.id)
                    origin_counts[doc.origin] = origin_counts.get(doc.origin, 0) + 1

                if self.verbose:
                    store_stats = self.tracker.get_stats("store_document")
                    if store_stats:
                        logger.debug(f"🔧 INGESTION VERBOSE: Document stored - key: {object_key}, size: {result.size} bytes (took {store_stats.get('average', 0):.3f}s)")
            except ProcessingError as exc:
                logger.error(
                    "Processing error at stage '%s' for %s: %s",
                    exc.stage or "unknown",
                    doc.uri or doc.path or "inline",
                    exc,
                )
                errors.append(exc.to_record())
            except Exception as exc:  # pragma: no cover
                source_label = doc.uri or doc.path or "inline"
                logger.exception("Unexpected failure while processing: %s", source_label)
                errors.append(
                    {
                        "error_type": "UnexpectedProcessingError",
                        "message": str(exc),
                        "source": source_label,
                        "detail": repr(exc),
                    }
                )

        # Log stage timing summary using PerformanceTracker
        all_stats = self.tracker.get_all_stats()
        for operation, stats in all_stats.items():
            logger.info(f"Processing stage - {operation}: avg={stats['average']:.3f}s, total={stats['total']:.2f}s")

        return stored_objects, origin_counts, skipped, processed_document_ids

    async def _maybe_chunk_documents(
        self,
        processed_document_ids: List[UUID],
        use_cases: UseCaseBundle,
        spec: PipelineSpec,
        run_id: str,
        dry_run: bool,
        errors: List[Dict[str, object]],
    ) -> Dict[str, Any]:
        """Apply semantic chunking to processed documents when chunking is enabled."""
        chunk_options = getattr(spec, "chunking", None)
        chunk_enabled = True if chunk_options is None else getattr(chunk_options, "enabled", True)

        summary: Dict[str, Any] = {
            "documents": 0,
            "total_chunks": 0,
            "dry_run": dry_run,
            "enabled": chunk_enabled,
            "preset": getattr(chunk_options, "preset", None) or "balanced",
        }

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Chunking configuration - enabled: {chunk_enabled}, documents_to_chunk: {len(processed_document_ids)}")
            logger.debug(f"🔧 INGESTION VERBOSE: Chunking preset: {summary['preset']}, dry_run: {dry_run}")

        if not (chunk_enabled and processed_document_ids):
            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Skipping chunking - enabled={chunk_enabled}, documents_count={len(processed_document_ids)}")
            return summary

        try:
            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Loading chunking configuration for preset: {getattr(chunk_options, 'preset', None)}")

            # Get resolved chunking configuration with all overrides applied
            chunk_config = resolve_chunking_configuration(spec)

            # Apply additional overrides from chunk_options (if any)
            overrides = getattr(chunk_options, "overrides", {}) if chunk_options else {}
            if overrides:
                if self.verbose:
                    logger.debug(f"🔧 INGESTION VERBOSE: Applying additional chunking config overrides: {list(overrides.keys())}")
                chunk_config.update(overrides)

            # Use enable_semantic from configuration with YAML override applied
            enable_semantic = chunk_config.get("enable_semantic", True)
            chunking_method = "semantic (with embeddings)" if enable_semantic else "sentence (no embeddings)"
            logger.info(f"📝 Chunking method: {chunking_method} (from preset: {summary['preset']}, YAML overrides applied)")

            metadata_overrides = {"run_id": run_id}
            extra_metadata = getattr(chunk_options, "metadata_overrides", {}) if chunk_options else {}
            metadata_overrides.update(extra_metadata)

            replace_existing = getattr(chunk_options, "replace_existing_chunks", True) if chunk_options else True

            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Chunking config - replace_existing: {replace_existing}")
                logger.debug(f"🔧 INGESTION VERBOSE: Starting semantic chunking for {len(processed_document_ids)} documents")

            # Use PerformanceTracker for chunking timing
            chunk_result = await self.tracker.measure("chunking", use_cases.chunk_documents.execute,
                SemanticChunkingRequest(
                    document_ids=processed_document_ids,
                    replace_existing_chunks=replace_existing,
                    dry_run=dry_run,
                    chunker_config=chunk_config,
                    metadata_overrides=metadata_overrides,
                )
            )
            chunk_response = chunk_result.result
            chunk_duration = chunk_result.duration

            if self.verbose:
                logger.debug(f"🔧 INGESTION VERBOSE: Chunking completed in {chunk_duration:.2f}s - results: {len(chunk_response.results)}, total_chunks: {chunk_response.total_chunks}")
                if chunk_response.total_chunks > 0:
                    avg_time_per_chunk = chunk_duration / chunk_response.total_chunks
                    logger.debug(f"🔧 INGESTION VERBOSE: Performance - {chunk_response.total_chunks / chunk_duration:.1f} chunks/sec, {avg_time_per_chunk * 1000:.2f}ms per chunk")
                if hasattr(chunk_response, 'results') and chunk_response.results:
                    for result in chunk_response.results[:3]:  # Log first 3 results to avoid spam
                        logger.debug(f"🔧 INGESTION VERBOSE: Document {result.document_id}: {result.chunk_count} chunks")
            else:
                logger.info(f"Semantic chunking completed: {chunk_response.total_chunks} chunks in {chunk_duration:.2f}s ({chunk_response.total_chunks / chunk_duration:.1f} chunks/sec)")

            summary.update(
                {
                    "documents": len(chunk_response.results),
                    "total_chunks": chunk_response.total_chunks,
                    "dry_run": chunk_response.dry_run,
                }
            )
        except ChunkingError as exc:
            logger.error("Chunking error: %s", exc)
            errors.append(exc.to_dict())
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected chunking failure")
            errors.append(
                {
                    "error_type": "UnexpectedChunkingError",
                    "message": str(exc),
                    "detail": repr(exc),
                }
            )

        return summary

    # ==================== Main Execute Method ====================

    async def execute(
        self, spec: PipelineSpec, dry_run: bool = False
    ) -> RagIngestionTaskReport:
        """Execute document ingestion phase.

        Coordinates the complete ingestion workflow:
        1. Infrastructure setup (storage, audits, categorization)
        2. Document discovery (files, databases, web, APIs)
        3. Document processing (extract, normalize, enrich, validate, store)
        4. Post-processing (chunking based on preset)

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)

        Returns:
            RagIngestionTaskReport with processing results, stored documents, and timing breakdown
        """
        phase_start_time = datetime.now(timezone.utc)
        logger.info("Running ingestion with shared document repository...")

        # Track timing for each pipeline step
        timing_breakdown: Dict[str, float] = {}

        if self.verbose:
            logger.debug("🔧 INGESTION VERBOSE: Starting ingestion phase with verbose logging")
            logger.debug(f"🔧 INGESTION VERBOSE: dry_run={dry_run}")
            logger.debug(f"🔧 INGESTION VERBOSE: spec.documents.path={spec.documents.path}")
            logger.debug(f"🔧 INGESTION VERBOSE: spec.documents.include={spec.documents.include}")
            logger.debug(f"🔧 INGESTION VERBOSE: spec.documents.recurse={spec.documents.recurse}")
            logger.debug(f"🔧 INGESTION VERBOSE: spec.storage.provider={spec.storage.provider}")
            logger.debug(f"🔧 INGESTION VERBOSE: spec.storage.bucket={spec.storage.bucket}")

        # Build use case bundle with injected repositories
        use_cases = self._build_use_cases(spec)
        run_id = datetime.now(timezone.utc).isoformat()

        if self.verbose:
            logger.debug(f"🔧 INGESTION VERBOSE: Generated run_id={run_id}")
            all_data_sources = await use_cases.repository.find_all()
            logger.debug(f"🔧 INGESTION VERBOSE: Use case bundle built with {len(all_data_sources)} existing data sources")

        # Phase 1: Infrastructure setup
        logger.info("pipeline initialization started | phase=setup")
        async with self.tracker.with_tracker("setup"):
            await self._init_pipeline(spec, use_cases)

        # Phase 2: Discovery
        if self.verbose:
            logger.debug("🔧 INGESTION VERBOSE: Starting document discovery phase")

        async with self.tracker.with_tracker("discovery"):
            documents, errors = await self._discover_documents(spec, use_cases)

        discovery_error_count = len(errors)
        if self.verbose:
            discovery_stats = self.tracker.get_stats("discovery")
            if discovery_stats:
                logger.debug(f"🔧 INGESTION VERBOSE: Discovery completed - found {len(documents)} documents, {discovery_error_count} errors (took {discovery_stats.get('total', 0):.2f}s)")

        # Phase 3: Processing
        client = self._storage_client(spec)

        if self.verbose:
            logger.debug("🔧 INGESTION VERBOSE: Starting document processing phase")
            logger.debug(f"🔧 INGESTION VERBOSE: Storage client created for provider: {spec.storage.provider}")

        async with self.tracker.with_tracker("processing"):
            stored_objects, origin_counts, skipped, processed_document_ids = (
                await self._process_documents(
                    documents=documents,
                    spec=spec,
                    use_cases=use_cases,
                    client=client,
                    run_id=run_id,
                    errors=errors,
                )
            )

        if self.verbose:
            processing_stats = self.tracker.get_stats("processing")
            if processing_stats:
                logger.debug(f"🔧 INGESTION VERBOSE: Processing completed - stored: {len(stored_objects)}, skipped: {skipped}, processed_ids: {len(processed_document_ids)} (took {processing_stats.get('total', 0):.2f}s)")

        # Phase 4: Post-processing (chunking)
        if self.verbose:
            logger.debug("🔧 INGESTION VERBOSE: Starting chunking phase")

        async with self.tracker.with_tracker("chunking"):
            chunk_summary = await self._maybe_chunk_documents(
                processed_document_ids=processed_document_ids,
                use_cases=use_cases,
                spec=spec,
                run_id=run_id,
                dry_run=dry_run,
                errors=errors,
            )

        if self.verbose:
            chunking_stats = self.tracker.get_stats("chunking")
            if chunking_stats:
                logger.debug(f"🔧 INGESTION VERBOSE: Chunking completed - documents: {chunk_summary.get('documents', 0)}, chunks: {chunk_summary.get('total_chunks', 0)}, enabled: {chunk_summary.get('enabled', False)} (took {chunking_stats.get('total', 0):.2f}s)")

        # Build summary
        processing_failures = max(len(errors) - discovery_error_count, 0)
        counters = {
            "run_id": run_id,
            "discovered": len(documents),
            "processed": len(stored_objects),
            "skipped": skipped,
            "failed": processing_failures,
            "by_origin": origin_counts,
            "discovery_errors": discovery_error_count,
            "total_errors": len(errors),
            "chunking": chunk_summary,
        }

        summary = RunSummary(
            processed=len(stored_objects),
            skipped=skipped,
            failed=processing_failures,
            stored=stored_objects,
            errors=errors,
            counters=counters,
        )

        logger.info(
            f"Ingestion completed: {summary.processed} documents stored in shared repository"
        )

        phase_end_time = datetime.now(timezone.utc)
        execution_time = (phase_end_time - phase_start_time).total_seconds()

        # Get pipeline timing breakdown from PerformanceTracker
        pipeline_stats = self.tracker.get_all_stats()
        timing_breakdown = {op: stats['total'] for op, stats in pipeline_stats.items()}

        # Log timing summary
        logger.info(f"Pipeline timing breakdown: setup={timing_breakdown.get('setup', 0):.2f}s, discovery={timing_breakdown.get('discovery', 0):.2f}s, processing={timing_breakdown.get('processing', 0):.2f}s, chunking={timing_breakdown.get('chunking', 0):.2f}s, total={execution_time:.2f}s")

        return RagIngestionTaskReport(
            success=summary.processed > 0,
            execution_time_seconds=execution_time,
            start_time=phase_start_time,
            end_time=phase_end_time,
            phase_result=summary,
            total_errors=len(summary.errors),
            total_warnings=0,
            timing_breakdown=timing_breakdown,
        )
