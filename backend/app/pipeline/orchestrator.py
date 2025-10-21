from __future__ import annotations

"""Unified orchestrator that coordinates ingestion use cases for the RAG pipeline."""

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
from app.infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository
from app.infrastructure.storage.object_storage.minio_client import ObjectStorageClient, ObjectStorageConfig
from app.infrastructure.external_services.ml_services.categorization.sklearn_categorizer import SklearnDocumentCategorizer
from app.config.ml_models import get_chunking_config


logger = logging.getLogger(__name__)


@dataclass
class UseCaseBundle:
    repository: InMemoryDataSourceRepository
    document_repository: InMemoryDocumentRepository
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


def _build_use_cases() -> UseCaseBundle:
    """Instantiate repositories and use cases required for a single pipeline run."""

    repository = InMemoryDataSourceRepository()
    document_repository = InMemoryDocumentRepository()
    return UseCaseBundle(
        repository=repository,
        document_repository=document_repository,
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
        chunk_documents=SemanticChunkingUseCase(document_repository=document_repository),
    )


def _guess_content_type(path: Path) -> Optional[str]:
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


def _infer_document_format(content_type: Optional[str], path: Optional[str]) -> DocumentFormat:
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

    doc_format = _infer_document_format(content.content_type, doc.path or doc.uri)

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


def _discover_file_documents(spec: PipelineSpec) -> List[DocumentSource]:
    """Scan the configured filesystem path(s) and wrap matching files as sources."""

    root = spec.documents.path
    if not root.exists():
        raise DocumentPathNotFoundError(
            f"Input path does not exist: {root}",
            document=str(root),
        )
    patterns = spec.documents.include or ["*.pdf"]
    paths: List[Path] = []
    for pattern in patterns:
        if spec.documents.recurse:
            paths.extend(root.rglob(pattern))
        else:
            paths.extend(root.glob(pattern))

    documents: List[DocumentSource] = []
    for path in paths:
        if path.is_file():
            documents.append(
                DocumentSource(
                    source_id=None,
                    uri=str(path),
                    path=str(path),
                    content_bytes=None,
                    content_type=_guess_content_type(path),
                    origin="file",
                )
            )
    return documents


async def _discover_database_documents(
    spec: PipelineSpec,
    use_cases: UseCaseBundle,
    errors: Optional[List[Dict[str, object]]] = None,
) -> List[DocumentSource]:
    """Run each configured database connector and capture sample exports as sources."""

    documents: List[DocumentSource] = []
    for option in spec.sources.databases:
        config = DatabaseConnectionConfig(
            name=option.name,
            driver=option.driver,
            host=option.host,
            port=option.port,
            username=option.username,
            password=option.password,
            database=option.database,
            schema=option.schema,
            params=option.params,
            use_ssl=option.use_ssl,
        )
        spec_obj = DatabaseConnectorSpec(
            connector_type=option.connector_type,
            config=config,
            sample_collection=option.sample_collection,
            sample_limit=option.sample_limit,
        )
        try:
            response = await use_cases.connect_databases.execute(
                ConnectDatabasesRequest(connectors=[spec_obj], fetch_samples=option.fetch_samples)
            )
        except Exception as exc:
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


async def _maybe_audit_sources(spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
    """Optionally scan configured sources and persist audit artifacts.

    When `spec.sources.audit.enabled` is true, this step expands the audit options
    into an `AuditDataSourcesRequest`, triggering directory crawls, database metadata
    sampling, and API endpoint checks. Results can be exported to CSV/JSON and
    optionally written into the shared repository for later inspection.
    """

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


async def _maybe_categorize_sources(spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
    """Optionally train or run the document source categorization workflow.

    Builds a `CategorizeSourcesRequest` with model hyperparameters, synthetic data
    flags, and target source IDs so the categorization use case can label sources and
    persist trained models when requested.
    """

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


async def _prepare_storage(spec: PipelineSpec, use_cases: UseCaseBundle) -> None:
    """Provision object storage buckets and credentials prior to ingestion.

    Converts the storage section of the spec into a `SetupObjectStorageRequest`,
    ensuring credentials are valid, buckets exist (creating them if necessary), and
    optional local base directories are prepared.
    """

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


def _storage_client(spec: PipelineSpec) -> ObjectStorageClient:
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


async def _extract_content(doc: DocumentSource, spec: PipelineSpec, use_cases: UseCaseBundle) -> SourceContent:
    """Pull raw text/markdown from a document source using Docling for PDFs or Tika otherwise."""

    try:
        content_type = doc.content_type
        if doc.path:
            path = Path(doc.path)
            content_type = content_type or _guess_content_type(path)
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


async def _normalize_content(text: str, use_cases: UseCaseBundle) -> StandardSourceContent:
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


async def _enrich_content(text: str, use_cases: UseCaseBundle) -> MetadataAnnotation:
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


async def _validate_content(text: str, use_cases: UseCaseBundle) -> QualityReport:
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


def _content_hash(payload: bytes) -> str:
    """Generate a SHA256 digest used for deterministic storage keys."""

    return hashlib.sha256(payload).hexdigest()


async def _init_pipeline(spec: PipelineSpec) -> UseCaseBundle:
    """Initialize pipeline dependencies and run infrastructure setup steps.

    Phase alignment: Infrastructure Preparation. The function builds the
    `UseCaseBundle`, provisions storage buckets, and executes optional audits or
    categorization so downstream stages can assume foundational services exist.

    Example:
        >>> spec = PipelineSpec.from_file("rag_pipeline.yaml")
        >>> use_cases = await _init_pipeline(spec)
        >>> sorted(use_cases.repository.list_all())
        []
        The example highlights that initialization completes without
        ingesting documents while ensuring repositories and storage clients are ready.
    """

    use_cases = _build_use_cases()
    await _prepare_storage(spec, use_cases)
    await _maybe_audit_sources(spec, use_cases)
    await _maybe_categorize_sources(spec, use_cases)
    return use_cases


async def _discover_documents(
    spec: PipelineSpec,
    use_cases: UseCaseBundle,
) -> tuple[List[DocumentSource], List[Dict[str, object]]]:
    """Aggregate documents from configured sources while recording discovery issues.

    Phase alignment: Discovery. Each configured connector contributes
    `DocumentSource` entries, and recoverable failures are translated into structured
    error dictionaries so that the overall pipeline can surface precise diagnostics
    without aborting the run.

    Example:
        >>> documents, errors = await _discover_documents(spec, use_cases)
        >>> summary = (len(documents), [err["source"] for err in errors])
        >>> summary
        (42, ['marketing-db'])
        The snapshot shows forty-two documents ready for processing and
        a single discovery error attributed to the `marketing-db` source.
    """

    documents: List[DocumentSource] = []
    errors: List[Dict[str, object]] = []

    try:
        documents.extend(_discover_file_documents(spec))
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
        documents.extend(await _discover_database_documents(spec, use_cases, errors))
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
        documents.extend(await _discover_web_documents(spec, use_cases, errors))
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
        documents.extend(await _document_external_apis(spec, use_cases, errors))
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
    documents: List[DocumentSource],
    spec: PipelineSpec,
    use_cases: UseCaseBundle,
    client: ObjectStorageClient,
    run_id: str,
    errors: List[Dict[str, object]],
) -> tuple[List[StorageObject], Dict[str, int], int, List[UUID]]:
    """Execute extraction, normalization, enrichment, validation, and persistence.

    Phase alignment: Processing. Every `DocumentSource` that yields content
    is transformed into normalized markdown, enriched metadata, quality scores, and a
    stored object. Stage-specific failures are appended to `errors` with rich context
    so later summaries can distinguish processing issues from discovery issues.

    Example:
        >>> stored, by_origin, skipped, ids = await _process_documents(docs, spec, use_cases, client, run_id, [])
        >>> by_origin
        {'file': 30, 'web': 5}
        In this excerpt, thirty-five documents produced stored objects while
        any empty-bodied sources increased the `skipped` counter instead of raising.
    """

    stored_objects: List[StorageObject] = []
    origin_counts: Dict[str, int] = {}
    skipped = 0
    processed_document_ids: List[UUID] = []

    for doc in documents:
        try:
            content = await _extract_content(doc, spec, use_cases)
            base_text = content.markdown or content.text or ""
            if not base_text:
                skipped += 1
                continue

            normalized = await _normalize_content(base_text, use_cases)
            enriched = await _enrich_content(normalized.normalized, use_cases)
            quality = await _validate_content(normalized.normalized, use_cases)

            now = datetime.now(timezone.utc)
            dated = now.strftime("%Y/%m/%d")
            payload = normalized.normalized.encode("utf-8")
            object_key = f"{dated}/{_content_hash(payload)[:16]}.md"

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

            document_entity = _build_document_entity(
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

    return stored_objects, origin_counts, skipped, processed_document_ids


async def _maybe_chunk_documents(
    processed_document_ids: List[UUID],
    use_cases: UseCaseBundle,
    spec: PipelineSpec,
    run_id: str,
    dry_run: bool,
    errors: List[Dict[str, object]],
) -> Dict[str, Any]:
    """Apply semantic chunking to processed documents when chunking is enabled.

    Phase alignment: Post-processing. The function respects chunking presets
    and overrides from the spec, enriches chunks with run metadata, and appends any
    chunking errors to the shared list without interrupting prior results.

    Example:
        >>> summary = await _maybe_chunk_documents(ids, use_cases, spec, run_id, dry_run=False, errors=[])
        >>> summary["documents"], summary["total_chunks"]
        (12, 220)
        The outcome demonstrates that twelve documents expanded into two
        hundred twenty chunks with dry-run disabled.
    """

    chunk_options = getattr(spec, "chunking", None)
    chunk_enabled = True if chunk_options is None else getattr(chunk_options, "enabled", True)

    summary: Dict[str, Any] = {
        "documents": 0,
        "total_chunks": 0,
        "dry_run": dry_run,
        "enabled": chunk_enabled,
        "preset": getattr(chunk_options, "preset", None) or "balanced",
    }

    if not (chunk_enabled and processed_document_ids):
        return summary

    try:
        chunk_config = get_chunking_config(getattr(chunk_options, "preset", None))
        overrides = getattr(chunk_options, "overrides", {}) if chunk_options else {}
        if overrides:
            chunk_config.update(overrides)

        metadata_overrides = {"run_id": run_id}
        extra_metadata = getattr(chunk_options, "metadata_overrides", {}) if chunk_options else {}
        metadata_overrides.update(extra_metadata)

        replace_existing = getattr(chunk_options, "replace_existing_chunks", True) if chunk_options else True

        chunk_response = await use_cases.chunk_documents.execute(
            SemanticChunkingRequest(
                document_ids=processed_document_ids,
                replace_existing_chunks=replace_existing,
                dry_run=dry_run,
                chunker_config=chunk_config,
                metadata_overrides=metadata_overrides,
            )
        )
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


async def run_pipeline(spec: PipelineSpec, dry_run: bool = False) -> RunSummary:
    """Coordinate the RAG ingestion workflow across the aligned phases.

    Phase timeline:
        1. Infrastructure – `_init_pipeline`
           • Build the `UseCaseBundle` so every downstream step shares repositories,
             storage handlers, and processing use cases.
           • Provision object storage buckets and credentials via `_prepare_storage`.
           • Run optional audit / categorization passes to warm supporting datasets.

        2. Discovery – `_discover_documents`
           • Aggregate filesystem matches (`_discover_file_documents`).
           • Execute database connectors and capture samples (`_discover_database_documents`).
           • Run configured scrapers to turn crawled pages into sources (`_discover_web_documents`).
           • Document external APIs into JSON payloads (`_document_external_apis`).
           • Apply any input limits (for example `spec.limits.max_files`).

        3. Processing – `_process_documents`
           • Extract raw content using Docling/Tika (`_extract_content`).
           • Normalize text tokens and structure (`_normalize_content`).
           • Enrich metadata (keywords, entities, authors) (`_enrich_content`).
           • Validate content quality / scores (`_validate_content`).
           • Persist normalized payloads in object storage and the in-memory repository.

        4. Post-processing – `_maybe_chunk_documents`
           • Load chunking preset/overrides and attach run metadata.
           • Execute semantic chunking against processed document IDs.
           • Capture chunk counts and append recoverable errors without failing the run.

    Example:
        >>> spec = PipelineSpec.from_file("rag_pipeline.yaml")
        >>> summary = await run_pipeline(spec)
        >>> summary.counters["chunking"]
        {'documents': 12, 'total_chunks': 220, 'dry_run': False, 'enabled': True, 'preset': 'balanced'}
        The narrative shows the end-to-end orchestration with chunking metrics
        surfaced in the returned `RunSummary`.
    """

    use_cases = await _init_pipeline(spec)
    run_id = datetime.now(timezone.utc).isoformat()

    documents, errors = await _discover_documents(spec, use_cases)
    discovery_error_count = len(errors)

    client = _storage_client(spec)
    stored_objects, origin_counts, skipped, processed_document_ids = await _process_documents(
        documents=documents,
        spec=spec,
        use_cases=use_cases,
        client=client,
        run_id=run_id,
        errors=errors,
    )

    chunk_summary = await _maybe_chunk_documents(
        processed_document_ids=processed_document_ids,
        use_cases=use_cases,
        spec=spec,
        run_id=run_id,
        dry_run=dry_run,
        errors=errors,
    )

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

    return RunSummary(
        processed=len(stored_objects),
        skipped=skipped,
        failed=processing_failures,
        stored=stored_objects,
        errors=errors,
        counters=counters,
    )
