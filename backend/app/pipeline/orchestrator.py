from __future__ import annotations

"""Unified orchestrator that coordinates ingestion use cases for the RAG pipeline."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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


logger = logging.getLogger(__name__)


@dataclass
class UseCaseBundle:
    repository: InMemoryDataSourceRepository
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


def _build_use_cases() -> UseCaseBundle:
    repository = InMemoryDataSourceRepository()
    return UseCaseBundle(
        repository=repository,
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
    )


def _guess_content_type(path: Path) -> Optional[str]:
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


def _discover_file_documents(spec: PipelineSpec) -> List[DocumentSource]:
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
    return hashlib.sha256(payload).hexdigest()


async def run_pipeline(spec: PipelineSpec) -> RunSummary:
    """
    Main data ingestion orchestration pipeline.
    
    Coordinates the complete RAG data ingestion workflow by:
    1. Preparing object storage infrastructure
    2. Auditing and categorizing data sources (optional)
    3. Discovering documents from multiple origins (files, databases, web, APIs)
    4. Extracting content using appropriate processors (Docling for PDFs, Tika for others)
    5. Normalizing text and enriching with metadata (keywords, entities, dates)
    6. Validating content quality and computing quality scores
    7. Storing processed documents in object storage with metadata and tags
    
    Pipeline Execution Sequence:
        Phase 1 - Infrastructure Setup:
            → _build_use_cases(): Initialize all use case handlers
            → _prepare_storage(): Setup object storage (MinIO/local)
            → _maybe_audit_sources(): Optional source auditing
            → _maybe_categorize_sources(): Optional ML-based categorization
        
        Phase 2 - Document Discovery:
            → _discover_file_documents(): Scan filesystem paths
            → _discover_database_documents(): Query databases (Postgres, MongoDB, etc.)
            → _discover_web_documents(): Scrape web pages
            → _document_external_apis(): Document external APIs
            → Apply max_files limit if configured
        
        Phase 3 - Per-Document Processing Loop:
            For each discovered document:
                → _extract_content(): Extract text (Docling for PDFs, Tika for others)
                → _normalize_content(): Normalize, tokenize, extract entities
                → _enrich_content(): Extract keywords, dates, authors
                → _validate_content(): Compute quality scores and validations
                → _content_hash(): Generate SHA256 hash for deduplication
                → client.put_object(): Store in object storage with metadata/tags
        
        Phase 4 - Summary Generation:
            → Aggregate counters (discovered, processed, skipped, failed)
            → Return RunSummary with statistics and stored objects
    
    Args:
        spec: Pipeline configuration specifying sources, storage, and processing options
        
    Returns:
        RunSummary containing processing statistics, stored objects, and any errors
        
    Example:
        >>> from pathlib import Path
        >>> from app.pipeline.config import PipelineSpec
        >>> 
        >>> # Load configuration from YAML
        >>> spec = PipelineSpec.from_file("rag_pipeline.yaml")
        >>> 
        >>> # Execute pipeline
        >>> summary = await run_pipeline(spec)
        >>> 
        >>> # Check results
        >>> print(f"Processed: {summary.processed}")
        >>> print(f"Skipped: {summary.skipped}")
        >>> print(f"Failed: {summary.failed}")
        >>> print(f"By origin: {summary.counters['by_origin']}")
        >>> # Output: {'file': 45, 'database': 12, 'web': 8, 'api': 3}
        >>> 
        >>> # Access stored objects
        >>> for obj in summary.stored:
        ...     print(f"{obj.bucket}/{obj.key} - {obj.size} bytes")
        ...     print(f"  Quality: {obj.metadata['quality_level']}")
        ...     print(f"  Keywords: {obj.metadata['keywords']}")
    """
    use_cases = _build_use_cases()
    await _prepare_storage(spec, use_cases)
    await _maybe_audit_sources(spec, use_cases)
    await _maybe_categorize_sources(spec, use_cases)

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

    client = _storage_client(spec)
    stored_objects: List[StorageObject] = []
    origin_counts: Dict[str, int] = {}
    skipped = 0
    discovery_error_count = len(errors)

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

    processing_failures = max(len(errors) - discovery_error_count, 0)
    counters = {
        "discovered": len(documents),
        "processed": len(stored_objects),
        "skipped": skipped,
        "failed": processing_failures,
        "by_origin": origin_counts,
        "discovery_errors": discovery_error_count,
        "total_errors": len(errors),
    }
    return RunSummary(
        processed=len(stored_objects),
        skipped=skipped,
        failed=processing_failures,
        stored=stored_objects,
        errors=errors,
        counters=counters,
    )
