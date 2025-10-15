from __future__ import annotations

"""
Single orchestrator for the RAG document processing pipeline (Phase 1).

Responsibilities
- Discover input documents
- Extract content using pluggable processors with safe fallbacks
- Normalize text
- Enrich metadata
- Validate quality
- Store outputs to object storage with idempotent key scheme
- Return run summary with counters and errors

Notes
- Plugin registry and advanced dedupe manifesting are planned for Phase 2/3.
- This Phase 1 orchestrator focuses on a clean, single entrypoint and config.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.pipeline.config import PipelineSpec
from app.pipeline.contracts import (
    DocumentSource,
    SourceContent,
    StandardSourceContent,
    MetadataAnnotation,
    QualityReport,
    StorageObject,
    RunSummary,
)
from app.core.compat import has_module
from app.infrastructure.external_services.data_sources.file_processors import (
    DoclingPdfProcessor,
    SpacyTextProcessor,
    NormalizeTextOptions,
    TikaIntegrationService,
    TikaClientConfig,
)
from app.application.use_cases.document_processing.validate_quality import ValidateQualityUseCase, ValidateQualityRequest
from app.infrastructure.storage.object_storage.minio_client import ObjectStorageClient, ObjectStorageConfig


logger = logging.getLogger(__name__)


@dataclass
class _StageTimers:
    discover_ms: float = 0.0
    extract_ms: float = 0.0
    normalize_ms: float = 0.0
    enrich_ms: float = 0.0
    validate_ms: float = 0.0
    store_ms: float = 0.0


def _iter_discover_documents(cfg: PipelineSpec) -> Iterable[DocumentSource]:
    root = cfg.inputs.path
    patterns = cfg.inputs.include or ["*.pdf"]
    paths: List[Path] = []
    for pattern in patterns:
        if cfg.inputs.recurse:
            paths.extend(root.rglob(pattern))
        else:
            paths.extend(root.glob(pattern))
    # Limits
    if cfg.limits.max_files is not None:
        paths = paths[: cfg.limits.max_files]

    for p in paths:
        yield DocumentSource(
            source_id=None,
            uri=None,
            path=str(p),
            content_bytes=None,
            content_type=_guess_content_type(p),
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


async def _extract(di: DocumentSource, cfg: PipelineSpec) -> SourceContent:
    path = Path(di.path or "")
    content_type = di.content_type or _guess_content_type(path)
    warnings: List[str] = []
    text: str = ""
    markdown: Optional[str] = None
    tables: List[Dict[str, object]] = []
    language: Optional[str] = None
    metadata: Dict[str, object] = {"source": path.name}

    if content_type == "application/pdf":
        processor = DoclingPdfProcessor()
        res = processor.process_pdf_from_path(path)
        if not res.success:
            warnings.append(res.error or "pdf processing failed")
        text = res.text
        markdown = res.markdown or (res.text or "")
        tables = res.tables
        language = res.language
        metadata.update(res.metadata)
        warnings.extend(res.warnings)
    else:
        # Non-PDF: prefer Tika when available, fallback to plain text
        tika = TikaIntegrationService(TikaClientConfig()) if has_module("tika") else None
        if tika is not None:
            res = await tika.extract_from_path(path)
            if not res.success:
                warnings.append(res.error or "tika extract failed")
            text = res.text or ""
            markdown = None
            tables = []
            language = res.language
            if res.metadata:
                metadata.update(res.metadata)
            warnings.extend(res.warnings)
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                warnings.append(str(exc))
                text = ""
            markdown = None
            tables = []
            language = None

    return SourceContent(
        text=text,
        markdown=markdown,
        tables=tables,
        metadata=metadata,
        content_type=content_type,
        language=language,
        warnings=warnings,
    )


async def _normalize(text: str) -> StandardSourceContent:
    spacy = SpacyTextProcessor(None)
    res = spacy.normalize_text(text, NormalizeTextOptions(lowercase=False, correct_typos=False, lemmatize=True))
    return StandardSourceContent(
        normalized=res.normalized,
        tokens=res.tokens,
        lemmas=res.lemmas,
        entities=res.entities,
        warnings=res.warnings,
    )


async def _enrich(text: str) -> MetadataAnnotation:
    spacy = SpacyTextProcessor(None)
    results = spacy.extract([text], top_k_keywords=10)
    pr = results[0] if results else None
    if pr is None:
        return MetadataAnnotation(keywords=[], entities=[], dates=[], authors=[], language=None)
    return MetadataAnnotation(
        keywords=pr.keywords,
        entities=pr.entities,
        dates=pr.dates,
        authors=pr.authors,
        language=pr.language,
        warnings=pr.warnings,
    )


async def _validate(text: str) -> QualityReport:
    uc = ValidateQualityUseCase()
    res = await uc.execute(ValidateQualityRequest(texts=[text]))
    qr = res.reports[0]
    return QualityReport(
        overall_score=qr.overall_score,
        quality_level=qr.quality_level,
        metrics=qr.metrics,
        validations=qr.validations,
        ge_report=qr.ge_report,
    )


def _content_hash(payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def _storage_client(cfg: PipelineSpec) -> ObjectStorageClient:
    client = ObjectStorageClient(
        ObjectStorageConfig(
            provider=cfg.storage.provider,
            endpoint=cfg.storage.endpoint,
            base_dir=cfg.storage.base_dir,
        )
    )
    client.ensure_bucket(cfg.storage.bucket)
    return client


async def run_pipeline(config: PipelineSpec) -> RunSummary:
    client = _storage_client(config)
    stored: List[StorageObject] = []
    errors: List[Dict[str, object]] = []
    processed = 0
    skipped = 0
    failed = 0

    # Discover
    docs = list(_iter_discover_documents(config))
    for doc in docs:
        try:
            # Extract
            extracted = await _extract(doc, config)
            base_text = extracted.markdown or extracted.text or ""

            # Normalize
            norm = await _normalize(base_text)

            # Enrich
            enriched = await _enrich(norm.normalized)

            # Validate
            quality = await _validate(norm.normalized)

            # Storage key (YYYY/MM/DD/{hash}.md)
            now = datetime.now(timezone.utc)
            dated = now.strftime("%Y/%m/%d")
            content_bytes = norm.normalized.encode("utf-8")
            c_hash = _content_hash(content_bytes)[:16]
            object_key = f"{dated}/{c_hash}.md"

            # Tags & metadata
            tags = {
                "validated": "true" if quality.overall_score >= 0.5 else "false",
                "processor_pdf": config.processors.pdf,
                "processor_non_pdf": config.processors.non_pdf,
                "normalize": config.processors.normalize,
            }
            metadata = {
                "source": (Path(doc.path).name if doc.path else doc.uri or "unknown"),
                "keywords": ",".join(enriched.keywords[:10]) if enriched.keywords else "",
                "overall_score": str(quality.overall_score),
                "quality_level": quality.quality_level,
            }

            # Store
            result = client.put_object(
                config.storage.bucket,
                object_key,
                content_bytes,
                content_type="text/markdown",
                metadata=metadata,
                tags=tags,
            )
            stored.append(
                StorageObject(
                    bucket=result.bucket,
                    key=result.key,
                    size=result.size,
                    metadata={k: str(v) for k, v in metadata.items()},
                    tags=tags,
                )
            )
            processed += 1
            logger.info(
                json.dumps(
                    {
                        "event": "doc_processed",
                        "path": doc.path,
                        "key": result.key,
                        "score": quality.overall_score,
                        "level": quality.quality_level,
                    }
                )
            )
        except Exception as exc:  # pragma: no cover
            failed += 1
            err = {"path": doc.path, "error": str(exc)}
            errors.append(err)
            logger.error(json.dumps({"event": "doc_failed", **err}))
            continue

    counters = {
        "discovered": len(docs),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }
    return RunSummary(
        processed=processed,
        skipped=skipped,
        failed=failed,
        stored=stored,
        errors=errors,
        counters=counters,
    )
