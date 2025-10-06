"""Simulate the Phase 1 pipeline end-to-end to storage (Task 1.4 demo)."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from app.application.use_cases.document_processing import (
    ProcessPdfsUseCase,
    ProcessPdfsRequest,
    CleanTextUseCase,
    CleanTextRequest,
    ExtractMetadataUseCase,
    ExtractMetadataRequest,
    ValidateQualityUseCase,
    ValidateQualityRequest,
)
from app.application.use_cases.storage_management import SetupObjectStorageUseCase, SetupObjectStorageRequest
from app.infrastructure.storage.object_storage.minio_client import ObjectStorageClient, ObjectStorageConfig


router = APIRouter(prefix="/rag/pipeline", tags=["RAG Pipeline"])


def get_use_cases():
    return (
        ProcessPdfsUseCase(),
        CleanTextUseCase(),
        ExtractMetadataUseCase(),
        ValidateQualityUseCase(),
        SetupObjectStorageUseCase(),
    )


class PipelineDoc(BaseModel):
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    content_base64: Optional[str] = None

    @validator("file_name", always=True)
    def _validate(cls, v, values):  # type: ignore[override]
        if not values.get("file_path") and not values.get("content_base64"):
            raise ValueError("Either file_path or content_base64 must be provided")
        if values.get("content_base64") and not v:
            raise ValueError("file_name is required when content_base64 is provided")
        return v


class RunPipelinePayload(BaseModel):
    documents: List[PipelineDoc]
    storage_provider: str = Field("local", description="'local' or 'minio'")
    storage_base_dir: Optional[str] = Field(None, description="Local base dir for buckets (dev only)")
    storage_endpoint: Optional[str] = Field(None, description="MinIO endpoint host:port")
    storage_access_key: Optional[str] = None
    storage_secret_key: Optional[str] = None
    storage_secure: bool = False
    bucket_processed: str = Field("rag-processed", description="Bucket used for processed markdown")


class RunPipelineResponse(BaseModel):
    processed: int
    stored_objects: List[str]
    steps: Dict[str, Any]
    storage: Dict[str, Any]


@router.post(
    "/run",
    response_model=RunPipelineResponse,
    summary="Run Phase 1 pipeline",
    description="Process documents end-to-end and store results to object storage.",
)
async def run_pipeline(
    payload: RunPipelinePayload,
):
    try:
        (
            pdf_uc,
            clean_uc,
            meta_uc,
            quality_uc,
            storage_uc,
        ) = get_use_cases()

        # 1) Process PDFs -> markdown
        file_paths: List[str] = []
        raw_docs: List[Any] = []
        for doc in payload.documents:
            if doc.file_path:
                file_paths.append(doc.file_path)
            if doc.content_base64:
                raw_docs.append((base64.b64decode(doc.content_base64), doc.file_name or "document.pdf"))

        pdf_req = ProcessPdfsRequest(file_paths=file_paths or None, raw_pdfs=raw_docs or None)
        pdf_res = await pdf_uc.execute(pdf_req)
        if not pdf_res.documents:
            raise ValueError("No documents processed")

        first_doc = pdf_res.documents[0]
        markdown_text = first_doc.markdown

        # 2) Clean text
        clean_req = CleanTextRequest(texts=[markdown_text])
        clean_res = await clean_uc.execute(clean_req)
        cleaned_text = clean_res.results[0].cleaned_text if clean_res.results else markdown_text

        # 3) Extract metadata
        meta_req = ExtractMetadataRequest(texts=[cleaned_text], top_k_keywords=10)
        meta_res = await meta_uc.execute(meta_req)
        keywords = meta_res.results[0].keywords if meta_res.results else []

        # 4) Validate quality
        qual_req = ValidateQualityRequest(texts=[cleaned_text], thresholds=None)
        qual_res = await quality_uc.execute(qual_req)
        if qual_res.reports:
            qr = qual_res.reports[0]
            quality_report = {
                "overall_score": qr.overall_score,
                "quality_level": qr.quality_level,
                "metrics": qr.metrics,
                "validations": qr.validations,
            }
            quality_ok = bool(qr.overall_score >= 0.5)
        else:
            quality_report = {}
            quality_ok = False

        # 5) Setup storage and store result
        setup_req = SetupObjectStorageRequest(
            provider=payload.storage_provider,
            endpoint=payload.storage_endpoint,
            access_key=payload.storage_access_key,
            secret_key=payload.storage_secret_key,
            secure=payload.storage_secure,
            base_dir=payload.storage_base_dir,
            test_object=False,
        )
        await storage_uc.execute(setup_req)

        client = ObjectStorageClient(
            ObjectStorageConfig(
                provider=payload.storage_provider,
                endpoint=payload.storage_endpoint,
                access_key=payload.storage_access_key,
                secret_key=payload.storage_secret_key,
                secure=payload.storage_secure,
                base_dir=Path(payload.storage_base_dir).resolve() if payload.storage_base_dir else None,
            )
        )
        client.ensure_bucket(payload.bucket_processed)

        # Compose object key and metadata/tags
        now = datetime.now(timezone.utc)
        dated = now.strftime("%Y/%m/%d")
        filename = (first_doc.source or "document.pdf").split("/")[-1].rsplit(".", 1)[0] + ".md"
        object_key = f"{dated}/{filename}"
        tags = {
            "pipeline": "phase1",
            "validated": str(quality_ok).lower(),
            "task": "1.4",
        }
        metadata = {
            "source": first_doc.source,
            "keywords": ",".join(keywords[:10]) if keywords else "",
            "overall_quality": str(quality_report.get("overall_score", 0)),
        }
        result = client.put_object(
            payload.bucket_processed,
            object_key,
            cleaned_text.encode("utf-8"),
            content_type="text/markdown",
            metadata=metadata,
            tags=tags,
        )

        steps = {
            "pdf_processing": {
                "doc_count": len(pdf_res.documents),
                "warnings": first_doc.warnings,
            },
            "cleaning": {
                "performed": True,
            },
            "metadata_extraction": {
                "keywords": keywords[:10],
            },
            "quality_validation": quality_report,
        }
        storage_info = {
            "provider": payload.storage_provider,
            "bucket": payload.bucket_processed,
            "key": object_key,
        }
        return RunPipelineResponse(
            processed=len(pdf_res.documents),
            stored_objects=[f"{payload.bucket_processed}:{object_key}"],
            steps=steps,
            storage=storage_info,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
