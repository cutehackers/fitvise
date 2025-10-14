"""RAG document processing API endpoints (Tasks 1.3.*)."""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

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


router = APIRouter(prefix="/rag/processing", tags=["RAG Processing"])


def get_pdf_use_case() -> ProcessPdfsUseCase:
    return ProcessPdfsUseCase()


def get_normalize_use_case() -> NormalizeTextUseCase:
    return NormalizeTextUseCase()


def get_metadata_use_case() -> ExtractMetadataUseCase:
    return ExtractMetadataUseCase()


def get_quality_use_case() -> ValidateQualityUseCase:
    return ValidateQualityUseCase()


class PdfDocumentPayload(BaseModel):
    file_path: Optional[str] = Field(None, description="Path on server to PDF")
    file_name: Optional[str] = Field(None, description="Name to use for inline content")
    content_base64: Optional[str] = Field(None, description="Base64 inline PDF content")

    @validator("file_name", always=True)
    def validate_inputs(cls, v, values):  # type: ignore[override]
        file_path = values.get("file_path")
        content_base64 = values.get("content_base64")
        if not file_path and not content_base64:
            raise ValueError("Either file_path or content_base64 must be provided")
        if content_base64 and not v:
            raise ValueError("file_name is required when content_base64 is provided")
        return v


class PdfProcessPayload(BaseModel):
    documents: List[PdfDocumentPayload]
    preserve_layout: bool = True
    extract_tables: bool = True


class PdfProcessResponseModel(BaseModel):
    documents: List[Dict[str, Any]]
    environment: Dict[str, Any]


@router.post("/pdf", response_model=PdfProcessResponseModel)
async def process_pdfs(
    payload: PdfProcessPayload, use_case: ProcessPdfsUseCase = Depends(get_pdf_use_case)
):
    try:
        file_paths: List[str] = []
        raw_docs: List[Any] = []
        for doc in payload.documents:
            if doc.file_path:
                file_paths.append(doc.file_path)
            if doc.content_base64:
                raw_docs.append((base64.b64decode(doc.content_base64), doc.file_name or "document.pdf"))
        request = ProcessPdfsRequest(
            file_paths=file_paths or None,
            raw_pdfs=raw_docs or None,
            preserve_layout=payload.preserve_layout,
            extract_tables=payload.extract_tables,
        )
        response = await use_case.execute(request)
        return PdfProcessResponseModel(
            documents=[d.as_dict() for d in response.documents], environment=response.environment
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class NormalizeTextPayload(BaseModel):
    texts: List[str]
    lowercase: bool = False
    correct_typos: bool = False
    lemmatize: bool = True
    extract_entities: bool = True


class NormalizeTextResponseModel(BaseModel):
    results: List[Dict[str, Any]]
    environment: Dict[str, Any]


@router.post("/normalize", response_model=NormalizeTextResponseModel)
async def normalize_text(
    payload: NormalizeTextPayload, use_case: NormalizeTextUseCase = Depends(get_normalize_use_case)
):
    try:
        request = NormalizeTextRequest(
            texts=payload.texts,
            lowercase=payload.lowercase,
            correct_typos=payload.correct_typos,
            lemmatize=payload.lemmatize,
            extract_entities=payload.extract_entities,
        )
        response = await use_case.execute(request)
        return NormalizeTextResponseModel(
            results=[r.as_dict() for r in response.results], environment=response.environment
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class ExtractMetadataPayload(BaseModel):
    texts: List[str]
    top_k_keywords: int = Field(10, ge=1, le=50)


class ExtractMetadataResponseModel(BaseModel):
    results: List[Dict[str, Any]]
    environment: Dict[str, Any]


@router.post("/metadata", response_model=ExtractMetadataResponseModel)
async def extract_metadata(
    payload: ExtractMetadataPayload, use_case: ExtractMetadataUseCase = Depends(get_metadata_use_case)
):
    try:
        request = ExtractMetadataRequest(texts=payload.texts, top_k_keywords=payload.top_k_keywords)
        response = await use_case.execute(request)
        return ExtractMetadataResponseModel(
            results=[r.as_dict() for r in response.results], environment=response.environment
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class ValidateQualityPayload(BaseModel):
    texts: List[str]
    min_overall_score: float = Field(0.5, ge=0.0, le=1.0)
    min_word_count: int = Field(10, ge=0)
    min_character_count: int = Field(100, ge=0)


class ValidateQualityResponseModel(BaseModel):
    reports: List[Dict[str, Any]]
    environment: Dict[str, Any]


@router.post("/quality", response_model=ValidateQualityResponseModel)
async def validate_quality(
    payload: ValidateQualityPayload, use_case: ValidateQualityUseCase = Depends(get_quality_use_case)
):
    try:
        thresholds = None
        if any(
            [
                payload.min_overall_score != 0.5,
                payload.min_word_count != 10,
                payload.min_character_count != 100,
            ]
        ):
            from app.domain.value_objects.quality_metrics import QualityThresholds

            thresholds = QualityThresholds(
                min_overall_score=payload.min_overall_score,
                min_word_count=payload.min_word_count,
                min_character_count=payload.min_character_count,
            )
        request = ValidateQualityRequest(texts=payload.texts, thresholds=thresholds)
        response = await use_case.execute(request)
        return ValidateQualityResponseModel(
            reports=[r.as_dict() for r in response.reports], environment=response.environment
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
