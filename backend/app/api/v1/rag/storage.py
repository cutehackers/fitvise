"""RAG object storage API endpoints (Task 1.4.1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.application.use_cases.storage_management import (
    SetupObjectStorageRequest,
    SetupObjectStorageResponse,
    SetupObjectStorageUseCase,
)


router = APIRouter(prefix="/rag/storage", tags=["RAG Storage"])


def get_storage_use_case() -> SetupObjectStorageUseCase:
    return SetupObjectStorageUseCase()


class ObjectStorageSetupPayload(BaseModel):
    provider: str = Field("local", description="'local' or 'minio'")
    endpoint: Optional[str] = Field(None, description="MinIO endpoint, e.g. 'localhost:9000'")
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    secure: bool = False
    base_dir: Optional[str] = Field(None, description="Local base dir for buckets (dev only)")
    buckets: Optional[List[str]] = None
    test_object: bool = True


class ObjectStorageSetupResponse(BaseModel):
    provider: str
    created_buckets: List[str]
    environment: Dict[str, Any]
    test_put_key: Optional[str]


@router.post("/setup", response_model=ObjectStorageSetupResponse)
async def setup_object_storage(
    payload: ObjectStorageSetupPayload,
    use_case: SetupObjectStorageUseCase = Depends(get_storage_use_case),
):
    try:
        request = SetupObjectStorageRequest(
            provider=payload.provider,
            endpoint=payload.endpoint,
            access_key=payload.access_key,
            secret_key=payload.secret_key,
            secure=payload.secure,
            base_dir=payload.base_dir,
            buckets=payload.buckets,
            test_object=payload.test_object,
        )
        response: SetupObjectStorageResponse = await use_case.execute(request)
        return ObjectStorageSetupResponse(
            provider=response.provider,
            created_buckets=response.created_buckets,
            environment=response.environment,
            test_put_key=response.test_put_key,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

