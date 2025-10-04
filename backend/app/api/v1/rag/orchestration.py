"""RAG orchestration API endpoints for building ETL DAGs (Task 1.4.2)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.application.use_cases.storage_management import (
    BuildEtlDagsUseCase,
    BuildEtlDagsRequest,
    BuildEtlDagsResponse,
)


router = APIRouter(prefix="/rag/orchestration", tags=["RAG Orchestration"])


def get_dag_use_case() -> BuildEtlDagsUseCase:
    return BuildEtlDagsUseCase()


class BuildDagsPayload(BaseModel):
    base_path: Optional[str] = Field(None, description="Base directory for Airflow artefacts")
    ingestion_dag_id: str = Field("rag_data_ingestion", description="Ingestion DAG ID")
    processing_dag_id: str = Field("rag_document_processing", description="Processing DAG ID")
    quality_dag_id: str = Field("rag_data_quality", description="Quality DAG ID")


class BuildDagsResponseModel(BaseModel):
    dag_files: List[str]
    diagnostics: Dict[str, Any]


@router.post("/build-dags", response_model=BuildDagsResponseModel)
async def build_etl_dags(
    payload: BuildDagsPayload,
    use_case: BuildEtlDagsUseCase = Depends(get_dag_use_case),
):
    try:
        request = BuildEtlDagsRequest(
            base_path=payload.base_path,
            ingestion_dag_id=payload.ingestion_dag_id,
            processing_dag_id=payload.processing_dag_id,
            quality_dag_id=payload.quality_dag_id,
        )
        result: BuildEtlDagsResponse = await use_case.execute(request)
        return BuildDagsResponseModel(dag_files=result.dag_files, diagnostics=result.diagnostics)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

