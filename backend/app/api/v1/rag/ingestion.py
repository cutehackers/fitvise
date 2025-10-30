"""RAG ingestion infrastructure API endpoints."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator

from app.application.use_cases.data_ingestion import (
    SetupAirflowUseCase,
    SetupAirflowRequest,
    IntegrateTikaUseCase,
    IntegrateTikaRequest,
    ConnectDatabasesUseCase,
    ConnectDatabasesRequest,
    SetupWebScrapingUseCase,
    SetupWebScrapingRequest,
    DatabaseConnectorSpec,
)
from app.infrastructure.external_services.data_sources.database_connectors import (  # type: ignore
    DatabaseConnectionConfig,
)

router = APIRouter(prefix="/rag/ingestion", tags=["RAG Ingestion"])


# ---------------------------------------------------------------------------
# Dependency factories (provide override hooks for tests)
# ---------------------------------------------------------------------------

def get_airflow_use_case() -> SetupAirflowUseCase:
    return SetupAirflowUseCase()


def get_tika_use_case() -> IntegrateTikaUseCase:
    return IntegrateTikaUseCase()


def get_db_use_case() -> ConnectDatabasesUseCase:
    return ConnectDatabasesUseCase()


def get_scraping_use_case() -> SetupWebScrapingUseCase:
    return SetupWebScrapingUseCase()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class AirflowSetupPayload(BaseModel):
    base_path: Optional[str] = Field(None, description="Base directory for Airflow artefacts")
    env_overrides: Optional[Dict[str, str]] = Field(None, description="ENV overrides for Airflow")
    compose_overrides: Optional[Dict[str, Any]] = Field(
        None, description="docker-compose overrides merged on top of defaults"
    )
    dag_id: str = Field("rag_hello_world", description="Identifier for the generated DAG")
    schedule: str = Field("@daily", description="Schedule interval for the DAG")
    tags: Optional[List[str]] = Field(None, description="Tags to apply to the DAG")


class AirflowSetupResponseModel(BaseModel):
    env_file: str
    docker_compose_file: str
    dag_file: str
    created_directories: List[str]
    diagnostics: Dict[str, Any]


class TikaDocumentPayload(BaseModel):
    file_path: Optional[str] = Field(None, description="Path available on the server to parse")
    file_name: Optional[str] = Field(None, description="File name to use for raw content parsing")
    content_base64: Optional[str] = Field(None, description="Base64 encoded content for inline docs")

    @validator("file_name", always=True)
    def validate_inputs(cls, v, values):  # type: ignore[override]
        file_path = values.get("file_path")
        content_base64 = values.get("content_base64")
        if not file_path and not content_base64:
            raise ValueError("Either file_path or content_base64 must be provided")
        if content_base64 and not v:
            raise ValueError("file_name is required when content_base64 is provided")
        return v


class TikaExtractionPayload(BaseModel):
    documents: List[TikaDocumentPayload]


class TikaExtractionResponse(BaseModel):
    success_count: int
    failure_count: int
    documents: List[Dict[str, Any]]
    environment_status: Dict[str, Any]


class DatabaseConnectorConfigModel(BaseModel):
    name: str
    connector_type: str = Field(..., description="postgres|mysql|mongo")
    driver: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    use_ssl: bool = False
    sample_collection: Optional[str] = None
    sample_limit: int = Field(5, ge=1, le=100)


class DatabaseConnectorResponseModel(BaseModel):
    connector_type: str
    config_name: str
    connection_result: Dict[str, Any]
    collections: Optional[List[str]] = None
    sample: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DatabaseConnectorPayload(BaseModel):
    connectors: List[DatabaseConnectorConfigModel]
    fetch_samples: bool = True


class DatabaseConnectorResponse(BaseModel):
    results: List[DatabaseConnectorResponseModel]


class WebScrapingPayload(BaseModel):
    start_urls: List[str]
    allowed_domains: Optional[List[str]] = None
    max_depth: int = Field(2, ge=0, le=5)
    max_pages: int = Field(10, ge=1, le=200)
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    follow_css_selectors: Optional[List[str]] = None
    follow_xpath: Optional[List[str]] = None
    authentication: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None
    follow_external_links: bool = False


class WebScrapingResponseModel(BaseModel):
    page_count: int
    elapsed_seconds: float
    pages: List[str]
    errors: List[str]
    warnings: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/airflow/setup", response_model=AirflowSetupResponseModel, status_code=status.HTTP_201_CREATED)
async def create_airflow_environment(
    payload: AirflowSetupPayload,
    use_case: SetupAirflowUseCase = Depends(get_airflow_use_case),
):
    """Generate Airflow support artefacts (env file, docker-compose and hello-world DAG)."""
    try:
        request = SetupAirflowRequest(
            base_path=payload.base_path,
            env_overrides=payload.env_overrides,
            compose_overrides=payload.compose_overrides,
            dag_id=payload.dag_id,
            schedule=payload.schedule,
            tags=payload.tags,
        )
        response = await use_case.execute(request)
        return AirflowSetupResponseModel(
            env_file=str(response.env_file),
            docker_compose_file=str(response.docker_compose_file),
            dag_file=str(response.dag_file),
            created_directories=[str(p) for p in response.created_directories],
            diagnostics=response.diagnostics,
        )
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - general safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/tika/extract", response_model=TikaExtractionResponse)
async def extract_documents_via_tika(
    payload: TikaExtractionPayload,
    use_case: IntegrateTikaUseCase = Depends(get_tika_use_case),
):
    """Run Apache Tika (or fallbacks) across supplied documents."""
    try:
        file_paths: List[str] = []
        raw_documents: List[Any] = []
        for doc in payload.documents:
            if doc.file_path:
                file_paths.append(doc.file_path)
            if doc.content_base64:
                content_bytes = base64.b64decode(doc.content_base64)
                raw_documents.append((content_bytes, doc.file_name or "document"))

        request = IntegrateTikaRequest(
            file_paths=file_paths or None,
            raw_documents=raw_documents or None,
        )
        response = await use_case.execute(request)
        return TikaExtractionResponse(
            success_count=response.success_count,
            failure_count=response.failure_count,
            documents=[doc.as_dict() for doc in response.documents],
            environment_status=response.environment_status,
        )
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/databases/connect", response_model=DatabaseConnectorResponse)
async def connect_to_databases(
    payload: DatabaseConnectorPayload,
    use_case: ConnectDatabasesUseCase = Depends(get_db_use_case),
):
    """Test database connectivity and fetch optional samples using configured connectors."""
    try:
        specs: List[DatabaseConnectorSpec] = []
        for connector in payload.connectors:
            config = DatabaseConnectionConfig(
                name=connector.name,
                driver=connector.driver,
                host=connector.host,
                port=connector.port,
                username=connector.username,
                password=connector.password,
                database=connector.database,
                schema=connector.schema,
                params=connector.params or {},
                use_ssl=connector.use_ssl,
            )
            specs.append(
                DatabaseConnectorSpec(
                    connector_type=connector.connector_type,
                    config=config,
                    sample_collection=connector.sample_collection,
                    sample_limit=connector.sample_limit,
                )
            )

        request = ConnectDatabasesRequest(connectors=specs, fetch_samples=payload.fetch_samples)
        response = await use_case.execute(request)
        return DatabaseConnectorResponse(
            results=[DatabaseConnectorResponseModel(**result.as_dict()) for result in response.results]
        )
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/web-scraping/run", response_model=WebScrapingResponseModel)
async def run_web_scraping_job(
    payload: WebScrapingPayload,
    use_case: SetupWebScrapingUseCase = Depends(get_scraping_use_case),
):
    """Kick off a scrape against supplied URLs using the framework stack."""
    try:
        request = SetupWebScrapingRequest(
            start_urls=payload.start_urls,
            allowed_domains=payload.allowed_domains,
            max_depth=payload.max_depth,
            max_pages=payload.max_pages,
            include_patterns=payload.include_patterns,
            exclude_patterns=payload.exclude_patterns,
            follow_css_selectors=payload.follow_css_selectors,
            follow_xpath=payload.follow_xpath,
            authentication=payload.authentication,
            headers=payload.headers,
            cookies=payload.cookies,
            follow_external_links=payload.follow_external_links,
        )
        response = await use_case.execute(request)
        return WebScrapingResponseModel(
            page_count=response.result.page_count,
            elapsed_seconds=response.result.elapsed_seconds,
            pages=response.pages,
            errors=response.errors,
            warnings=response.result.warnings,
        )
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
