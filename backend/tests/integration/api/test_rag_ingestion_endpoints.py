"""Integration tests for RAG ingestion API endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.rag import ingestion
from app.application.use_cases.data_ingestion.setup_airflow import (
    SetupAirflowResponse,
)
from app.application.use_cases.data_ingestion.integrate_tika import (
    IntegrateTikaResponse,
    TikaDocumentResult,
)
from app.application.use_cases.data_ingestion.connect_databases import (
    ConnectDatabasesResponse,
    DatabaseConnectorResult,
)
from app.application.use_cases.data_ingestion.setup_web_scraping import (
    SetupWebScrapingResponse,
)
from app.infrastructure.orchestration.airflow_manager import AirflowSetupReport
from app.infrastructure.external_services.data_sources.file_processors.tika_processor import (
    TikaExtractionResult,
)
from app.infrastructure.external_services.data_sources.database_connectors.base import (
    ConnectionTestResult,
    QueryResult,
)
from app.infrastructure.external_services.data_sources.web_scrapers.scrapy_framework import (
    CrawlResult,
    CrawledPage,
)

API_PREFIX = "/api/v1"
client = TestClient(app)


def teardown_module(module):  # noqa: D401 - ensure dependency overrides are cleared
    """Reset dependency overrides after tests complete."""
    app.dependency_overrides.clear()


class _StubAirflowUseCase:
    async def execute(self, request):  # type: ignore[override]
        report = AirflowSetupReport(
            env_file=Path("/tmp/.env.airflow"),
            docker_compose_file=Path("/tmp/docker-compose-airflow.yml"),
            dag_file=Path("/tmp/dags/rag_hello_world.py"),
            created_directories=[Path("/tmp/dags"), Path("/tmp/logs")],
        )
        diagnostics: Dict[str, Any] = {"env_file_exists": True}
        return SetupAirflowResponse(success=True, report=report, diagnostics=diagnostics)


class _StubTikaUseCase:
    async def execute(self, request):  # type: ignore[override]
        extraction = TikaExtractionResult(
            text="Sample content",
            metadata={"parser": "stub"},
            content_type="text/plain",
            language="en",
            success=True,
        )
        doc_result = TikaDocumentResult(source="stub.txt", result=extraction)
        environment = {"tika_available": False}
        return IntegrateTikaResponse(documents=[doc_result], environment_status=environment)


class _StubDatabaseUseCase:
    async def execute(self, request):  # type: ignore[override]
        connection_result = ConnectionTestResult(success=True, latency_ms=12.5, server_info={"version": "stub"})
        sample = QueryResult(rows=[{"id": 1, "value": "demo"}], row_count=1, columns=["id", "value"], query="SELECT *")
        result = DatabaseConnectorResult(
            connector_type="postgres",
            config_name="primary",
            connection_result=connection_result,
            collections=["public.demo"],
            sample=sample,
            error=None,
        )
        return ConnectDatabasesResponse(results=[result])


class _StubScrapingUseCase:
    async def execute(self, request):  # type: ignore[override]
        pages: List[CrawledPage] = [
            CrawledPage(url="https://example.com", status_code=200, content="<html></html>", extracted_links=[])
        ]
        result = CrawlResult(pages=pages, elapsed_seconds=0.5, errors=[], warnings=["stub"])
        return SetupWebScrapingResponse(result=result)


def test_create_airflow_environment_endpoint():
    app.dependency_overrides[ingestion.get_airflow_use_case] = lambda: _StubAirflowUseCase()

    response = client.post(
        f"{API_PREFIX}/rag/ingestion/airflow/setup",
        json={"base_path": "/tmp/rag-airflow"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["env_file"].endswith(".env.airflow")
    assert data["diagnostics"]["env_file_exists"] is True


def test_tika_extraction_endpoint():
    app.dependency_overrides[ingestion.get_tika_use_case] = lambda: _StubTikaUseCase()

    response = client.post(
        f"{API_PREFIX}/rag/ingestion/tika/extract",
        json={"documents": [{"file_path": "/tmp/doc.txt"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success_count"] == 1
    assert data["documents"][0]["source"] == "stub.txt"


def test_database_connector_endpoint():
    app.dependency_overrides[ingestion.get_db_use_case] = lambda: _StubDatabaseUseCase()

    response = client.post(
        f"{API_PREFIX}/rag/ingestion/databases/connect",
        json={
            "fetch_samples": True,
            "connectors": [
                {
                    "name": "primary",
                    "connector_type": "postgres",
                    "driver": "postgresql+psycopg2",
                    "host": "localhost",
                    "port": 5432,
                    "database": "demo",
                    "schema": "public",
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["connection_result"]["success"] is True


def test_web_scraping_endpoint():
    app.dependency_overrides[ingestion.get_scraping_use_case] = lambda: _StubScrapingUseCase()

    response = client.post(
        f"{API_PREFIX}/rag/ingestion/web-scraping/run",
        json={"start_urls": ["https://example.com"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["page_count"] == 1
    assert data["warnings"] == ["stub"]
