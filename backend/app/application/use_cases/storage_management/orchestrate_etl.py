"""Use case for building ETL orchestration DAGs (Task 1.4.2).

Generates Airflow DAGs for data ingestion, document processing, and data
quality validation. Uses the existing AirflowEnvironmentManager to ensure
folder structure and write DAG files with daily schedules and basic
failure logging/notification hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
import string
from pathlib import Path
from typing import Dict, List, Optional

from app.infrastructure.orchestration import AirflowEnvironmentManager


def _render_template(template_filename: str, mapping: Dict[str, str]) -> str:
    templates_dir = Path(__file__).parent / "templates"
    tpl = templates_dir / template_filename
    text = tpl.read_text(encoding="utf-8")
    return string.Template(text).substitute(mapping)


def _ingestion_dag_source(dag_id: str = "rag_data_ingestion") -> str:
    return _render_template("ingestion_dag_tmpl.py", {"dag_id": dag_id, "schedule": "@daily", "tags": "['rag','etl','ingestion']"})


def _processing_dag_source(dag_id: str = "rag_document_processing") -> str:
    return _render_template("processing_dag_tmpl.py", {"dag_id": dag_id, "schedule": "@daily", "tags": "['rag','etl','processing']"})


def _quality_dag_source(dag_id: str = "rag_data_quality") -> str:
    return _render_template("quality_dag_tmpl.py", {"dag_id": dag_id, "schedule": "@daily", "tags": "['rag','etl','quality']"})


@dataclass
class BuildEtlDagsRequest:
    base_path: Optional[str] = None
    ingestion_dag_id: str = "rag_data_ingestion"
    processing_dag_id: str = "rag_document_processing"
    quality_dag_id: str = "rag_data_quality"


@dataclass
class BuildEtlDagsResponse:
    success: bool
    dag_files: List[str]
    diagnostics: Dict[str, str]


class BuildEtlDagsUseCase:
    def __init__(self, manager: Optional[AirflowEnvironmentManager] = None) -> None:
        self.manager = manager or AirflowEnvironmentManager()

    async def execute(self, request: BuildEtlDagsRequest) -> BuildEtlDagsResponse:
        if request.base_path:
            self.manager.base_path = Path(request.base_path).resolve()
            self.manager.dags_dir = self.manager.base_path / "dags"
            self.manager.logs_dir = self.manager.base_path / "logs"
            self.manager.plugins_dir = self.manager.base_path / "plugins"

        # Ensure directories exist
        self.manager.prepare_directories(create_missing=True)

        # Write DAGs
        ingestion_path = self.manager.dags_dir / f"{request.ingestion_dag_id}.py"
        processing_path = self.manager.dags_dir / f"{request.processing_dag_id}.py"
        quality_path = self.manager.dags_dir / f"{request.quality_dag_id}.py"

        ingestion_path.write_text(_ingestion_dag_source(request.ingestion_dag_id), encoding="utf-8")
        processing_path.write_text(_processing_dag_source(request.processing_dag_id), encoding="utf-8")
        quality_path.write_text(_quality_dag_source(request.quality_dag_id), encoding="utf-8")

        dag_files = [str(ingestion_path), str(processing_path), str(quality_path)]
        diagnostics = {
            "dags_dir": str(self.manager.dags_dir),
            "ingestion_exists": str(ingestion_path.exists()),
            "processing_exists": str(processing_path.exists()),
            "quality_exists": str(quality_path.exists()),
        }
        return BuildEtlDagsResponse(success=True, dag_files=dag_files, diagnostics=diagnostics)
