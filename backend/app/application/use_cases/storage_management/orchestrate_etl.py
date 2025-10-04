"""Use case for building ETL orchestration DAGs (Task 1.4.2).

Generates Airflow DAGs for data ingestion, document processing, and data
quality validation. Uses the existing AirflowEnvironmentManager to ensure
folder structure and write DAG files with daily schedules and basic
failure logging/notification hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.infrastructure.orchestration import AirflowEnvironmentManager


def _ingestion_dag_source(dag_id: str = "rag_data_ingestion") -> str:
    return "\n".join(
        [
            '"""RAG - Data Ingestion DAG (auto-generated)."""',
            "from __future__ import annotations",
            "",
            "from datetime import datetime, timedelta",
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "",
            "def _fail_handler(context):",
            '    print(f"[FAIL] Task {context['task_instance'].task_id} failed at {context['ts']}")',
            '    print("Sending failure notification (simulated)...")',
            "",
            "def _discover_sources(**kwargs):",
            '    print("Scanning registered data sources (simulated)...")',
            '    return {"discovered": 3}',
            "",
            "def _ingest_new_data(**kwargs):",
            '    print("Ingesting new/changed data (simulated incremental)...")',
            "",
            "default_args = {",
            '    "owner": "fitvise",',
            '    "depends_on_past": False,',
            '    "retries": 1,',
            '    "retry_delay": timedelta(minutes=5),',
            '    "email_on_failure": True,',
            '    "email": ["rag-alerts@example.com"],',
            "}",
            f"with DAG('{dag_id}', start_date=datetime.utcnow(), schedule_interval='@daily', catchup=False, default_args=default_args, tags=['rag','etl','ingestion']) as dag:",
            "    discover = PythonOperator(task_id='discover_sources', python_callable=_discover_sources, on_failure_callback=_fail_handler)",
            "    ingest = PythonOperator(task_id='ingest_incremental', python_callable=_ingest_new_data, on_failure_callback=_fail_handler)",
            "    discover >> ingest",
            "",
        ]
    )


def _processing_dag_source(dag_id: str = "rag_document_processing") -> str:
    return "\n".join(
        [
            '"""RAG - Document Processing DAG (auto-generated)."""',
            "from __future__ import annotations",
            "",
            "from datetime import datetime, timedelta",
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "",
            "def _fail_handler(context):",
            '    print(f"[FAIL] Task {context['task_instance'].task_id} failed at {context['ts']}")',
            '    print("Sending failure notification (simulated)...")',
            "",
            "def _extract_text(**kwargs):",
            '    print("Extracting text via Docling/Tika (simulated)...")',
            "",
            "def _clean_text(**kwargs):",
            '    print("Running spaCy-based cleaning (simulated)...")',
            "",
            "def _enrich_metadata(**kwargs):",
            '    print("Extracting keywords/entities (simulated)...")',
            "",
            "default_args = {",
            '    "owner": "fitvise",',
            '    "depends_on_past": False,',
            '    "retries": 1,',
            '    "retry_delay": timedelta(minutes=5),',
            '    "email_on_failure": True,',
            '    "email": ["rag-alerts@example.com"],',
            "}",
            f"with DAG('{dag_id}', start_date=datetime.utcnow(), schedule_interval='@daily', catchup=False, default_args=default_args, tags=['rag','etl','processing']) as dag:",
            "    extract = PythonOperator(task_id='extract_text', python_callable=_extract_text, on_failure_callback=_fail_handler)",
            "    clean = PythonOperator(task_id='clean_text', python_callable=_clean_text, on_failure_callback=_fail_handler)",
            "    metadata = PythonOperator(task_id='enrich_metadata', python_callable=_enrich_metadata, on_failure_callback=_fail_handler)",
            "    extract >> clean >> metadata",
            "",
        ]
    )


def _quality_dag_source(dag_id: str = "rag_data_quality") -> str:
    return "\n".join(
        [
            '"""RAG - Data Quality Validation DAG (auto-generated)."""',
            "from __future__ import annotations",
            "",
            "from datetime import datetime, timedelta",
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "",
            "def _fail_handler(context):",
            '    print(f"[FAIL] Task {context['task_instance'].task_id} failed at {context['ts']}")',
            '    print("Sending failure notification (simulated)...")',
            "",
            "def _validate_quality(**kwargs):",
            '    print("Running Great Expectations checks (simulated)...")',
            "",
            "default_args = {",
            '    "owner": "fitvise",',
            '    "depends_on_past": False,',
            '    "retries": 0,',
            '    "retry_delay": timedelta(minutes=5),',
            '    "email_on_failure": True,',
            '    "email": ["rag-alerts@example.com"],',
            "}",
            f"with DAG('{dag_id}', start_date=datetime.utcnow(), schedule_interval='@daily', catchup=False, default_args=default_args, tags=['rag','etl','quality']) as dag:",
            "    validate = PythonOperator(task_id='validate_data_quality', python_callable=_validate_quality, on_failure_callback=_fail_handler)",
            "",
        ]
    )


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

