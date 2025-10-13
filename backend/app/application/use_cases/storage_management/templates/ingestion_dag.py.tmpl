"""RAG - Data Ingestion DAG (auto-generated)."""
from __future__ import annotations

import asyncio, os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def _try_import():
    try:
        from app.application.use_cases.knowledge_sources.audit_data_sources import (
            AuditDataSourcesUseCase,
            AuditDataSourcesRequest,
        )
        return AuditDataSourcesUseCase, AuditDataSourcesRequest
    except Exception:
        return None, None


def _fail_handler(context):
    print(f"[FAIL] Task {context['task_instance'].task_id} failed at {context['ts']}")
    print("Sending failure notification (simulated)...")


def _discover_sources(**kwargs):
    AU, AR = _try_import()
    if AU is None:
        print("Use cases not available; skipping discovery.")
        return {"discovered": 0}
    paths = os.getenv("RAG_SCAN_PATHS", "").split(",") if os.getenv("RAG_SCAN_PATHS") else None

    async def _run():
        uc = AU()
        req = AR(scan_paths=paths, max_scan_depth=2, min_file_count=1, save_to_repository=False)
        res = await uc.execute(req)
        return res.total_discovered

    discovered = asyncio.run(_run())
    print(f"Discovered sources: {discovered}")
    return {"discovered": discovered}


def _ingest_new_data(**kwargs):
    print("Ingesting new/changed data (placeholder) – integrate connectors here.")


default_args = {
    "owner": "fitvise",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["rag-alerts@example.com"],
}


with DAG(
    dag_id="$dag_id",
    start_date=datetime.utcnow(),
    schedule_interval="$schedule",
    catchup=False,
    default_args=default_args,
    tags=$tags,
) as dag:
    discover = PythonOperator(
        task_id="discover_sources", python_callable=_discover_sources, on_failure_callback=_fail_handler
    )
    ingest = PythonOperator(
        task_id="ingest_incremental", python_callable=_ingest_new_data, on_failure_callback=_fail_handler
    )
    discover >> ingest

