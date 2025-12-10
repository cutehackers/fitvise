"""RAG - Data Quality Validation DAG (auto-generated)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.operators.python import PythonOperator


def _fail_handler(context):
    print(f"[FAIL] Task {context['task_instance'].task_id} failed at {context['ts']}")
    print("Sending failure notification (simulated)...")


def _validate_quality(**kwargs):
    try:
        from app.application.use_cases.document_processing import ValidateQualityUseCase, ValidateQualityRequest
    except Exception:
        print("ValidateQualityUseCase unavailable; skipping.")
        return {"validated": 0}

    async def _run():
        uc = ValidateQualityUseCase()
        res = await uc.execute(ValidateQualityRequest(texts=["quality sample text"]))
        return len(res.reports)

    n = asyncio.run(_run())
    print(f"Validated {n} items")
    return {"validated": n}


default_args = {
    "owner": "fitvise",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["rag-alerts@example.com"],
}


with DAG(
    dag_id="$dag_id",
    start_date=datetime.now(timezone.utc),
    schedule_interval="$schedule",
    catchup=False,
    default_args=default_args,
    tags=$tags,
) as dag:
    validate = PythonOperator(
        task_id="validate_data_quality", python_callable=_validate_quality, on_failure_callback=_fail_handler
    )
