"""RAG - Document Processing DAG (auto-generated)."""
from __future__ import annotations

import asyncio, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator


def _fail_handler(context):
    print(f"[FAIL] Task {context['task_instance'].task_id} failed at {context['ts']}")
    print("Sending failure notification (simulated)...")


def _extract_text(**kwargs):
    try:
        from app.application.use_cases.document_processing import ProcessPdfsUseCase, ProcessPdfsRequest
    except Exception:
        print("ProcessPdfsUseCase unavailable; skipping.")
        return {"documents": 0}
    input_dir = os.getenv("RAG_INPUT_DIR")
    docs = []
    if input_dir and Path(input_dir).is_dir():
        for p in Path(input_dir).rglob("*.pdf"):
            docs.append(str(p))
    if not docs:
        print("No PDFs found; skipping extraction.")
        return {"documents": 0}

    async def _run():
        uc = ProcessPdfsUseCase()
        res = await uc.execute(ProcessPdfsRequest(file_paths=docs))
        return len(res.documents)

    n = asyncio.run(_run())
    print(f"Extracted {n} documents")
    return {"documents": n}


def _clean_text(**kwargs):
    try:
        from app.application.use_cases.document_processing import NormalizeTextUseCase, NormalizeTextRequest
    except Exception:
        print("NormalizeTextUseCase unavailable; skipping.")
        return {"cleaned": 0}
    texts = ["placeholder text for cleaning"]

    async def _run():
        uc = NormalizeTextUseCase()
        res = await uc.execute(NormalizeTextRequest(texts=texts))
        return len(res.results)

    n = asyncio.run(_run())
    print(f"Cleaned {n} texts")
    return {"cleaned": n}


def _enrich_metadata(**kwargs):
    try:
        from app.application.use_cases.document_processing import ExtractMetadataUseCase, ExtractMetadataRequest
    except Exception:
        print("ExtractMetadataUseCase unavailable; skipping.")
        return {"meta": 0}
    texts = ["placeholder text for metadata"]

    async def _run():
        uc = ExtractMetadataUseCase()
        res = await uc.execute(ExtractMetadataRequest(texts=texts, top_k_keywords=5))
        return len(res.results)

    n = asyncio.run(_run())
    print(f"Extracted metadata for {n} texts")
    return {"meta": n}


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
    start_date=datetime.now(timezone.utc),
    schedule_interval="$schedule",
    catchup=False,
    default_args=default_args,
    tags=$tags,
) as dag:
    extract = PythonOperator(
        task_id="extract_text", python_callable=_extract_text, on_failure_callback=_fail_handler
    )
    clean = PythonOperator(
        task_id="clean_text", python_callable=_clean_text, on_failure_callback=_fail_handler
    )
    metadata = PythonOperator(
        task_id="enrich_metadata", python_callable=_enrich_metadata, on_failure_callback=_fail_handler
    )
    extract >> clean >> metadata
