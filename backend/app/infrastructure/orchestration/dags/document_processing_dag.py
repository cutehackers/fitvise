"""Airflow DAG: Document processing pipeline (optional).

This DAG demonstrates an end-to-end chain for Phase 1.3:
- Extract PDF to markdown using DoclingPdfProcessor with fallbacks
- Clean text via SpacyTextProcessor
- Extract metadata via SpacyTextProcessor
- Validate quality using ValidateQualityUseCase

The DAG is guarded by try/except so importing it without Airflow won't crash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _build_pipeline_functions():  # Localize imports to avoid Airflow dependency at import-time
    from app.infrastructure.external_services.data_sources.file_processors import DoclingPdfProcessor, SpacyTextProcessor
    from app.application.use_cases.document_processing import (
        ValidateQualityUseCase,
        ValidateQualityRequest,
    )

    processor_pdf = DoclingPdfProcessor()
    processor_text = SpacyTextProcessor()
    quality_uc = ValidateQualityUseCase()

    def extract_pdf_callable(**context):
        pdf_path = context["dag_run"].conf.get("pdf_path") if context.get("dag_run") else None
        if not pdf_path:
            pdf_path = os.environ.get("DOC_PIPELINE_PDF", "./sample.pdf")
        path = Path(pdf_path)
        result = processor_pdf.process_pdf_from_path(path)
        if not result.success:
            raise RuntimeError(result.error or "PDF processing failed")
        context["ti"].xcom_push(key="markdown", value=result.markdown or result.text)
        return {"ok": True, "tables": len(result.tables)}

    def clean_text_callable(**context):
        md = context["ti"].xcom_pull(key="markdown") or ""
        cleaned = processor_text.clean_text(md)
        context["ti"].xcom_push(key="cleaned_text", value=cleaned.cleaned)
        return {"ok": True, "tokens": len(cleaned.tokens)}

    def metadata_callable(**context):
        cleaned = context["ti"].xcom_pull(key="cleaned_text") or ""
        metas = processor_text.extract([cleaned], top_k_keywords=10)
        meta_dict: Dict[str, Any] = metas[0].as_dict() if metas else {}
        context["ti"].xcom_push(key="metadata", value=json.dumps(meta_dict))
        return {"ok": True, "top_keywords": meta_dict.get("keywords", [])[:5]}

    def quality_callable(**context):
        cleaned = context["ti"].xcom_pull(key="cleaned_text") or ""
        request = ValidateQualityRequest(texts=[cleaned])
        # Run synchronously since DAG tasks are sync; the use case returns async
        import asyncio

        response = asyncio.run(quality_uc.execute(request))
        report = response.reports[0].as_dict() if response.reports else {}
        context["ti"].xcom_push(key="quality_report", value=json.dumps(report))
        return {"ok": True, "quality": report.get("overall_score")}

    return extract_pdf_callable, clean_text_callable, metadata_callable, quality_callable


try:  # Optional Airflow DAG
    from datetime import datetime
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    extract_pdf_callable, clean_text_callable, metadata_callable, quality_callable = _build_pipeline_functions()

    with DAG(
        dag_id="rag_document_processing_pipeline",
        description="Phase 1.3: PDF -> Clean -> Metadata -> Quality",
        schedule_interval=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["rag", "processing", "phase1"],
    ) as dag:
        t1 = PythonOperator(task_id="extract_pdf", python_callable=extract_pdf_callable)
        t2 = PythonOperator(task_id="clean_text", python_callable=clean_text_callable)
        t3 = PythonOperator(task_id="extract_metadata", python_callable=metadata_callable)
        t4 = PythonOperator(task_id="validate_quality", python_callable=quality_callable)

        t1 >> t2 >> t3 >> t4
except Exception:  # pragma: no cover - keep repository import-safe without Airflow
    dag = None  # type: ignore

