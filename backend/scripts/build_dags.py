#!/usr/bin/env python3
"""
Build ETL DAGs (ingestion, processing, quality) under an Airflow project path.

Example:
  python backend/scripts/build_dags.py --base-path /tmp/airflow \
    --ingestion-dag rag_data_ingestion --processing-dag rag_document_processing --quality-dag rag_data_quality
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add backend/ to path

from app.application.use_cases.storage_management import (
    BuildEtlDagsUseCase,
    BuildEtlDagsRequest,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 1 ETL DAGs")
    parser.add_argument("--base-path", required=True, help="Airflow project base path (contains dags/logs/plugins)")
    parser.add_argument("--ingestion-dag", default="rag_data_ingestion")
    parser.add_argument("--processing-dag", default="rag_document_processing")
    parser.add_argument("--quality-dag", default="rag_data_quality")
    args = parser.parse_args()

    uc = BuildEtlDagsUseCase()
    res = await uc.execute(
        BuildEtlDagsRequest(
            base_path=args.base_path,
            ingestion_dag_id=args.ingestion_dag,
            processing_dag_id=args.processing_dag,
            quality_dag_id=args.quality_dag,
        )
    )
    print(json.dumps({"success": res.success, "dags": res.dag_files, "diagnostics": res.diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

