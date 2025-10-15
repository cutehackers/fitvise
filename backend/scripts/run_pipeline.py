#!/usr/bin/env python3
from __future__ import annotations

"""
CLI wrapper for the unified RAG pipeline orchestrator.

ScheduleOptions
- `manual`: run ad-hoc via this script. The config is still required, but no
  automated trigger is implied.
- `cron`: pipeline is triggered by an external cron job using the supplied
  `cron` expression.
- `airflow`: execution is delegated to an Airflow DAG; the spec informs the DAG
  of paths, processors, and storage.

Usage:
  python backend/scripts/run_pipeline.py --config rag_pipeline.yaml [--dry-run]

Example `rag_pipeline.yaml` files

  # Manual run (on-demand)
  inputs:
    path: ./data/manual
    recurse: true
    include: ["*.pdf"]
  schedule:
    mode: manual
  storage:
    provider: local
    base_dir: ./storage
    bucket: rag-processed

  # Cron-based monthly run (trigger cron separately)
  inputs:
    path: /var/rag/monthly
  schedule:
    mode: cron
    cron: "0 3 1 * *"  # 03:00 on day 1 each month
  storage:
    provider: minio
    endpoint: localhost:9000
    bucket: rag-processed

  # Airflow-managed run (DAG calls the orchestrator)
  inputs:
    path: s3://rag/raw
    recurse: true
  schedule:
    mode: airflow
    cron: "@monthly"  # optional override for DAG generator
  storage:
    provider: minio
    endpoint: airflow-minio:9000
    bucket: rag-processed
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add backend/ to sys.path so "app" imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.config import PipelineSpec
from app.pipeline.orchestrator import run_pipeline


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG Phase 1 document processing pipeline")
    parser.add_argument("--config", required=True, help="Path to rag_pipeline.yaml (or .json)")
    parser.add_argument("--dry-run", action="store_true", help="Discover only; do not store outputs")
    args = parser.parse_args()

    cfg = PipelineSpec.from_file(args.config)

    # Optionally disable storing: set bucket to temp and use local storage
    if args.dry_run:
        # Override to a throwaway bucket/path
        data = cfg.model_dump()
        data.setdefault("storage", {})
        data["storage"]["bucket"] = "rag-dry-run"
        data["storage"]["provider"] = "local"
        cfg = PipelineSpec.model_validate(data)

    summary = await run_pipeline(cfg)
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
