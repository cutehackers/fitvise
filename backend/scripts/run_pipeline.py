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

Remember: `inputs` always defines which documents are processed (usually
filesystem paths). The optional `sources` section drives pre-ingestion flows
such as audits, categorisation, DB/web/API pulls.

Example `rag_pipeline.yaml` files

  # Manual run (on-demand)
  documents:
    path: ./data/manual
    recurse: true
    include: ["*.pdf"]
  schedule:
    mode: manual
  storage:
    provider: local
    base_dir: ./storage
    bucket: rag-processed
  sources:
    audit:
      enabled: true
      scan_paths: ["./data/manual"]
    databases: []
    web: []

  # Cron-based monthly run (trigger cron separately, e.g. crontab)
  documents:
    path: /var/rag/monthly
    include: ["*.pdf", "*.md"]
  schedule:
    mode: cron
    cron: "0 3 1 * *"  # 03:00 on day 1 each month
  storage:
    provider: minio
    endpoint: localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket: rag-processed
  sources:
    databases:
      - name: finance
        connector_type: postgres
        driver: postgresql+psycopg2
        host: db.internal
        port: 5432
        database: reporting
        username: rag
        password: secret
        schema: public
        params:
          sslmode: disable
        sample_limit: 25
        sample_collection: null
        fetch_samples: true
    web: []
    document_apis:
      enabled: false
    audit:
      enabled: false
    categorize:
      enabled: false

  # Airflow-managed run (DAG calls the orchestrator)
  documents:
    path: s3://rag/raw
    recurse: true
  schedule:
    mode: airflow
    cron: "@monthly"  # optional override captured in DAG generator
  storage:
    provider: minio
    endpoint: airflow-minio:9000
    access_key: airflow
    secret_key: airflowsecret
    bucket: rag-processed
    secure: true
  sources:
    web:
      - start_urls:
          - "https://internal-wiki.local/docs"
          - "https://internal-wiki.local/handbooks"
        allowed_domains:
          - "internal-wiki.local"
        max_depth: 2
        max_pages: 100
        include_patterns:
          - "/docs/.*"
          - "/handbooks/.*"
        exclude_patterns:
          - ".*login.*"
        follow_css_selectors:
          - "main article"
        follow_xpath: []
        headers:
          User-Agent: "FitViseBot/1.0"
        cookies: {}
        follow_external_links: false
    databases: []
    document_apis:
      enabled: true
      include_common_apis: true
      validate_endpoints: false
    audit:
      enabled: false
    categorize:
      enabled: false
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

    spec = PipelineSpec.from_file(args.config)

    # Optionally disable storing: set bucket to temp and use local storage
    if args.dry_run:
        # Override to a throwaway bucket/path
        data = spec.model_dump()
        data.setdefault("storage", {})
        data["storage"]["bucket"] = "rag-dry-run"
        data["storage"]["provider"] = "local"
        spec = PipelineSpec.model_validate(data)

    summary = await run_pipeline(spec)
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
