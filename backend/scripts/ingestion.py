#!/usr/bin/env python3
from __future__ import annotations

"""
RAG Document Ingestion Pipeline (Phase 2)

Phase 2 of RAG build pipeline. Handles document discovery, processing,
normalization, chunking, and storage to prepare documents for embedding.

Usage:
  python backend/scripts/ingestion.py --config rag_pipeline.yaml [--dry-run] [--verbose]

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
    bucket: rag-source
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
    bucket: rag-source
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
    bucket: rag-source
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
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add backend/ to sys.path so "app" imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.config import PipelineSpec
from app.pipeline.orchestrator import UseCaseBundle, _build_use_cases, _discover_documents, _process_documents, _maybe_chunk_documents, _storage_client, RunSummary, _init_pipeline, ingestion
from app.pipeline.contracts import StorageObject
from app.infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository


async def run_ingestion(
    config: PipelineSpec,
    document_repository: InMemoryDocumentRepository,
    dry_run: bool = False,
    verbose: bool = False
):
    """Run ingestion with shared document repository instance.

    Args:
        config: Pipeline specification
        document_repository: Shared repository instance from orchestrator
        dry_run: Run in dry-run mode
        verbose: Enable verbose logging

    Returns:
        Ingestion summary with processed documents stored in shared repository
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Running ingestion with shared document repository...")

    # Create a custom use case bundle with the shared repository
    standard_bundle = _build_use_cases()

    # Create custom bundle with shared repository
    custom_bundle = UseCaseBundle(
        repository=standard_bundle.repository,
        document_repository=document_repository,  # Use shared repository
        storage=standard_bundle.storage,
        process_pdfs=standard_bundle.process_pdfs,
        normalize_text=standard_bundle.normalize_text,
        extract_metadata=standard_bundle.extract_metadata,
        validate_quality=standard_bundle.validate_quality,
        tika=standard_bundle.tika,
        connect_databases=standard_bundle.connect_databases,
        web_scraping=standard_bundle.web_scraping,
        audit_sources=standard_bundle.audit_sources,
        categorize_sources=standard_bundle.categorize_sources,
        document_apis=standard_bundle.document_apis,
        chunk_documents=standard_bundle.chunk_documents,
    )

    # Initialize phase
    use_cases = await _init_pipeline(config)
    run_id = datetime.now(timezone.utc).isoformat()

    # Discover documents
    documents, errors = await _discover_documents(config, custom_bundle)
    discovery_error_count = len(errors)

    # Process documents (this will store them in the shared repository)
    client = _storage_client(config)
    stored_objects, origin_counts, skipped, processed_document_ids = await _process_documents(
        documents=documents,
        spec=config,
        use_cases=custom_bundle,  # Use custom bundle with shared repository
        client=client,
        run_id=run_id,
        errors=errors,
    )

    # Chunk documents
    chunk_summary = await _maybe_chunk_documents(
        processed_document_ids=processed_document_ids,
        use_cases=custom_bundle,  # Use custom bundle with shared repository
        spec=config,
        run_id=run_id,
        dry_run=dry_run,
        errors=errors,
    )

    processing_failures = max(len(errors) - discovery_error_count, 0)
    counters = {
        "run_id": run_id,
        "discovered": len(documents),
        "processed": len(stored_objects),
        "skipped": skipped,
        "failed": processing_failures,
        "by_origin": origin_counts,
        "discovery_errors": discovery_error_count,
        "total_errors": len(errors),
        "chunking": chunk_summary,
    }

    summary = RunSummary(
        processed=len(stored_objects),
        skipped=skipped,
        failed=processing_failures,
        stored=stored_objects,
        errors=errors,
        counters=counters,
    )

    logger.info(f"Ingestion completed: {summary.processed} documents stored in shared repository")
    return summary


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG Phase 2 document ingestion pipeline")
    parser.add_argument("--config", required=True, help="Path to rag_pipeline.yaml (or .json)")
    parser.add_argument("--dry-run", action="store_true", help="Discover only; do not store outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", help="Path to save ingestion summary as JSON")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    spec = PipelineSpec.from_file(args.config)

    # Optionally disable storing: set bucket to temp and use local storage
    if args.dry_run:
        # Override to a throwaway bucket/path
        data = spec.model_dump()
        data.setdefault("storage", {})
        data["storage"]["bucket"] = "rag-dry-run"
        data["storage"]["provider"] = "local"
        spec = PipelineSpec.model_validate(data)

    summary = await ingestion(spec, dry_run=args.dry_run)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary.as_dict(), f, indent=2)
        print(f"Ingestion summary saved to {args.output}")

    # Print summary to console
    print("\n" + "="*60)
    print("RAG DOCUMENT INGESTION SUMMARY")
    print("="*60)
    print(f"Documents Discovered: {summary.counters.get('discovered', 0)}")
    print(f"Documents Processed: {summary.processed}")
    print(f"Documents Skipped: {summary.skipped}")
    print(f"Documents Failed: {summary.failed}")
    print(f"Chunks Generated: {summary.counters.get('chunking', {}).get('total_chunks', 0)}")
    print(f"Processing Errors: {summary.counters.get('total_errors', 0)}")

    if summary.errors:
        print(f"\n⚠️  {len(summary.errors)} errors occurred:")
        for error in summary.errors[:5]:  # Show first 5 errors
            print(f"   - {error.get('message', str(error))}")
        if len(summary.errors) > 5:
            print(f"   ... and {len(summary.errors) - 5} more errors")

    print("\n" + json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
