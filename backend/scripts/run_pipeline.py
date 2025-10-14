#!/usr/bin/env python3
"""
Run the Phase 1 pipeline (Task 1.3 → Task 1.4) from CLI.

Processes one file or every PDF in a directory, then stores cleaned
markdown to object storage with metadata and tags.

Examples:
  python backend/scripts/run_pipeline.py --input-file /path/to/file.pdf \
      --provider local --base-dir /tmp/rag_storage --bucket rag-processed

  python backend/scripts/run_pipeline.py --input-dir /path/to/pdf_folder \
      --provider minio --endpoint localhost:9000 --access-key minioadmin --secret-key minioadmin
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add backend/ to path

from app.application.use_cases.document_processing import (
    ProcessPdfsUseCase,
    ProcessPdfsRequest,
    NormalizeTextUseCase,
    NormalizeTextRequest,
    ExtractMetadataUseCase,
    ExtractMetadataRequest,
    ValidateQualityUseCase,
    ValidateQualityRequest,
)
from app.application.use_cases.storage_management import (
    SetupObjectStorageUseCase,
    SetupObjectStorageRequest,
)
from app.infrastructure.storage.object_storage.minio_client import (
    ObjectStorageClient,
    ObjectStorageConfig,
)


def _iter_pdf_paths(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.pdf") if p.is_file()]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 pipeline and store results")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-file", help="A single PDF file to process", default=None)
    g.add_argument("--input-dir", help="Process all PDFs under this directory", default=None)

    parser.add_argument("--provider", choices=["local", "minio"], default="local")
    parser.add_argument("--base-dir", help="Base directory for local provider", default=None)
    parser.add_argument("--endpoint", help="MinIO endpoint (host:port)", default=None)
    parser.add_argument("--access-key", help="MinIO access key", default=None)
    parser.add_argument("--secret-key", help="MinIO secret key", default=None)
    parser.add_argument("--secure", action="store_true")
    parser.add_argument("--bucket", default="rag-processed")
    parser.add_argument("--ensure-storage", action="store_true", help="Ensure buckets before run")

    args = parser.parse_args()

    # Collect document paths
    docs: List[Path] = []
    if args.input_file:
        p = Path(args.input_file)
        if not p.is_file():
            print(json.dumps({"error": f"File not found: {p}"}))
            return 1
        docs = [p]
    else:
        root = Path(args.input_dir)
        if not root.is_dir():
            print(json.dumps({"error": f"Directory not found: {root}"}))
            return 1
        docs = _iter_pdf_paths(root)
        if not docs:
            print(json.dumps({"error": f"No PDFs under: {root}"}))
            return 1

    # Use cases
    pdf_uc = ProcessPdfsUseCase()
    clean_uc = NormalizeTextUseCase()
    meta_uc = ExtractMetadataUseCase()
    qual_uc = ValidateQualityUseCase()
    storage_uc = SetupObjectStorageUseCase()

    # Storage ensure (optional)
    if args.ensure_storage:
        await storage_uc.execute(
            SetupObjectStorageRequest(
                provider=args.provider,
                endpoint=args.endpoint,
                access_key=args.access_key,
                secret_key=args.secret_key,
                secure=args.secure,
                base_dir=args.base_dir,
                buckets=["rag-raw", "rag-processed", "rag-metadata"],
                test_object=True,
            )
        )

    client = ObjectStorageClient(
        ObjectStorageConfig(
            provider=args.provider,
            endpoint=args.endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
            secure=args.secure,
            base_dir=Path(args.base_dir).resolve() if args.base_dir else None,
        )
    )
    client.ensure_bucket(args.bucket)

    stored = []
    for path in docs:
        # 1) PDF → markdown
        pdf_res = await pdf_uc.execute(ProcessPdfsRequest(file_paths=[str(path)]))
        if not pdf_res.documents:
            continue
        d = pdf_res.documents[0]
        markdown = d.markdown

        # 2) Clean
        clean_res = await clean_uc.execute(NormalizeTextRequest(texts=[markdown]))
        cleaned = clean_res.results[0].cleaned_text if clean_res.results else markdown

        # 3) Metadata
        meta_res = await meta_uc.execute(ExtractMetadataRequest(texts=[cleaned], top_k_keywords=10))
        keywords = meta_res.results[0].keywords if meta_res.results else []

        # 4) Quality
        qual_res = await qual_uc.execute(ValidateQualityRequest(texts=[cleaned]))
        if qual_res.reports:
            qr = qual_res.reports[0]
            overall = qr.overall_score
        else:
            overall = 0.0

        # 5) Store
        now = datetime.now(timezone.utc)
        dated = now.strftime("%Y/%m/%d")
        name = path.stem + ".md"
        key = f"{dated}/{name}"
        tags = {"pipeline": "phase1", "task": "1.4", "validated": str(overall >= 0.5).lower()}
        meta = {"source": str(path), "keywords": ",".join(keywords[:10]), "overall_quality": str(overall)}
        client.put_object(args.bucket, key, cleaned.encode("utf-8"), content_type="text/markdown", metadata=meta, tags=tags)
        stored.append({"bucket": args.bucket, "key": key})

    print(json.dumps({"processed": len(docs), "stored": stored}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
