#!/usr/bin/env python3
"""
Setup object storage buckets (Task 1.4.1 helper).

Examples:
  python backend/scripts/setup_storage.py --provider local --base-dir /tmp/rag_storage \
      --buckets rag-raw rag-processed rag-metadata

  python backend/scripts/setup_storage.py --provider minio --endpoint localhost:9000 \
      --access-key minioadmin --secret-key minioadmin --buckets rag-raw rag-processed rag-metadata
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add backend/ to path

from app.application.use_cases.storage_management import (
    SetupObjectStorageUseCase,
    SetupObjectStorageRequest,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Setup object storage for the RAG pipeline")
    parser.add_argument("--provider", choices=["local", "minio"], default="local")
    parser.add_argument("--base-dir", help="Base directory for local provider", default=None)
    parser.add_argument("--endpoint", help="MinIO endpoint (host:port)", default=None)
    parser.add_argument("--access-key", help="MinIO access key", default=None)
    parser.add_argument("--secret-key", help="MinIO secret key", default=None)
    parser.add_argument("--secure", action="store_true", help="Use TLS for MinIO endpoint")
    parser.add_argument("--buckets", nargs="+", default=["rag-raw", "rag-processed", "rag-metadata"])
    parser.add_argument("--no-test-object", action="store_true", help="Skip test object write")
    args = parser.parse_args()

    use_case = SetupObjectStorageUseCase()
    req = SetupObjectStorageRequest(
        provider=args.provider,
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=args.secure,
        base_dir=args.base_dir,
        buckets=args.buckets,
        test_object=not args.no_test_object,
    )
    res = await use_case.execute(req)
    print(json.dumps({
        "provider": res.provider,
        "created_buckets": res.created_buckets,
        "environment": res.environment,
        "test_put_key": res.test_put_key,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

