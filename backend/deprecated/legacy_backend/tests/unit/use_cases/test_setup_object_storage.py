from __future__ import annotations

import os
from pathlib import Path

from app.application.use_cases.storage_management import (
    SetupObjectStorageUseCase,
    SetupObjectStorageRequest,
)


async def test_setup_object_storage_local_tmp(tmp_path: Path):
    base = tmp_path / "rag_storage"
    uc = SetupObjectStorageUseCase()
    req = SetupObjectStorageRequest(
        provider="local",
        base_dir=str(base),
        buckets=["rag-raw", "rag-processed", "rag-metadata"],
        test_object=True,
    )
    res = await uc.execute(req)

    assert res.success is True
    assert set(res.created_buckets) == {"rag-raw", "rag-processed", "rag-metadata"}
    # Ensure directories exist
    for b in res.created_buckets:
        assert (base / b).is_dir(), f"Bucket dir missing: {b}"

