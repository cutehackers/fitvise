from __future__ import annotations

from pathlib import Path

from app.application.use_cases.storage_management import (
    BuildEtlDagsUseCase,
    BuildEtlDagsRequest,
)


async def test_build_etl_dags_creates_files(tmp_path: Path):
    base = tmp_path / "airflow"
    uc = BuildEtlDagsUseCase()
    res = await uc.execute(
        BuildEtlDagsRequest(
            base_path=str(base),
            ingestion_dag_id="ingestion_test",
            processing_dag_id="processing_test",
            quality_dag_id="quality_test",
        )
    )
    assert res.success is True
    for name in ("ingestion_test.py", "processing_test.py", "quality_test.py"):
        assert (base / "dags" / name).exists(), f"Missing DAG file: {name}"

