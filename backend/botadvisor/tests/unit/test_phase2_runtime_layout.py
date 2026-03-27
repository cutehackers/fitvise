from __future__ import annotations

import importlib
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[3]
BOTADVISOR_ROOT = BACKEND_ROOT / "botadvisor"


def test_settings_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("BOTADVISOR_ENV", "test")
    monkeypatch.setenv("WEAVIATE_URL", "http://vector.example:8080")
    monkeypatch.setenv("STORAGE_LOCAL_PATH", "/tmp/botadvisor-artifacts")

    from botadvisor.app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.environment == "test"
    assert settings.weaviate_url == "http://vector.example:8080"
    assert settings.storage_local_path == "/tmp/botadvisor-artifacts"

    get_settings.cache_clear()


@pytest.mark.parametrize(
    "module_name",
    [
        "botadvisor.app.chat",
        "botadvisor.app.core",
        "botadvisor.app.ingestion",
        "botadvisor.app.llm",
        "botadvisor.app.observability",
        "botadvisor.app.retrieval",
        "botadvisor.app.storage",
    ],
)
def test_feature_boundaries_are_importable_packages(module_name: str):
    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    "file_path",
    [
        BOTADVISOR_ROOT / "scripts" / "embed_upsert.py",
        BOTADVISOR_ROOT / "scripts" / "ingest.py",
        BOTADVISOR_ROOT / "app" / "storage" / "local_storage.py",
        BOTADVISOR_ROOT / "app" / "storage" / "minio_client.py",
    ],
)
def test_runtime_modules_use_canonical_botadvisor_imports(file_path: Path):
    source = file_path.read_text(encoding="utf-8")

    assert "sys.path" not in source
    assert "from app." not in source
    assert "from botadvisor.app." in source
