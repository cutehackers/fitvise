from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "setup_vector_store.py"


def load_module():
    spec = importlib.util.spec_from_file_location("botadvisor_setup_vector_store", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_accepts_force_and_dimension(monkeypatch):
    module = load_module()
    captured: dict[str, object] = {}

    def fake_setup_vector_store(*, force: bool, dimension: int | None):
        captured["force"] = force
        captured["dimension"] = dimension
        return {
            "backend": "weaviate",
            "collection_name": "BotAdvisorDocs",
            "created": True,
            "force": force,
            "dimension": dimension,
        }

    monkeypatch.setattr(module, "setup_vector_store", fake_setup_vector_store)

    exit_code = module.main(["--force", "--dimension", "768"])

    assert exit_code == 0
    assert captured == {
        "force": True,
        "dimension": 768,
    }


def test_setup_vector_store_returns_summary_for_existing_collection(monkeypatch):
    module = load_module()

    class FakeCollections:
        def exists(self, name: str) -> bool:
            assert name == "BotAdvisorDocs"
            return True

        def delete(self, name: str) -> None:
            raise AssertionError("delete should not be called without --force")

        def create(self, **kwargs) -> None:
            raise AssertionError("create should not be called when collection exists")

    class FakeClient:
        def __init__(self):
            self.collections = FakeCollections()

        def close(self) -> None:
            pass

    monkeypatch.setattr(module, "connect_to_weaviate", lambda: FakeClient())
    monkeypatch.setattr(module, "resolve_embedding_dimension", lambda requested_dimension=None: 384)

    summary = module.setup_vector_store(force=False, dimension=None)

    assert summary == {
        "backend": "weaviate",
        "collection_name": "BotAdvisorDocs",
        "created": False,
        "force": False,
        "dimension": 384,
    }
