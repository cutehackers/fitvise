from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "embed_upsert.py"


def load_module():
    spec = importlib.util.spec_from_file_location("botadvisor_embed_upsert", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_module_imports_from_backend_project():
    module = load_module()

    assert module.__name__ == "botadvisor_embed_upsert"


def test_main_accepts_input_alias(monkeypatch, tmp_path):
    module = load_module()
    captured: dict[str, object] = {}

    class FakeScript:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self, input_path: str):
            captured["input_path"] = input_path
            return {"successful_batches": 1, "failed_batches": 0}

    monkeypatch.setattr(module, "EmbedUpsertScript", FakeScript)

    exit_code = module.main(
        [
            "--input",
            str(tmp_path),
            "--store",
            "chroma",
            "--model",
            "test-model",
            "--batch-size",
            "16",
            "--collection",
            "BotAdvisorDocs",
        ]
    )

    assert exit_code == 0
    assert captured["input_path"] == str(tmp_path)
    assert captured["kwargs"] == {
        "store_type": "chroma",
        "model_name": "test-model",
        "batch_size": 16,
        "collection_name": "BotAdvisorDocs",
        "url": None,
    }


def test_run_returns_processing_summary(tmp_path):
    module = load_module()
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    (input_dir / "first.json").write_text("[]", encoding="utf-8")
    (input_dir / "second.json").write_text("[]", encoding="utf-8")

    script = module.EmbedUpsertScript.__new__(module.EmbedUpsertScript)
    script.total_nodes_processed = 0
    script.total_batches = 0
    script.successful_batches = 0
    script.failed_batches = 0

    def process_file(file_path: Path):
        script.total_nodes_processed += 3
        script.total_batches += 1
        script.successful_batches += 1

    script.process_file = process_file

    summary = module.EmbedUpsertScript.run(script, str(input_dir))

    assert summary == {
        "files_found": 2,
        "nodes_processed": 6,
        "total_batches": 2,
        "successful_batches": 2,
        "failed_batches": 0,
    }


def test_init_vector_store_uses_weaviate_v4_client(monkeypatch):
    module = load_module()
    calls: dict[str, object] = {}

    class FakeVectorStore:
        def __init__(self, *, weaviate_client, index_name):
            calls["client"] = weaviate_client
            calls["index_name"] = index_name

    def fake_connect_to_local(*, host, port, grpc_port):
        calls["connect_to_local"] = {
            "host": host,
            "port": port,
            "grpc_port": grpc_port,
        }
        return "weaviate-client"

    monkeypatch.setattr(module, "WeaviateVectorStore", FakeVectorStore, raising=False)
    monkeypatch.setattr(
        module,
        "weaviate",
        type("FakeWeaviate", (), {"connect_to_local": staticmethod(fake_connect_to_local)})(),
        raising=False,
    )

    script = module.EmbedUpsertScript.__new__(module.EmbedUpsertScript)
    script.store_type = "weaviate"
    script.collection_name = "BotAdvisorDocs"
    script.url = None

    vector_store = module.EmbedUpsertScript._init_vector_store(script)

    assert isinstance(vector_store, FakeVectorStore)
    assert calls == {
        "connect_to_local": {
            "host": "localhost",
            "port": 8080,
            "grpc_port": 50051,
        },
        "client": "weaviate-client",
        "index_name": "BotAdvisorDocs",
    }
