from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("botadvisor_ingest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_accepts_explicit_paths_and_platform(monkeypatch, tmp_path):
    module = load_module()
    captured: dict[str, object] = {}
    input_file = tmp_path / "sample.txt"
    input_file.write_text("hello", encoding="utf-8")
    output_dir = tmp_path / "chunks"

    class FakeIngestor:
        def __init__(self):
            self.documents_processed = 1
            self.chunks_generated = 2
            self.duplicates_skipped = 0

        def process_file(self, file_path: Path, platform: str, out_dir: Path):
            captured["file_path"] = file_path
            captured["platform"] = platform
            captured["out_dir"] = out_dir

    monkeypatch.setattr(module, "SimpleIngestionScript", FakeIngestor)

    exit_code = module.main(
        [
            "--input",
            str(input_file),
            "--out",
            str(output_dir),
            "--platform",
            "filesystem",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "file_path": input_file,
        "platform": "filesystem",
        "out_dir": output_dir,
    }
