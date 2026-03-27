from __future__ import annotations

from pathlib import Path


def test_backend_root_env_example_exists():
    project_root = Path(__file__).resolve().parents[3]

    assert (project_root / ".env.example").exists()
