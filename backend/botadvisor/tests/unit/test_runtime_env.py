from __future__ import annotations

from pathlib import Path
import re


def test_backend_root_env_example_exists():
    project_root = Path(__file__).resolve().parents[3]

    assert (project_root / ".env.example").exists()


def test_backend_root_env_example_covers_all_runtime_settings():
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "botadvisor" / "app" / "core" / "config.py"
    env_example_path = project_root / ".env.example"

    settings_keys = set(re.findall(r'alias="([A-Z0-9_]+)"', config_path.read_text(encoding="utf-8")))
    example_keys = {
        line.split("=", 1)[0]
        for line in env_example_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }

    assert not (settings_keys - example_keys)
