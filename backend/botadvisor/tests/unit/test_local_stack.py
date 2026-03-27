from __future__ import annotations

import sys
from pathlib import Path


def test_start_local_dependencies_uses_canonical_compose_file(monkeypatch):
    from botadvisor.app.dev.local_stack import start_local_dependencies

    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, *, check, cwd):
        calls.append((command, cwd))

    monkeypatch.setattr("subprocess.run", fake_run)

    project_root = Path("/tmp/backend")
    start_local_dependencies(project_root)

    assert calls == [
        (
            [
                "docker",
                "compose",
                "-f",
                str(project_root / "botadvisor" / "docker-compose.yaml"),
                "up",
                "-d",
                "weaviate",
            ],
            project_root,
        )
    ]


def test_bootstrap_local_vector_store_runs_module_command(monkeypatch):
    from botadvisor.app.dev.local_stack import bootstrap_local_vector_store

    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, *, check, cwd):
        calls.append((command, cwd))

    monkeypatch.setattr("subprocess.run", fake_run)

    project_root = Path("/tmp/backend")
    bootstrap_local_vector_store(project_root)

    assert calls == [
        (
            [sys.executable, "-m", "botadvisor.scripts.setup_vector_store"],
            project_root,
        )
    ]


def test_run_local_api_server_executes_uvicorn_module(monkeypatch, tmp_path):
    from botadvisor.app.dev.local_stack import run_local_api_server

    calls: list[tuple[list[str], Path]] = []

    def fake_execv(executable, command):
        calls.append((command, Path.cwd()))
        raise SystemExit(0)

    monkeypatch.setattr("os.execv", fake_execv)

    project_root = tmp_path / "backend"
    project_root.mkdir()
    monkeypatch.chdir(project_root)

    try:
        run_local_api_server(project_root, host="0.0.0.0", port=9000, reload_enabled=False)
    except SystemExit:
        pass

    assert calls == [
        (
            [
                sys.executable,
                "-m",
                "uvicorn",
                "botadvisor.app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
            ],
            project_root,
        )
    ]


def test_dev_script_up_runs_startup_sequence(monkeypatch):
    from botadvisor.scripts import dev

    calls: list[tuple[str, Path]] = []
    project_root = Path("/tmp/backend")

    monkeypatch.setattr(dev, "resolve_project_root", lambda: project_root)
    monkeypatch.setattr(dev, "start_local_dependencies", lambda root: calls.append(("deps", root)))
    monkeypatch.setattr(dev, "bootstrap_local_vector_store", lambda root: calls.append(("bootstrap", root)))
    monkeypatch.setattr(
        dev,
        "run_local_api_server",
        lambda root, host, port, reload_enabled: calls.append(("api", root)),
    )

    exit_code = dev.main(["up"])

    assert exit_code == 0
    assert calls == [("deps", project_root), ("bootstrap", project_root), ("api", project_root)]


def test_dev_script_release_check_delegates_to_release_command(monkeypatch):
    from botadvisor.scripts import dev

    calls: list[tuple[str, float]] = []

    monkeypatch.setattr(
        dev,
        "execute_release_check",
        lambda *, base_url, timeout_seconds: calls.append((base_url, timeout_seconds)) or 0,
    )

    exit_code = dev.main(["release-check", "--host", "0.0.0.0", "--port", "9000", "--timeout", "3.5"])

    assert exit_code == 0
    assert calls == [("http://0.0.0.0:9000", 3.5)]
