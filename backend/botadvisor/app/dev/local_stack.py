"""Local development stack control for canonical BotAdvisor workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def compose_file_path(project_root: Path) -> Path:
    """Return the canonical docker compose file for local dependencies."""
    return project_root / "botadvisor" / "docker-compose.yaml"


def start_local_dependencies(project_root: Path) -> None:
    """Start the local dependency stack required by BotAdvisor."""
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file_path(project_root)),
            "up",
            "-d",
            "weaviate",
        ],
        check=True,
        cwd=project_root,
    )


def stop_local_dependencies(project_root: Path) -> None:
    """Stop the local dependency stack started for BotAdvisor development."""
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file_path(project_root)),
            "down",
        ],
        check=True,
        cwd=project_root,
    )


def bootstrap_local_vector_store(project_root: Path) -> None:
    """Initialize the canonical local vector store schema."""
    subprocess.run(
        [sys.executable, "-m", "botadvisor.scripts.setup_vector_store"],
        check=True,
        cwd=project_root,
    )


def run_local_api_server(project_root: Path, *, host: str, port: int, reload_enabled: bool) -> None:
    """Replace the current process with the canonical development API server."""
    os.chdir(project_root)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "botadvisor.app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload_enabled:
        command.append("--reload")

    os.execv(sys.executable, command)
