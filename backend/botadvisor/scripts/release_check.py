"""Canonical release-readiness check for the BotAdvisor runtime."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from botadvisor.app.dev.local_stack import (
    bootstrap_local_vector_store,
    spawn_local_api_server,
    start_local_dependencies,
    stop_local_api_server,
)
from botadvisor.app.dev.smoke import evaluate_health_response


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the release-readiness check."""
    parser = argparse.ArgumentParser(description="Run canonical BotAdvisor release-readiness checks.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the BotAdvisor API.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the BotAdvisor API.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout in seconds for the health check.")
    parser.add_argument(
        "--boot-smoke",
        action="store_true",
        help="Start local dependencies and the API server before running the readiness check.",
    )
    return parser


def execute_release_check(
    *, base_url: str, timeout_seconds: float, http_get: Callable[..., Any] = httpx.get
) -> int:
    """Run the canonical release-readiness check against a running BotAdvisor API."""
    response = http_get(f"{base_url.rstrip('/')}/health", timeout=timeout_seconds)
    result = evaluate_health_response(status_code=response.status_code, payload=response.json())
    print(result.summary)
    return 0 if result.passed else 1


def resolve_project_root() -> Path:
    """Return the backend project root for release-readiness commands."""
    return Path(__file__).resolve().parents[2]


def wait_for_runtime_ready(
    *,
    base_url: str,
    timeout_seconds: float,
    release_check_runner: Callable[..., int] = execute_release_check,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll the runtime until readiness passes or the timeout is reached."""
    deadline = monotonic() + timeout_seconds
    while True:
        exit_code = release_check_runner(base_url=base_url, timeout_seconds=timeout_seconds)
        if exit_code == 0:
            return 0
        if monotonic() >= deadline:
            return 1
        sleep(0.5)


def execute_boot_smoke_check(
    *,
    project_root: str | Path,
    host: str,
    port: int,
    timeout_seconds: float,
    start_dependencies: Callable[[Path], None] = start_local_dependencies,
    bootstrap_vector_store: Callable[[Path], None] = bootstrap_local_vector_store,
    spawn_api_server: Callable[..., Any] = spawn_local_api_server,
    wait_for_ready: Callable[..., int] = wait_for_runtime_ready,
    stop_api_server: Callable[[Any], None] = stop_local_api_server,
) -> int:
    """Validate the local boot flow end-to-end for the canonical runtime."""
    project_root = Path(project_root)
    start_dependencies(project_root)
    bootstrap_vector_store(project_root)
    process = spawn_api_server(project_root=project_root, host=host, port=port)
    try:
        return wait_for_ready(base_url=f"http://{host}:{port}", timeout_seconds=timeout_seconds)
    finally:
        stop_api_server(process)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the release-readiness CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.boot_smoke:
        return execute_boot_smoke_check(
            project_root=resolve_project_root(),
            host=args.host,
            port=args.port,
            timeout_seconds=args.timeout,
        )
    return execute_release_check(
        base_url=f"http://{args.host}:{args.port}",
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
