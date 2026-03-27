"""Canonical local development entrypoint for BotAdvisor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from botadvisor.app.dev.local_stack import (
    bootstrap_local_vector_store,
    run_local_api_server,
    start_local_dependencies,
    stop_local_dependencies,
)


def resolve_project_root() -> Path:
    """Return the backend project root for canonical local commands."""
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the local development command."""
    parser = argparse.ArgumentParser(description="Run canonical BotAdvisor local development workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    up_parser = subparsers.add_parser("up", help="Start dependencies, bootstrap Weaviate, and run the API server.")
    up_parser.add_argument("--host", default="127.0.0.1", help="Host for the local API server.")
    up_parser.add_argument("--port", type=int, default=8000, help="Port for the local API server.")
    up_parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn reload mode.")
    up_parser.add_argument("--skip-compose", action="store_true", help="Skip docker compose startup.")
    up_parser.add_argument("--skip-bootstrap", action="store_true", help="Skip vector store bootstrap.")

    subparsers.add_parser("down", help="Stop local BotAdvisor dependencies.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the canonical local development workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = resolve_project_root()

    if args.command == "down":
        stop_local_dependencies(project_root)
        return 0

    if not args.skip_compose:
        start_local_dependencies(project_root)
    if not args.skip_bootstrap:
        bootstrap_local_vector_store(project_root)

    run_local_api_server(project_root, host=args.host, port=args.port, reload_enabled=not args.no_reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
