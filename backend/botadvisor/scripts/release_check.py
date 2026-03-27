"""Canonical release-readiness check for the BotAdvisor runtime."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Sequence

import httpx

from botadvisor.app.dev.smoke import evaluate_health_response


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the release-readiness check."""
    parser = argparse.ArgumentParser(description="Run canonical BotAdvisor release-readiness checks.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the BotAdvisor API.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the BotAdvisor API.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout in seconds for the health check.")
    return parser


def execute_release_check(
    *, base_url: str, timeout_seconds: float, http_get: Callable[..., Any] = httpx.get
) -> int:
    """Run the canonical release-readiness check against a running BotAdvisor API."""
    response = http_get(f"{base_url.rstrip('/')}/health", timeout=timeout_seconds)
    result = evaluate_health_response(status_code=response.status_code, payload=response.json())
    print(result.summary)
    return 0 if result.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the release-readiness CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return execute_release_check(
        base_url=f"http://{args.host}:{args.port}",
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
