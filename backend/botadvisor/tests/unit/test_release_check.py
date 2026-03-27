from __future__ import annotations

import pytest


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


def test_execute_release_check_returns_zero_for_ready_runtime():
    from botadvisor.scripts.release_check import execute_release_check

    exit_code = execute_release_check(
        base_url="http://runtime.local",
        timeout_seconds=5.0,
        http_get=lambda url, timeout: FakeResponse(
            200,
            {
                "status": "healthy",
                "service": "botadvisor-api",
                "retrieval_available": True,
                "langfuse_enabled": False,
                "checks": {
                    "retrieval": {"status": "healthy"},
                    "vector_store": {"status": "healthy"},
                    "llm_path": {"status": "healthy"},
                },
            },
        ),
    )

    assert exit_code == 0


def test_execute_release_check_returns_one_for_degraded_runtime():
    from botadvisor.scripts.release_check import execute_release_check

    exit_code = execute_release_check(
        base_url="http://runtime.local",
        timeout_seconds=5.0,
        http_get=lambda url, timeout: FakeResponse(
            200,
            {
                "status": "degraded",
                "service": "botadvisor-api",
                "retrieval_available": True,
                "langfuse_enabled": False,
                "checks": {
                    "retrieval": {"status": "healthy"},
                    "vector_store": {"status": "degraded"},
                    "llm_path": {"status": "healthy"},
                },
            },
        ),
    )

    assert exit_code == 1


def test_release_check_help_exits_cleanly():
    from botadvisor.scripts.release_check import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
