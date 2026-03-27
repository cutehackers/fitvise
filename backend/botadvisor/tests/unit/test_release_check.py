from __future__ import annotations

from pathlib import Path

import pytest


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class FakeProcess:
    def __init__(self):
        self.terminated = False


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


def test_execute_boot_smoke_check_runs_boot_sequence_and_stops_api_server():
    from botadvisor.scripts.release_check import execute_boot_smoke_check

    events: list[tuple[str, object]] = []
    process = FakeProcess()

    exit_code = execute_boot_smoke_check(
        project_root="/tmp/backend",
        host="127.0.0.1",
        port=8000,
        timeout_seconds=5.0,
        start_dependencies=lambda root: events.append(("deps", root)),
        bootstrap_vector_store=lambda root: events.append(("bootstrap", root)),
        spawn_api_server=lambda *, project_root, host, port: events.append(("spawn", (project_root, host, port))) or process,
        wait_for_ready=lambda *, base_url, timeout_seconds: events.append(("ready", (base_url, timeout_seconds))) or 0,
        stop_api_server=lambda proc: events.append(("stop", proc)),
    )

    assert exit_code == 0
    assert events == [
        ("deps", Path("/tmp/backend")),
        ("bootstrap", Path("/tmp/backend")),
        ("spawn", (Path("/tmp/backend"), "127.0.0.1", 8000)),
        ("ready", ("http://127.0.0.1:8000", 5.0)),
        ("stop", process),
    ]


def test_execute_boot_smoke_check_stops_api_server_on_failed_readiness():
    from botadvisor.scripts.release_check import execute_boot_smoke_check

    process = FakeProcess()
    stopped: list[FakeProcess] = []

    exit_code = execute_boot_smoke_check(
        project_root="/tmp/backend",
        host="127.0.0.1",
        port=8000,
        timeout_seconds=5.0,
        start_dependencies=lambda root: None,
        bootstrap_vector_store=lambda root: None,
        spawn_api_server=lambda *, project_root, host, port: process,
        wait_for_ready=lambda *, base_url, timeout_seconds: 1,
        stop_api_server=lambda proc: stopped.append(proc),
    )

    assert exit_code == 1
    assert stopped == [process]
