from __future__ import annotations

from pathlib import Path


def test_evaluate_health_response_accepts_healthy_payload():
    from botadvisor.app.dev.smoke import evaluate_health_response

    result = evaluate_health_response(
        status_code=200,
        payload={
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
    )

    assert result.passed is True
    assert result.summary == "runtime is ready"
    assert result.details["http_status"] == "200"


def test_evaluate_health_response_fails_when_required_check_is_degraded():
    from botadvisor.app.dev.smoke import evaluate_health_response

    result = evaluate_health_response(
        status_code=200,
        payload={
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
    )

    assert result.passed is False
    assert result.summary == "runtime is not ready"
    assert result.details["vector_store"] == "degraded"


def test_smoke_module_does_not_pull_in_cli_or_process_control():
    smoke_path = Path(__file__).resolve().parents[2] / "app" / "dev" / "smoke.py"

    source = smoke_path.read_text(encoding="utf-8")

    assert "import argparse" not in source
    assert "import subprocess" not in source
    assert "os.execv" not in source
