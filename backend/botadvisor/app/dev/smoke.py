"""Runtime smoke evaluation for canonical BotAdvisor release checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REQUIRED_CHECKS = ("retrieval", "vector_store", "llm_path")
REQUIRED_FIELDS = ("status", "service", "retrieval_available", "langfuse_enabled", "checks")


@dataclass(frozen=True)
class SmokeRuntimeResult:
    """Result of evaluating the canonical runtime health payload."""

    passed: bool
    summary: str
    details: dict[str, str]


def evaluate_health_response(*, status_code: int, payload: Mapping[str, Any]) -> SmokeRuntimeResult:
    """Evaluate whether the canonical runtime health response is release-ready."""
    details = {"http_status": str(status_code)}

    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    if status_code != 200 or missing_fields:
        if missing_fields:
            details["missing_fields"] = ",".join(missing_fields)
        return SmokeRuntimeResult(passed=False, summary="runtime is not ready", details=details)

    checks = payload.get("checks", {})
    for name in REQUIRED_CHECKS:
        check_status = str(checks.get(name, {}).get("status", "missing"))
        details[name] = check_status

    runtime_ready = str(payload.get("status")) == "healthy" and all(
        details[name] == "healthy" for name in REQUIRED_CHECKS
    )
    return SmokeRuntimeResult(
        passed=runtime_ready,
        summary="runtime is ready" if runtime_ready else "runtime is not ready",
        details=details,
    )
