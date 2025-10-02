"""Simple schema validation helpers.

For Phase 1, we only provide minimal checks and let higher-level
use cases transform to DataQualityMetrics for evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, List


def validate_required_keys(payload: Dict[str, Any], required: List[str]) -> List[str]:
    missing = []
    for key in required:
        if key not in payload or payload[key] in (None, ""):
            missing.append(key)
    return missing

