from __future__ import annotations

"""
Lightweight compatibility helpers for optional dependencies.

Keep optional import logic out of business logic modules and avoid a
generic "utils" bucket by colocating these in core.
"""

from typing import Any, Optional


def has_module(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def optional_import(module: str) -> Optional[Any]:  # type: ignore[no-untyped-def]
    try:
        return __import__(module)
    except Exception:
        return None

