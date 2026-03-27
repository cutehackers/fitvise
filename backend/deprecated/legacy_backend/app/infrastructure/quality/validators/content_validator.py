"""Content validation helpers.

Light heuristics to surface potential issues in text content.
"""
from __future__ import annotations

import re
from typing import Dict


def basic_content_flags(text: str) -> Dict[str, int]:
    return {
        "empty_sections": len([m.group(0) for m in re.finditer(r"\n\s*\n", text)]),
        "suspicious_char_ratio": int((len(re.findall(r"[^\x09\x0A\x0D\x20-\x7E]", text)) / max(1, len(text))) * 100),
    }

