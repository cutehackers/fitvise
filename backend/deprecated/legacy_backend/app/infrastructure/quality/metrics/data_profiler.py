"""Simple data profiler utilities.

Provides basic counts and ratios useful in Phase 1 quality checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Profile:
    total_characters: int
    total_words: int
    total_sentences: int


def profile_text(text: str) -> Profile:
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    return Profile(total_characters=len(text), total_words=len(words), total_sentences=len(sentences))

