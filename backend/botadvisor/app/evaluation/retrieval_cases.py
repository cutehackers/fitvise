"""Contracts for canonical retrieval evaluation cases and results."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def _normalize_expected_values(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized_values = tuple(value.strip() for value in values if value.strip())
    return normalized_values


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """A single hand-reviewed retrieval case from the canonical gold set."""

    case_id: str
    query: str
    top_k: int = 5
    platform: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    expected_doc_ids: tuple[str, ...] = ()
    expected_source_ids: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    must_include_text: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        query = self.query.strip()

        if not case_id:
            raise ValueError("case_id must not be blank")
        if not query:
            raise ValueError("query must not be blank")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

        normalized_doc_ids = _normalize_expected_values(self.expected_doc_ids)
        normalized_source_ids = _normalize_expected_values(self.expected_source_ids)
        normalized_sections = _normalize_expected_values(self.expected_sections)
        normalized_required_text = _normalize_expected_values(self.must_include_text)

        if not (
            normalized_doc_ids
            or normalized_source_ids
            or normalized_sections
            or normalized_required_text
        ):
            raise ValueError("at least one expected retrieval signal is required")

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "platform", self.platform.strip() if self.platform else None)
        object.__setattr__(self, "expected_doc_ids", normalized_doc_ids)
        object.__setattr__(self, "expected_source_ids", normalized_source_ids)
        object.__setattr__(self, "expected_sections", normalized_sections)
        object.__setattr__(self, "must_include_text", normalized_required_text)
        object.__setattr__(self, "notes", self.notes.strip())


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Case-level retrieval evaluation outcome."""

    case_id: str
    relevance_hit: bool
    citation_integrity_hit: bool
    metadata_integrity_hit: bool
    expected_doc_recall: float
    diagnostic_notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all required retrieval checks passed."""
        return (
            self.relevance_hit
            and self.citation_integrity_hit
            and self.metadata_integrity_hit
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the case-level result to a JSON-serializable dictionary."""
        return {
            "case_id": self.case_id,
            "relevance_hit": self.relevance_hit,
            "citation_integrity_hit": self.citation_integrity_hit,
            "metadata_integrity_hit": self.metadata_integrity_hit,
            "expected_doc_recall": self.expected_doc_recall,
            "diagnostic_notes": list(self.diagnostic_notes),
            "passed": self.passed,
        }


def load_retrieval_evaluation_cases(cases_path: Path) -> list[RetrievalEvaluationCase]:
    """Load canonical retrieval evaluation cases from a checked-in JSON file."""
    raw_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    return [RetrievalEvaluationCase(**raw_case) for raw_case in raw_payload]
