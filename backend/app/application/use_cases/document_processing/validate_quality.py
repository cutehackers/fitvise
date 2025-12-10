"""Data quality validation use case (Task 1.3.4).

Primary: Great Expectations-based validation when installed.
Fallback: Internal validation using domain QualityThresholds and
DataQualityMetrics with content profiling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.domain.value_objects.quality_metrics import (
    ContentQualityMetrics,
    DataQualityMetrics,
    QualityThresholds,
    ValidationResult,
    ValidationRule,
)


def _try_import_great_expectations():
    try:
        import great_expectations as ge  # type: ignore

        return ge
    except Exception:
        return None


def _try_import_langdetect():
    try:
        from langdetect import detect, detect_langs  # type: ignore

        return detect, detect_langs
    except Exception:
        return None, None


def _try_import_textstat():
    try:
        import textstat  # type: ignore

        return textstat
    except Exception:
        return None


def _profile_content(text: str) -> ContentQualityMetrics:
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    detect, detect_langs = _try_import_langdetect()
    language = None
    language_conf = None
    if detect is not None and detect_langs is not None:
        try:
            language = detect(text)
            lang_scores = detect_langs(text)
            language_conf = float(lang_scores[0].prob) if lang_scores else None
        except Exception:
            language = None
            language_conf = None
    textstat = _try_import_textstat()
    readability = None
    if textstat is not None:
        try:
            readability = float(textstat.flesch_reading_ease(text))
        except Exception:
            readability = None
    return ContentQualityMetrics(
        total_characters=len(text),
        total_words=len(words),
        total_sentences=len([s for s in sentences if s.strip()]),
        total_paragraphs=len(paragraphs),
        detected_language=language,
        language_confidence=language_conf,
        readability_score=readability,
        has_title=bool(re.search(r"^#+\s+\S", text, flags=re.MULTILINE))
        or bool(re.search(r"^\s*title\s*:\s*\S+", text, flags=re.IGNORECASE | re.MULTILINE)),
        has_headings=bool(re.search(r"^#{1,6}\s+\S", text, flags=re.MULTILINE)),
        has_tables="|" in text or "+-" in text,
        has_images="![](" in text or "![" in text,
        has_links="[" in text and "](" in text,
        empty_sections=len([m.group(0) for m in re.finditer(r"\n\s*\n", text)]),
        broken_links=0,
        missing_metadata_fields=0,
        encoding_issues=0,
        format_errors=0,
    )


@dataclass
class QualityReport:
    success: bool
    overall_score: float
    quality_level: str
    metrics: Dict[str, Any]
    validations: List[Dict[str, Any]]
    ge_report: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "overall_score": self.overall_score,
            "quality_level": self.quality_level,
            "metrics": self.metrics,
            "validations": self.validations,
            "ge_report": self.ge_report,
        }


class ValidateQualityRequest:
    def __init__(self, texts: Sequence[str], thresholds: Optional[QualityThresholds] = None) -> None:
        self.texts = list(texts)
        self.thresholds = thresholds or QualityThresholds()


class ValidateQualityResponse:
    def __init__(self, reports: List[QualityReport], environment: Dict[str, Any]) -> None:
        self.reports = reports
        self.environment = environment

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reports": [r.as_dict() for r in self.reports],
            "environment": self.environment,
        }


class ValidateQualityUseCase:
    def __init__(self) -> None:
        self._ge = _try_import_great_expectations()

    async def execute(self, request: ValidateQualityRequest) -> ValidateQualityResponse:
        reports: List[QualityReport] = []

        profiles: List[ContentQualityMetrics] = [_profile_content(t) for t in request.texts]
        ge_report = self._run_great_expectations_if_possible(profiles, request.thresholds)

        for profile in profiles:
            validations: List[ValidationResult] = [
                ValidationResult(
                    rule=ValidationRule.COMPLETENESS,
                    passed=profile.total_words >= request.thresholds.min_word_count,
                    score=min(1.0, profile.total_words / max(1, request.thresholds.min_word_count)),
                    message="Minimum word count",
                ),
                ValidationResult(
                    rule=ValidationRule.VALIDITY,
                    passed=profile.total_characters >= request.thresholds.min_character_count,
                    score=min(1.0, profile.total_characters / max(1, request.thresholds.min_character_count)),
                    message="Minimum character count",
                ),
                ValidationResult(
                    rule=ValidationRule.CONSISTENCY,
                    passed=profile.empty_sections < 20,
                    score=max(0.0, 1.0 - (profile.empty_sections / 100.0)),
                    message="Reasonable structure",
                ),
            ]

            metrics = DataQualityMetrics(
                measured_at=datetime.now(timezone.utc),
                validation_results=validations,
                content_metrics=profile,
            )

            reports.append(
                QualityReport(
                    success=request.thresholds.is_acceptable_quality(metrics),
                    overall_score=metrics.overall_quality_score,
                    quality_level=metrics.quality_level.value,
                    metrics={
                        "total_words": profile.total_words,
                        "total_characters": profile.total_characters,
                        "total_sentences": profile.total_sentences,
                        "total_paragraphs": profile.total_paragraphs,
                        "language": profile.detected_language,
                        "language_confidence": profile.language_confidence,
                        "readability_score": profile.readability_score,
                    },
                    validations=[
                        {
                            "rule": v.rule.value,
                            "passed": v.passed,
                            "score": v.score,
                            "message": v.message,
                        }
                        for v in validations
                    ],
                    ge_report=ge_report,
                )
            )

        env = {"great_expectations_available": self._ge is not None}
        return ValidateQualityResponse(reports=reports, environment=env)

    def _run_great_expectations_if_possible(
        self, profiles: List[ContentQualityMetrics], thresholds: QualityThresholds
    ) -> Optional[Dict[str, Any]]:
        if self._ge is None:
            return None
        try:
            df = pd.DataFrame(
                [
                    {
                        "total_words": p.total_words,
                        "total_characters": p.total_characters,
                        "readability": p.readability_score if p.readability_score is not None else -1.0,
                    }
                    for p in profiles
                ]
            )
            ge_df = self._ge.from_pandas(df)  # type: ignore[attr-defined]
            results = []
            results.append(
                ge_df.expect_column_min_to_be_between(
                    column="total_words", min_value=thresholds.min_word_count
                ).to_json_dict()
            )
            results.append(
                ge_df.expect_column_min_to_be_between(
                    column="total_characters", min_value=thresholds.min_character_count
                ).to_json_dict()
            )
            summary = {
                "success": all(r.get("success", False) for r in results if isinstance(r, dict)),
                "results": results,
            }
            return summary
        except Exception:
            return None

