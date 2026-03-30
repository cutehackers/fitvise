from __future__ import annotations

import json
from pathlib import Path

import pytest

from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationResult


def test_summarize_retrieval_results_reports_failed_cases_and_recall():
    from botadvisor.scripts.evaluate_retrieval import summarize_retrieval_results

    summary = summarize_retrieval_results(
        [
            RetrievalEvaluationResult(
                case_id="case-1",
                relevance_hit=True,
                citation_integrity_hit=True,
                metadata_integrity_hit=True,
                expected_doc_recall=1.0,
                diagnostic_notes=(),
            ),
            RetrievalEvaluationResult(
                case_id="case-2",
                relevance_hit=True,
                citation_integrity_hit=False,
                metadata_integrity_hit=True,
                expected_doc_recall=0.5,
                diagnostic_notes=("missing citation metadata",),
            ),
        ]
    )

    assert summary == {
        "total_cases": 2,
        "passed_cases": 1,
        "failed_case_ids": ["case-2"],
        "pass_rate": 0.5,
        "average_expected_doc_recall": 0.75,
    }


def test_execute_retrieval_evaluation_writes_artifacts(tmp_path: Path):
    from botadvisor.scripts.evaluate_retrieval import execute_retrieval_evaluation

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_cases": 1,
                    "passed_cases": 1,
                    "failed_case_ids": [],
                    "pass_rate": 1.0,
                    "average_expected_doc_recall": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = execute_retrieval_evaluation(
        artifact_dir=tmp_path,
        baseline_path=baseline_path,
        run_label="test-run",
        evaluator=lambda: [
            RetrievalEvaluationResult(
                case_id="case-1",
                relevance_hit=True,
                citation_integrity_hit=True,
                metadata_integrity_hit=True,
                expected_doc_recall=1.0,
                diagnostic_notes=(),
            )
        ],
    )

    assert exit_code == 0
    assert (tmp_path / "test-run.json").exists()
    assert (tmp_path / "test-run.md").exists()

    payload = json.loads((tmp_path / "test-run.json").read_text(encoding="utf-8"))
    assert payload["summary"]["passed_cases"] == 1
    assert payload["results"][0]["case_id"] == "case-1"
    assert "failed cases: none" in (tmp_path / "test-run.md").read_text(encoding="utf-8").lower()


def test_execute_retrieval_evaluation_returns_one_for_baseline_regression(tmp_path: Path):
    from botadvisor.scripts.evaluate_retrieval import execute_retrieval_evaluation

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_cases": 1,
                    "passed_cases": 1,
                    "failed_case_ids": [],
                    "pass_rate": 1.0,
                    "average_expected_doc_recall": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = execute_retrieval_evaluation(
        artifact_dir=tmp_path,
        baseline_path=baseline_path,
        run_label="regression-run",
        evaluator=lambda: [
            RetrievalEvaluationResult(
                case_id="case-1",
                relevance_hit=True,
                citation_integrity_hit=False,
                metadata_integrity_hit=True,
                expected_doc_recall=0.5,
                diagnostic_notes=("citation missing",),
            )
        ],
    )

    assert exit_code == 1


def test_evaluate_retrieval_help_exits_cleanly():
    from botadvisor.scripts.evaluate_retrieval import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
