from __future__ import annotations

import json
from pathlib import Path


def test_retrieval_gold_case_file_exists():
    evaluation_root = Path(__file__).resolve().parents[2] / "evaluation"

    assert (evaluation_root / "retrieval_gold_cases.json").exists()


def test_retrieval_gold_cases_are_small_unique_and_parseable():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase

    evaluation_root = Path(__file__).resolve().parents[2] / "evaluation"
    cases_path = evaluation_root / "retrieval_gold_cases.json"

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))

    assert isinstance(raw_cases, list)
    assert 8 <= len(raw_cases) <= 12

    case_ids = [case["case_id"] for case in raw_cases]
    assert len(case_ids) == len(set(case_ids))

    parsed_cases = [RetrievalEvaluationCase(**case) for case in raw_cases]

    assert all(parsed_case.case_id for parsed_case in parsed_cases)


def test_retrieval_gold_cases_reference_checked_in_corpus_files():
    evaluation_root = Path(__file__).resolve().parents[2] / "evaluation"
    cases_path = evaluation_root / "retrieval_gold_cases.json"
    corpus_root = evaluation_root / "corpus"

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    checked_in_source_ids = {
        f"evaluation_corpus/{path.name}"
        for path in corpus_root.glob("*.md")
    }

    assert len(checked_in_source_ids) >= 4

    for case in raw_cases:
        for source_id in case.get("expected_source_ids", []):
            assert source_id in checked_in_source_ids
