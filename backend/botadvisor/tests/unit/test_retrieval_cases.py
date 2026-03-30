from __future__ import annotations

import pytest


def test_retrieval_evaluation_case_requires_explicit_expectations():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase

    case = RetrievalEvaluationCase(
        case_id="filesystem-protein-intake",
        query="protein intake basics",
        top_k=4,
        platform="filesystem",
        expected_doc_ids=("doc-1",),
        expected_source_ids=("source-1",),
        expected_sections=("section-a",),
        must_include_text=("protein",),
    )

    assert case.case_id == "filesystem-protein-intake"
    assert case.query == "protein intake basics"
    assert case.top_k == 4
    assert case.platform == "filesystem"
    assert case.expected_doc_ids == ("doc-1",)
    assert case.expected_source_ids == ("source-1",)
    assert case.expected_sections == ("section-a",)
    assert case.must_include_text == ("protein",)


def test_retrieval_evaluation_case_rejects_empty_identity_fields():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase

    with pytest.raises(ValueError, match="case_id must not be blank"):
        RetrievalEvaluationCase(case_id="", query="protein intake")

    with pytest.raises(ValueError, match="query must not be blank"):
        RetrievalEvaluationCase(case_id="case-1", query="")


def test_retrieval_evaluation_case_rejects_missing_expectations():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase

    with pytest.raises(ValueError, match="at least one expected retrieval signal"):
        RetrievalEvaluationCase(case_id="case-1", query="protein intake")


def test_retrieval_evaluation_case_rejects_invalid_top_k():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        RetrievalEvaluationCase(
            case_id="case-1",
            query="protein intake",
            top_k=0,
            expected_doc_ids=("doc-1",),
        )


def test_retrieval_evaluation_result_passes_only_when_all_required_checks_pass():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationResult

    result = RetrievalEvaluationResult(
        case_id="filesystem-protein-intake",
        relevance_hit=True,
        citation_integrity_hit=True,
        metadata_integrity_hit=True,
        expected_doc_recall=1.0,
        diagnostic_notes=(),
    )

    assert result.passed is True

    failing_result = RetrievalEvaluationResult(
        case_id="filesystem-protein-intake",
        relevance_hit=True,
        citation_integrity_hit=False,
        metadata_integrity_hit=True,
        expected_doc_recall=1.0,
        diagnostic_notes=("missing section metadata",),
    )

    assert failing_result.passed is False
