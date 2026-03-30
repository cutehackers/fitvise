from __future__ import annotations

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document_metadata import DocumentMetadata


def create_retrieved_chunk(
    *,
    chunk_id: str,
    content: str,
    doc_id: str,
    source_id: str,
    platform: str,
    section: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata=DocumentMetadata(
            doc_id=doc_id,
            source_id=source_id,
            platform=platform,
            source_url=f"file:///{source_id}",
            page=None,
            section=section,
        ),
        score=0.9,
    )


def test_retrieval_evaluation_service_reports_relevance_and_recall():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase
    from botadvisor.app.evaluation.retrieval_service import evaluate_retrieval_case

    case = RetrievalEvaluationCase(
        case_id="protein-target-range",
        query="recommended protein intake",
        platform="filesystem",
        expected_source_ids=("evaluation_corpus/protein_intake_basics.md",),
        must_include_text=("1.6 to 2.2 grams per kilogram",),
    )

    result = evaluate_retrieval_case(
        case,
        [
            create_retrieved_chunk(
                chunk_id="chunk-1",
                content="Strength-focused athletes often target roughly 1.6 to 2.2 grams per kilogram.",
                doc_id="doc-1",
                source_id="evaluation_corpus/protein_intake_basics.md",
                platform="filesystem",
                section="chunk_0",
            )
        ],
    )

    assert result.relevance_hit is True
    assert result.citation_integrity_hit is True
    assert result.metadata_integrity_hit is True
    assert result.expected_doc_recall == 1.0
    assert result.passed is True


def test_retrieval_evaluation_service_flags_missing_citation_fields():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase
    from botadvisor.app.evaluation.retrieval_service import evaluate_retrieval_case

    case = RetrievalEvaluationCase(
        case_id="protein-target-range",
        query="recommended protein intake",
        platform="filesystem",
        expected_source_ids=("evaluation_corpus/protein_intake_basics.md",),
    )

    result = evaluate_retrieval_case(
        case,
        [
            create_retrieved_chunk(
                chunk_id="chunk-1",
                content="protein guidance",
                doc_id="",
                source_id="evaluation_corpus/protein_intake_basics.md",
                platform="filesystem",
                section="chunk_0",
            )
        ],
    )

    assert result.relevance_hit is True
    assert result.citation_integrity_hit is False
    assert result.passed is False


def test_retrieval_evaluation_service_flags_platform_metadata_leaks():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase
    from botadvisor.app.evaluation.retrieval_service import evaluate_retrieval_case

    case = RetrievalEvaluationCase(
        case_id="sleep-duration-range",
        query="hours of sleep for recovery",
        platform="filesystem",
        expected_source_ids=("evaluation_corpus/sleep_recovery.md",),
    )

    result = evaluate_retrieval_case(
        case,
        [
            create_retrieved_chunk(
                chunk_id="chunk-1",
                content="Many adults perform better with 7 to 9 hours of sleep.",
                doc_id="doc-1",
                source_id="evaluation_corpus/sleep_recovery.md",
                platform="web",
                section="chunk_0",
            )
        ],
    )

    assert result.relevance_hit is True
    assert result.metadata_integrity_hit is False
    assert result.passed is False


def test_retrieval_evaluation_service_checks_request_filter_integrity():
    from botadvisor.app.evaluation.retrieval_cases import RetrievalEvaluationCase
    from botadvisor.app.evaluation.retrieval_service import evaluate_retrieval_case

    case = RetrievalEvaluationCase(
        case_id="creatine-maintenance-dose",
        query="daily creatine dose",
        platform="filesystem",
        filters={"section": "chunk_0"},
        expected_source_ids=("evaluation_corpus/creatine_guidance.md",),
        expected_sections=("chunk_0",),
    )

    result = evaluate_retrieval_case(
        case,
        [
            create_retrieved_chunk(
                chunk_id="chunk-1",
                content="A common maintenance approach is 3 to 5 grams per day.",
                doc_id="doc-1",
                source_id="evaluation_corpus/creatine_guidance.md",
                platform="filesystem",
                section="chunk_1",
            )
        ],
    )

    assert result.relevance_hit is True
    assert result.metadata_integrity_hit is False
    assert result.expected_doc_recall == 0.0
    assert result.passed is False
