"""Pure retrieval-evaluation rules for the canonical quality gate."""

from __future__ import annotations

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.evaluation.retrieval_cases import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
)


def evaluate_retrieval_case(
    case: RetrievalEvaluationCase,
    retrieved_chunks: list[Chunk],
) -> RetrievalEvaluationResult:
    """Evaluate retrieved chunks against one canonical retrieval case."""
    metadata_valid_chunks = [
        chunk
        for chunk in retrieved_chunks
        if _chunk_matches_requested_metadata(case, chunk)
    ]

    matched_source_ids = {
        chunk.metadata.source_id
        for chunk in retrieved_chunks
        if chunk.metadata.source_id in case.expected_source_ids
    }
    matched_doc_ids = {
        chunk.metadata.doc_id
        for chunk in retrieved_chunks
        if chunk.metadata.doc_id in case.expected_doc_ids
    }
    matched_sections = {
        chunk.metadata.section
        for chunk in retrieved_chunks
        if chunk.metadata.section in case.expected_sections
    }
    matched_required_text = {
        required_text
        for required_text in case.must_include_text
        if any(required_text in chunk.content for chunk in retrieved_chunks)
    }

    relevance_hit = bool(
        matched_source_ids
        or matched_doc_ids
        or matched_sections
        or matched_required_text
    )

    matched_chunks = [
        chunk
        for chunk in retrieved_chunks
        if (
            chunk.metadata.source_id in matched_source_ids
            or chunk.metadata.doc_id in matched_doc_ids
            or chunk.metadata.section in matched_sections
            or any(required_text in chunk.content for required_text in matched_required_text)
        )
    ]

    citation_integrity_hit = bool(matched_chunks) and all(
        chunk.metadata.doc_id
        and chunk.metadata.source_id
        and chunk.metadata.platform
        and chunk.metadata.section
        for chunk in matched_chunks
    )

    metadata_integrity_hit = _chunks_match_requested_metadata(case, retrieved_chunks)

    expected_doc_recall = _calculate_expected_doc_recall(
        case=case,
        matched_source_ids={
            chunk.metadata.source_id
            for chunk in metadata_valid_chunks
            if chunk.metadata.source_id in case.expected_source_ids
        },
        matched_doc_ids={
            chunk.metadata.doc_id
            for chunk in metadata_valid_chunks
            if chunk.metadata.doc_id in case.expected_doc_ids
        },
        matched_sections={
            chunk.metadata.section
            for chunk in metadata_valid_chunks
            if chunk.metadata.section in case.expected_sections
        },
        matched_required_text={
            required_text
            for required_text in case.must_include_text
            if any(required_text in chunk.content for chunk in metadata_valid_chunks)
        },
    )

    diagnostic_notes: list[str] = []
    if not relevance_hit:
        diagnostic_notes.append("no expected retrieval signal matched the returned chunks")
    if relevance_hit and not citation_integrity_hit:
        diagnostic_notes.append("matched retrieval results are missing citation metadata")
    if not metadata_integrity_hit:
        diagnostic_notes.append("returned chunks violate the requested platform or filter metadata")

    return RetrievalEvaluationResult(
        case_id=case.case_id,
        relevance_hit=relevance_hit,
        citation_integrity_hit=citation_integrity_hit,
        metadata_integrity_hit=metadata_integrity_hit,
        expected_doc_recall=expected_doc_recall,
        diagnostic_notes=tuple(diagnostic_notes),
    )


def _chunks_match_requested_metadata(
    case: RetrievalEvaluationCase,
    retrieved_chunks: list[Chunk],
) -> bool:
    return all(_chunk_matches_requested_metadata(case, chunk) for chunk in retrieved_chunks)


def _chunk_matches_requested_metadata(
    case: RetrievalEvaluationCase,
    chunk: Chunk,
) -> bool:
    if case.platform and chunk.metadata.platform != case.platform:
        return False

    for key, value in case.filters.items():
        chunk_value = getattr(chunk.metadata, key, None)
        if chunk_value != value:
            return False

    return True


def _calculate_expected_doc_recall(
    *,
    case: RetrievalEvaluationCase,
    matched_source_ids: set[str],
    matched_doc_ids: set[str],
    matched_sections: set[str | None],
    matched_required_text: set[str],
) -> float:
    if case.expected_doc_ids:
        return len(matched_doc_ids) / len(case.expected_doc_ids)
    if case.expected_source_ids:
        return len(matched_source_ids) / len(case.expected_source_ids)
    if case.expected_sections:
        return len({section for section in matched_sections if section}) / len(case.expected_sections)
    if case.must_include_text:
        return len(matched_required_text) / len(case.must_include_text)
    return 0.0
