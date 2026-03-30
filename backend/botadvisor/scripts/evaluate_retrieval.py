#!/usr/bin/env python3
"""Canonical retrieval evaluation command for BotAdvisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from botadvisor.app.core.config import get_settings
from botadvisor.app.core.entity.retriever_request import RetrieverRequest
from botadvisor.app.evaluation.corpus import build_evaluation_chunks, write_evaluation_chunks
from botadvisor.app.evaluation.retrieval_cases import (
    RetrievalEvaluationResult,
    load_retrieval_evaluation_cases,
)
from botadvisor.app.evaluation.retrieval_service import evaluate_retrieval_case
from botadvisor.app.observability.logging import configure_logger, get_logger
from botadvisor.app.retrieval.config import RetrievalConfig
from botadvisor.app.retrieval.factory import create_hybrid_retriever
from botadvisor.scripts.embed_upsert import EmbedUpsertScript
from botadvisor.scripts.setup_vector_store import connect_to_weaviate, setup_vector_store


configure_logger()
logger = get_logger("evaluate_retrieval")


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for retrieval evaluation."""
    parser = argparse.ArgumentParser(description="Run canonical retrieval evaluation for BotAdvisor")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory where run artifacts are written. Defaults to backend/.tmp/retrieval_evaluation.",
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="Optional baseline JSON path. Defaults to the checked-in retrieval baseline.",
    )
    parser.add_argument(
        "--run-label",
        default="retrieval-evaluation",
        help="Artifact file prefix used for JSON and Markdown outputs.",
    )
    parser.add_argument(
        "--prepare-corpus",
        action="store_true",
        help="Reset the canonical collection and index the checked-in evaluation corpus before evaluation.",
    )
    return parser


def summarize_retrieval_results(results: list[RetrievalEvaluationResult]) -> dict[str, object]:
    """Summarize case-level retrieval results for artifact comparison."""
    total_cases = len(results)
    passed_cases = sum(result.passed for result in results)
    failed_case_ids = [result.case_id for result in results if not result.passed]
    average_expected_doc_recall = (
        sum(result.expected_doc_recall for result in results) / total_cases if total_cases else 0.0
    )
    pass_rate = passed_cases / total_cases if total_cases else 0.0

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_case_ids": failed_case_ids,
        "pass_rate": pass_rate,
        "average_expected_doc_recall": average_expected_doc_recall,
    }


def resolve_backend_root() -> Path:
    """Return the backend root directory for the canonical repository layout."""
    return Path(__file__).resolve().parents[2]


def resolve_evaluation_root(project_root: Path) -> Path:
    """Return the checked-in evaluation root."""
    return project_root / "botadvisor" / "evaluation"


def resolve_artifact_dir(project_root: Path, requested_artifact_dir: str | None) -> Path:
    """Resolve the output directory used for ad hoc retrieval evaluation artifacts."""
    if requested_artifact_dir:
        return Path(requested_artifact_dir)
    return project_root / ".tmp" / "retrieval_evaluation"


def resolve_baseline_path(project_root: Path, requested_baseline_path: str | None) -> Path:
    """Resolve the baseline artifact path for retrieval regression comparison."""
    if requested_baseline_path:
        return Path(requested_baseline_path)
    return resolve_evaluation_root(project_root) / "baseline" / "retrieval_baseline.json"


def _write_markdown_summary(
    *,
    markdown_path: Path,
    run_label: str,
    summary: dict[str, object],
    results: list[RetrievalEvaluationResult],
) -> None:
    failed_case_ids = summary["failed_case_ids"]
    failed_case_line = ", ".join(failed_case_ids) if failed_case_ids else "none"
    lines = [
        f"# Retrieval Evaluation Summary: {run_label}",
        "",
        f"- total cases: {summary['total_cases']}",
        f"- passed cases: {summary['passed_cases']}",
        f"- failed cases: {failed_case_line}",
        f"- pass rate: {summary['pass_rate']:.2f}",
        f"- average expected doc recall: {summary['average_expected_doc_recall']:.2f}",
        "",
        "## Case Results",
        "",
    ]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"- `{result.case_id}`: {status}, recall={result.expected_doc_recall:.2f}, "
            f"relevance={result.relevance_hit}, citation={result.citation_integrity_hit}, "
            f"metadata={result.metadata_integrity_hit}"
        )
        for note in result.diagnostic_notes:
            lines.append(f"  note: {note}")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retrieval_artifacts(
    *,
    artifact_dir: Path,
    run_label: str,
    summary: dict[str, object],
    results: list[RetrievalEvaluationResult],
) -> tuple[Path, Path]:
    """Write JSON and Markdown artifacts for a retrieval evaluation run."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / f"{run_label}.json"
    markdown_path = artifact_dir / f"{run_label}.md"

    payload = {
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown_summary(markdown_path=markdown_path, run_label=run_label, summary=summary, results=results)
    return json_path, markdown_path


def _baseline_allows_current_results(
    *,
    summary: dict[str, object],
    results: list[RetrievalEvaluationResult],
    baseline_path: Path,
) -> bool:
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_summary = baseline_payload.get("summary", {})

    if summary["passed_cases"] < baseline_summary.get("passed_cases", 0):
        return False
    if summary["pass_rate"] < baseline_summary.get("pass_rate", 0.0):
        return False
    if summary["average_expected_doc_recall"] < baseline_summary.get("average_expected_doc_recall", 0.0):
        return False

    baseline_results = {item["case_id"]: item for item in baseline_payload.get("results", [])}
    current_results = {result.case_id: result for result in results}

    for case_id, baseline_result in baseline_results.items():
        current_result = current_results.get(case_id)
        if current_result is None:
            return False
        if baseline_result.get("passed", False) and not current_result.passed:
            return False
        if current_result.expected_doc_recall < baseline_result.get("expected_doc_recall", 0.0):
            return False

    return True


def prepare_evaluation_vector_store(project_root: Path) -> None:
    """Reset the canonical collection and index the checked-in evaluation corpus."""
    evaluation_root = resolve_evaluation_root(project_root)
    corpus_root = evaluation_root / "corpus"
    retrieval_config = RetrievalConfig()
    chunks = build_evaluation_chunks(corpus_root)

    setup_vector_store(force=True, dimension=None)

    with TemporaryDirectory() as temp_dir:
        chunk_path = write_evaluation_chunks(chunks, Path(temp_dir) / "retrieval_evaluation_chunks.json")
        script = EmbedUpsertScript(
            store_type="weaviate",
            model_name=retrieval_config.embed_model_name,
            batch_size=16,
            collection_name=retrieval_config.index_name,
            url=get_settings().weaviate_url,
        )
        summary = script.run(str(chunk_path))
        logger.info("Prepared evaluation vector store", extra={"summary": summary})


def run_canonical_retrieval_evaluation(*, project_root: Path, prepare_corpus: bool) -> list[RetrievalEvaluationResult]:
    """Run the canonical retrieval evaluation against the live retrieval service."""
    evaluation_root = resolve_evaluation_root(project_root)
    cases = load_retrieval_evaluation_cases(evaluation_root / "retrieval_gold_cases.json")

    if prepare_corpus:
        prepare_evaluation_vector_store(project_root)

    client = connect_to_weaviate()
    try:
        retrieval_service = create_hybrid_retriever(client, RetrievalConfig())
        results: list[RetrievalEvaluationResult] = []

        for case in cases:
            chunks = retrieval_service.retrieve(
                RetrieverRequest(
                    query=case.query,
                    platform=case.platform,
                    filters=case.filters,
                    top_k=case.top_k,
                )
            )
            results.append(evaluate_retrieval_case(case, chunks))

        return results
    finally:
        client.close()


def execute_retrieval_evaluation(
    *,
    artifact_dir: Path,
    baseline_path: Path,
    run_label: str,
    evaluator=None,
) -> int:
    """Execute retrieval evaluation, write artifacts, and compare to baseline."""
    results = evaluator() if evaluator is not None else run_canonical_retrieval_evaluation(
        project_root=resolve_backend_root(),
        prepare_corpus=False,
    )
    summary = summarize_retrieval_results(results)
    write_retrieval_artifacts(
        artifact_dir=artifact_dir,
        run_label=run_label,
        summary=summary,
        results=results,
    )

    if not baseline_path.exists():
        return 0

    return 0 if _baseline_allows_current_results(summary=summary, results=results, baseline_path=baseline_path) else 1


def main(argv: list[str] | None = None) -> int:
    """Run the retrieval evaluation CLI."""
    args = create_parser().parse_args(argv)
    project_root = resolve_backend_root()
    artifact_dir = resolve_artifact_dir(project_root, args.artifact_dir)
    baseline_path = resolve_baseline_path(project_root, args.baseline_path)

    return execute_retrieval_evaluation(
        artifact_dir=artifact_dir,
        baseline_path=baseline_path,
        run_label=args.run_label,
        evaluator=lambda: run_canonical_retrieval_evaluation(
            project_root=project_root,
            prepare_corpus=args.prepare_corpus,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
