# Retrieval Quality

This document defines the canonical retrieval-evaluation workflow for the live
`backend/botadvisor` runtime.

## Goal

Keep hybrid retrieval quality measurable before adding more advanced agent
features such as tool calling, multi-step orchestration, or planner/executor
flows.

## Principles

- Measure the canonical runtime, not an isolated prototype.
- Keep the gold set small, explicit, and reviewable by one maintainer.
- Evaluate retrieval first; answer-generation quality is a separate concern.
- Store results as repository artifacts so regressions are comparable over time.
- Treat retrieval regressions as release blockers for agent-feature expansion.

## Evaluation Scope

Phase 11 evaluates the canonical retrieval path implemented in:

- `backend/botadvisor/app/retrieval/service.py`
- `backend/botadvisor/app/retrieval/factory.py`
- `backend/botadvisor/app/retrieval/langchain_adapter.py`

The evaluation target is hybrid retrieval with:

- Weaviate as the canonical vector store
- LlamaIndex as the canonical retriever implementation
- request-level metadata filtering
- citation-preserving chunk output

## Gold Query Set

The canonical gold set should be a repository-managed JSON artifact with a small
number of hand-reviewed cases.

Each case should include:

- `case_id`
- `query`
- `top_k`
- `platform`
- `filters`
- `expected_doc_ids`
- `expected_source_ids`
- `expected_sections`
- `must_include_text`
- `notes`

The first version should stay intentionally small, targeting 8 to 12 queries.
Coverage should favor stable, high-signal cases over dataset breadth.

The gold set is paired with a checked-in fixture corpus under
`backend/botadvisor/evaluation/corpus/`. Source identifiers in gold cases must
use stable logical values such as `evaluation_corpus/<file>.md`, not absolute
machine-specific paths.

## Evaluation Contract

Each evaluation case produces a structured result with:

- query metadata
- retrieved chunks
- retrieved document identifiers
- citation metadata
- pass/fail checks
- diagnostic notes

The required checks are:

1. relevance
   At least one expected document or source appears in the top-k results.
2. citation integrity
   Retrieved chunks expose usable `doc_id`, `source_id`, `platform`, and
   section/page metadata when expected by the case.
3. metadata correctness
   request-level filters are respected and do not leak cross-platform or
   mismatched-source results.
4. hybrid regression
   result quality remains above a small baseline threshold that can be compared
   across runs.

## Scoring Model

Phase 11 should keep scoring simple and reviewable.

- `relevance_hit`: boolean
- `citation_integrity_hit`: boolean
- `metadata_integrity_hit`: boolean
- `expected_doc_recall`: float from 0.0 to 1.0
- `score`: weighted aggregate for summaries only

The pass condition for a case is:

- relevance hit is true
- citation integrity hit is true
- metadata integrity hit is true

The pass condition for a run is:

- no failing mandatory cases
- aggregate recall does not fall below the stored baseline threshold

## Result Artifacts

Each canonical evaluation run should produce:

- a machine-readable JSON artifact for regression comparison
- a short Markdown summary for human review

The repository should keep:

- a checked-in baseline result artifact
- the checked-in gold query set

Ad hoc run outputs may be written to a local ignored directory, but the baseline
artifact used for comparison must remain in the repository.

## Quality Gate

Before adding more advanced agent features, the retrieval baseline must satisfy:

- canonical evaluation command passes
- no regression against the checked-in baseline
- gold set remains current with the active corpus assumptions

If a retrieval change intentionally changes behavior, the maintainer must update:

- the gold query set
- the baseline artifact
- the diagnostic notes explaining why the baseline changed

## Non-Goals

Phase 11 does not attempt to solve:

- full answer-quality evaluation
- LLM judge workflows
- large-scale benchmark automation
- multi-dataset evaluation
- ranking optimization beyond the canonical hybrid path
