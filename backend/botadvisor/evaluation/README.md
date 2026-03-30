# Retrieval Evaluation Assets

This directory contains the checked-in assets for canonical retrieval
evaluation.

## Contents

- `corpus/`
  Deterministic fixture documents used to build the retrieval baseline.
- `retrieval_gold_cases.json`
  Small hand-reviewed query set for the canonical retrieval gate.
- `baseline/retrieval_baseline.json`
  Checked-in baseline summary used for regression comparison.

## Canonical Command

From `backend/`:

```bash
uv run python -m botadvisor.scripts.evaluate_retrieval --prepare-corpus
```

This command:

1. resets the canonical Weaviate collection
2. indexes the checked-in evaluation corpus
3. runs the gold query set through the live retrieval service
4. writes local JSON and Markdown artifacts
5. compares the current run against the checked-in baseline

## Update Rule

Only update the checked-in baseline when a retrieval behavior change is
intentional and reviewed. When that happens, update:

- `retrieval_gold_cases.json`
- `baseline/retrieval_baseline.json`
- `docs/retrieval_quality.md`
