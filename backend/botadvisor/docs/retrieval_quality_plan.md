# Retrieval Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical retrieval-evaluation workflow with a gold query set, a repeatable CLI, and regression artifacts for the live BotAdvisor runtime.

**Architecture:** The evaluation path stays repo-native. A small checked-in gold query set defines expected retrieval behavior, a dedicated evaluation service computes case-level checks against the canonical retrieval module, and a CLI writes JSON plus Markdown artifacts for review and regression comparison.

**Tech Stack:** Python, pytest, pydantic/dataclasses already in repo, JSON and Markdown artifacts, existing BotAdvisor retrieval service and scripts package

---

### Task 1: Define Retrieval Evaluation Contracts

**Files:**
- Create: `backend/botadvisor/app/evaluation/retrieval_cases.py`
- Create: `backend/botadvisor/tests/unit/test_retrieval_cases.py`
- Modify: `backend/botadvisor/docs/tasks.md`

- [ ] **Step 1: Write the failing test**

Add tests that define the shape of a gold evaluation case and reject incomplete
or vague case data.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_cases.py -q`

- [ ] **Step 3: Write minimal implementation**

Add explicit case/result dataclasses or pydantic models with naming aligned to
the coding conventions.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_cases.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): define retrieval evaluation contracts`

### Task 2: Add Canonical Gold Query Set

**Files:**
- Create: `backend/botadvisor/evaluation/retrieval_gold_cases.json`
- Create: `backend/botadvisor/tests/unit/test_retrieval_gold_cases.py`
- Modify: `backend/botadvisor/docs/retrieval_quality.md`

- [ ] **Step 1: Write the failing test**

Add tests that assert the gold set file exists, stays small, uses explicit case
identifiers, and contains the required fields.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_gold_cases.py -q`

- [ ] **Step 3: Write minimal implementation**

Create the first checked-in gold query set with 8 to 12 hand-reviewed cases.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_gold_cases.py -q`

- [ ] **Step 5: Commit**

Commit: `test(botadvisor): add retrieval gold query set`

### Task 3: Add Retrieval Evaluation Service

**Files:**
- Create: `backend/botadvisor/app/evaluation/retrieval_service.py`
- Create: `backend/botadvisor/tests/unit/test_retrieval_evaluation_service.py`
- Modify: `backend/botadvisor/app/retrieval/service.py` (only if needed for a stable seam)

- [ ] **Step 1: Write the failing test**

Add tests that evaluate mocked retrieval results for relevance, citation
integrity, metadata integrity, and recall.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_evaluation_service.py -q`

- [ ] **Step 3: Write minimal implementation**

Add a focused evaluation service that accepts canonical cases plus retrieval
results and returns explicit case-level findings.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_evaluation_service.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add retrieval evaluation service`

### Task 4: Add Canonical Evaluation CLI And Artifacts

**Files:**
- Create: `backend/botadvisor/scripts/evaluate_retrieval.py`
- Create: `backend/botadvisor/tests/unit/test_evaluate_retrieval.py`
- Create: `backend/botadvisor/evaluation/baseline/retrieval_baseline.json`
- Create: `backend/botadvisor/evaluation/README.md`
- Modify: `backend/botadvisor/README.md`

- [ ] **Step 1: Write the failing test**

Add tests for parser behavior, artifact paths, and run-summary generation.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_evaluate_retrieval.py -q`

- [ ] **Step 3: Write minimal implementation**

Add a CLI that reads the gold cases, runs the evaluation service, writes JSON
and Markdown artifacts, and compares against the checked-in baseline.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_evaluate_retrieval.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add retrieval evaluation command`

### Task 5: Close Phase 11 With Full Verification

**Files:**
- Modify: `backend/botadvisor/docs/tasks.md`
- Modify: `backend/botadvisor/docs/backlog.md`
- Modify: `backend/botadvisor/docs/retrieval_quality.md`

- [ ] **Step 1: Run focused verification**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_cases.py botadvisor/tests/unit/test_retrieval_gold_cases.py botadvisor/tests/unit/test_retrieval_evaluation_service.py botadvisor/tests/unit/test_evaluate_retrieval.py -q`

- [ ] **Step 2: Run full backend verification**

Run: `cd backend && uv run pytest -q`

- [ ] **Step 3: Run lint**

Run: `cd backend && uv run ruff check botadvisor/app botadvisor/scripts botadvisor/tests`

- [ ] **Step 4: Run CLI help check**

Run: `cd backend && uv run python -m botadvisor.scripts.evaluate_retrieval --help`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): complete retrieval evaluation baseline`
