# Phase 10 Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable release-readiness flow for the canonical `backend/botadvisor` runtime so a solo developer can verify startup, health, and baseline API behavior before shipping changes.

**Architecture:** Keep hardening thin and explicit. Reuse the existing `botadvisor.scripts.dev` entrypoint, add a dedicated smoke-test path for runtime validation, extend health/readiness signals only where they support real release checks, and document one canonical verification flow instead of introducing deployment complexity.

**Tech Stack:** Python, FastAPI, uv, Docker Compose, pytest, ruff, BotAdvisor runtime scripts

---

## File Structure

- Modify: `backend/botadvisor/scripts/dev.py`
  Add a smoke-test-oriented command or command path that exercises the canonical local runtime without adding hidden behavior.
- Create: `backend/botadvisor/scripts/release_check.py`
  Hold the repeatable release verification entrypoint for smoke checks and command orchestration.
- Create: `backend/botadvisor/app/dev/smoke.py`
  Keep runtime smoke-check logic separate from CLI parsing and process control.
- Modify: `backend/botadvisor/app/health.py`
  Tighten readiness semantics only if needed by the smoke flow.
- Create: `backend/botadvisor/tests/unit/test_release_check.py`
  Verify the release-check command contract.
- Create: `backend/botadvisor/tests/unit/test_smoke_runtime.py`
  Verify smoke-check orchestration and readiness behavior.
- Modify: `backend/botadvisor/tests/unit/test_local_stack.py`
  Extend the local stack contract only where the new release path depends on it.
- Modify: `backend/botadvisor/README.md`
  Add the canonical release verification commands.
- Modify: `backend/botadvisor/docs/tasks.md`
  Mark `Phase 10` items as they complete.
- Create: `backend/botadvisor/docs/release_hardening.md`
  Document the release-readiness checklist and assumptions.

## Task 1: Define Smoke Runtime Contracts

**Files:**
- Create: `backend/botadvisor/tests/unit/test_smoke_runtime.py`
- Modify: `backend/botadvisor/app/health.py`
- Create: `backend/botadvisor/app/dev/smoke.py`

- [ ] **Step 1: Write failing smoke-runtime tests**

Add tests that verify:
- a smoke runtime check can validate API health payload shape
- readiness fails when required checks are degraded
- smoke logic stays separate from CLI/process boot code

- [ ] **Step 2: Run the targeted smoke-runtime tests to confirm failure**

Run:
```bash
cd backend
uv run pytest botadvisor/tests/unit/test_smoke_runtime.py -q
```

Expected: failing tests because the smoke runtime module does not exist yet.

- [ ] **Step 3: Implement the minimal smoke runtime module**

Create `backend/botadvisor/app/dev/smoke.py` with focused functions for:
- checking response status and payload shape
- interpreting runtime readiness from `/health`
- returning explicit pass/fail summaries

- [ ] **Step 4: Tighten `health.py` only where the smoke contract requires it**

Keep the API response backwards-compatible, but make sure readiness semantics are explicit enough for release checks.

- [ ] **Step 5: Re-run the targeted smoke-runtime tests**

Run:
```bash
cd backend
uv run pytest botadvisor/tests/unit/test_smoke_runtime.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/botadvisor/app/dev/smoke.py backend/botadvisor/app/health.py backend/botadvisor/tests/unit/test_smoke_runtime.py
git commit -m "feat(botadvisor): add smoke runtime checks"
```

## Task 2: Add Canonical Release Check Command

**Files:**
- Create: `backend/botadvisor/tests/unit/test_release_check.py`
- Create: `backend/botadvisor/scripts/release_check.py`
- Modify: `backend/botadvisor/scripts/dev.py`
- Modify: `backend/botadvisor/tests/unit/test_local_stack.py`

- [ ] **Step 1: Write failing command-contract tests**

Add tests that verify:
- `release_check` exposes a canonical CLI
- the command can run smoke checks against the local runtime
- the command returns non-zero when readiness fails
- `dev.py` can invoke the release path without absorbing smoke logic

- [ ] **Step 2: Run the targeted release-check tests to confirm failure**

Run:
```bash
cd backend
uv run pytest botadvisor/tests/unit/test_release_check.py botadvisor/tests/unit/test_local_stack.py -q
```

Expected: failure because `release_check.py` and related command behavior do not exist yet.

- [ ] **Step 3: Implement the release-check CLI**

Create `backend/botadvisor/scripts/release_check.py` with:
- explicit argument parsing
- repeatable command execution
- smoke summary output
- non-zero exit on failed release checks

- [ ] **Step 4: Keep `dev.py` thin**

If `dev.py` gains a release-oriented subcommand, it should delegate to the new release-check module rather than mixing release logic into the local stack boot path.

- [ ] **Step 5: Re-run the targeted command tests**

Run:
```bash
cd backend
uv run pytest botadvisor/tests/unit/test_release_check.py botadvisor/tests/unit/test_local_stack.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/botadvisor/scripts/release_check.py backend/botadvisor/scripts/dev.py backend/botadvisor/tests/unit/test_release_check.py backend/botadvisor/tests/unit/test_local_stack.py
git commit -m "feat(botadvisor): add canonical release check command"
```

## Task 3: Add Repeatable Release Verification Flow

**Files:**
- Modify: `backend/botadvisor/README.md`
- Create: `backend/botadvisor/docs/release_hardening.md`
- Modify: `backend/botadvisor/docs/tasks.md`

- [ ] **Step 1: Document the canonical release check flow**

Add:
- the exact smoke command
- required local prerequisites
- what constitutes a pass or failure
- what a solo developer should check before push or deploy

- [ ] **Step 2: Update `tasks.md` as Phase 10 items land**

Mark completed items only after the smoke flow and release command are verified.

- [ ] **Step 3: Re-run docs-adjacent verification**

Run:
```bash
cd backend
uv run pytest botadvisor/tests/unit/test_runtime_env.py -q
uv run python -m botadvisor.scripts.release_check --help
```

Expected: PASS and usable CLI help output.

- [ ] **Step 4: Commit**

```bash
git add backend/botadvisor/README.md backend/botadvisor/docs/release_hardening.md backend/botadvisor/docs/tasks.md
git commit -m "docs(botadvisor): document release hardening flow"
```

## Task 4: Full Verification Gate

**Files:**
- Verify only

- [ ] **Step 1: Run the full canonical test suite**

Run:
```bash
cd backend
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 2: Run canonical lint checks**

Run:
```bash
cd backend
uv run ruff check botadvisor/app botadvisor/scripts botadvisor/tests
```

Expected: PASS.

- [ ] **Step 3: Run patch integrity check**

Run:
```bash
cd /Users/junhyounglee/workspace/fitvise
git diff --check
```

Expected: no output.

- [ ] **Step 4: Commit any final release-hardening adjustments**

```bash
git add <files changed during verification fixes>
git commit -m "chore(botadvisor): finish release hardening"
```
