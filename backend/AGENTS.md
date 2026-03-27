# Fitvise Backend Agent Guide

This file defines how agents should understand and modify the backend project.

## Current Project Truth

- The future canonical backend is `backend/botadvisor`.
- The legacy backend is `backend/app`.
- `backend/app` is migration source material, not the long-term runtime target.
- The long-term goal is to replace `backend/app` with `backend/botadvisor` and then archive or remove the legacy runtime.

## Canonical Planning Docs

Agents must treat these as the source of truth before making structural changes:

- `backend/botadvisor/docs/architecture.md`
- `backend/botadvisor/docs/coding_conventions.md`
- `backend/botadvisor/docs/product_scope.md`
- `backend/botadvisor/docs/backlog.md`
- `backend/botadvisor/docs/migration.md`
- `backend/botadvisor/docs/tasks.md`

If these documents and the code disagree, update the code or the docs so they converge. Do not leave them drifting.

## What Agents Should Optimize For

Prioritize the following in order:

1. Solo-developer maintainability
2. Retrieval quality and correctness
3. Low operational complexity
4. Clear module boundaries
5. Explicit, testable runtime behavior

Do not optimize first for framework cleverness, generalized extensibility, or preserving legacy architecture.

## Canonical Architecture Summary

BotAdvisor is a hybrid modular monolith.

- Shared foundations stay small.
- Product behavior lives in focused feature modules.
- Scripts remain first-class operational entry points.
- API remains thin.

Core design choices:

- retrieval core: `LlamaIndex`
- primary vector store: `Weaviate`
- retrieval requirement: `hybrid search`
- orchestration layer: `LangChain`
- `LangGraph` is deferred until there is a real graph-shaped workflow
- observability: `LangFuse` plus structured logging

### Architectural Boundaries

- Routers translate HTTP to service calls and nothing more.
- Retrieval logic belongs inside retrieval modules.
- LangChain does not own retrieval implementation.
- Storage modules own persistence concerns only.
- Observability modules provide tracing and logging, but do not own business logic.

## Migration Rules

- Prefer creating or refactoring code under `backend/botadvisor`.
- Do not add new product features under `backend/app` unless the task is explicitly legacy-only.
- Do not import from `backend/app` into final `backend/botadvisor` runtime code.
- You may mine legacy code for behavior, contracts, or tests, but port the behavior into `botadvisor` cleanly.
- Port behavior, not architecture.

Legacy code that works but violates the new quality bar is migration input, not a stable foundation.

## Harness Engineering Rules

These rules apply to every new file and every meaningful refactor.

### Project Path Rules

- New runtime code goes into `backend/botadvisor`.
- New canonical docs go into `backend/botadvisor/docs`.
- Treat `backend/docs` as historical or compatibility-oriented unless the user explicitly asks otherwise.

### Structure Rules

- Keep `botadvisor` as a hybrid modular monolith.
- Prefer feature-focused modules over deep layered boilerplate.
- Do not introduce a large dependency injection container into `botadvisor`.
- Avoid speculative plugin systems, provider registries, or adapter trees unless real variation already exists.

### Retrieval Rules

- Retrieval is a core product capability, not a side concern.
- Use `LlamaIndex` as the retrieval engine.
- Use `Weaviate` as the default production vector backend.
- Preserve metadata required for citations and filters.
- Keep hybrid-search behavior inside retrieval modules and retrieval configuration.
- LangChain adapters and tools must wrap retrieval, not reimplement it.

### API Rules

- Keep routers thin.
- Validation belongs at schema or boundary level.
- Orchestration belongs in focused modules or services.
- Persistence and retrieval calls do not belong directly in routers.
- Ingestion, embed/upsert, and vector-store bootstrap remain script-first operational paths.

### Script Rules

- Scripts are production code.
- Separate argument parsing from business logic.
- Keep retries, logging, and exit behavior explicit.
- Return machine-readable summaries when practical.
- Avoid hidden process-global side effects when a local dependency can be injected or passed in.

### Code Quality Rules

- Every file must have one clear responsibility.
- Every function must have one clear reason to change.
- Prefer small, boring, explicit code.
- Avoid god services, vague helpers, and oversized files.
- If a file mixes parsing, IO, orchestration, and formatting, split it.
- If an abstraction does not reduce real coupling or improve testing, remove it.
- Code that violates SRP or SOLID is a refactor target even if it currently works.

### Logging And Observability Rules

- Use structured logging where possible.
- Bind request or trace identifiers at runtime boundaries.
- Use LangFuse for meaningful end-to-end operations, not every tiny helper.
- Do not swallow important failures behind silent fallback behavior.

## Testing And Verification Rules

- Do not claim work is complete without fresh verification.
- Prefer test-first development for feature work and bug fixes.
- Use unit tests for contracts and orchestration seams.
- Use integration tests where adapter behavior or runtime wiring matters.
- A backlog item is not `done` just because code exists. It must also satisfy structure, quality, and verification expectations.

## Backlog Handling Rules

Canonical backlog statuses are:

- `done`
- `partial`
- `todo`
- `drop`

Interpret them strictly:

- `done`: implemented, reusable, verified, and aligned with the canonical architecture
- `partial`: implementation exists, but migration or cleanup is still required
- `todo`: core implementation is still missing
- `drop`: intentionally removed from the solo-developer scope

Do not upgrade a backlog item to `done` by editing docs alone.

## Working Commands

Use the current `backend` project tooling:

```bash
cd backend
uv sync
uv run pytest
uv run ruff check .
uv run black .
```

BotAdvisor-oriented examples:

```bash
cd backend
uv run python -m botadvisor.scripts.ingest --help
uv run python -m botadvisor.scripts.embed_upsert --help
uv run python -m botadvisor.scripts.setup_vector_store --help
```

## When In Doubt

- Follow `backend/botadvisor/docs/architecture.md`.
- Follow `backend/botadvisor/docs/coding_conventions.md`.
- Prefer simpler code and fewer layers.
- Prefer reducing legacy dependence.
- Prefer explicit runtime behavior over flexible-looking abstractions.
