# BotAdvisor Release Hardening

## Purpose

This document defines the canonical release-readiness flow for `backend/botadvisor`.
It is the operational companion to `release_readiness.md`, which holds the implementation plan.

## Release-Readiness Assumptions

- `backend/botadvisor` is the only live runtime target.
- The canonical local dependency is Weaviate from `backend/botadvisor/docker-compose.yaml`.
- The canonical local API entrypoint is `uv run python -m botadvisor.scripts.dev up`.
- The canonical readiness probe is `uv run python -m botadvisor.scripts.release_check`.
- Legacy env keys and legacy backend runtime surfaces are not part of the release contract.

## Canonical Smoke Flow

1. Start the runtime from the backend root.

```bash
cd backend
cp .env.example .env
uv sync
uv run python -m botadvisor.scripts.dev up
```

2. In a separate terminal, run the release-readiness check.

```bash
cd backend
uv run python -m botadvisor.scripts.release_check
```

3. Treat the runtime as ready only when the command exits with code `0` and prints `runtime is ready`.

## End-To-End Boot Smoke

To validate the local boot path with one command:

```bash
cd backend
uv run python -m botadvisor.scripts.release_check --boot-smoke
```

This path:

- starts local dependencies
- bootstraps the vector store schema
- starts the API server in a child process
- polls `/health` until the runtime is ready or the timeout is hit
- stops the spawned API server before exiting

## Readiness Rules

The release check fails when any of the following is true:

- `/health` does not return HTTP `200`
- the response payload is missing required health fields
- the top-level runtime status is not `healthy`
- `retrieval` is not `healthy`
- `vector_store` is not `healthy`
- `llm_path` is not `healthy`

## Solo-Developer Release Checklist

Before push or deploy:

- confirm `.env` still matches the canonical BotAdvisor runtime contract
- run the canonical smoke flow against a running local runtime
- run `uv run pytest -q`
- run `uv run ruff check botadvisor/app botadvisor/scripts botadvisor/tests`
- verify no unexpected patch formatting issues with `git diff --check`

## Current Limits

- The release check verifies runtime readiness, not full ingestion or chat behavior
- The current flow assumes a locally reachable API endpoint
- CI automation is still pending; the verification path is currently repeatable but developer-invoked
