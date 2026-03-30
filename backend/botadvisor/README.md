# BotAdvisor

Canonical Fitvise backend.

`backend/botadvisor` is the only live runtime target.
Legacy source material is archived under `backend/deprecated/legacy_backend` and is not part of the canonical execution path.

## Current Scope

The runtime currently standardizes:

- shared package layout under `botadvisor.app`
- canonical settings in `botadvisor.app.core.config`
- script-first ingestion and embedding entrypoints
- canonical Weaviate bootstrap via `botadvisor.scripts.setup_vector_store`
- storage backend selection for local and MinIO artifact persistence
- Weaviate-first hybrid retrieval and thin FastAPI endpoints
- structured logging and LangFuse tracing foundations
- module boundaries for `chat`, `ingestion`, `retrieval`, `storage`, and `observability`

## Quick Start

Prerequisites:

- Docker
- Python with `uv`
- Ollama running locally at `http://localhost:11434` if you want live chat responses

From the backend root:

```bash
cd backend
cp .env.example .env
uv sync
uv run python -m botadvisor.scripts.dev up
```

This canonical startup flow:

- starts local Weaviate with Docker Compose
- bootstraps the canonical vector store schema
- starts the FastAPI server with reload enabled

Stop local dependencies with:

```bash
uv run python -m botadvisor.scripts.dev down
```

## Working Directory

Run canonical BotAdvisor commands from the backend root:

```bash
cd backend
```

The package root is `botadvisor`, so execution should use module form instead of direct script-path execution.

## Canonical Commands

### Ingestion

```bash
uv run python -m botadvisor.scripts.ingest --help
```

Example:

```bash
uv run python -m botadvisor.scripts.ingest \
  --input ./documents/sample.txt \
  --out ./data/chunks \
  --platform filesystem
```

### Embed And Upsert

```bash
uv run python -m botadvisor.scripts.embed_upsert --help
```

Example:

```bash
uv run python -m botadvisor.scripts.embed_upsert \
  --input ./data/chunks \
  --store weaviate \
  --batch-size 32
```

### Vector Store Bootstrap

```bash
uv run python -m botadvisor.scripts.setup_vector_store --help
```

Example:

```bash
uv run python -m botadvisor.scripts.setup_vector_store --force
```

### API Server

```bash
uv run uvicorn botadvisor.app.main:app --reload
```

Canonical local runtime entrypoint:

```bash
uv run python -m botadvisor.scripts.dev up
```

Current canonical endpoints:

- `GET /health`
- `POST /query`
- `POST /chat` returns `application/x-ndjson` streaming chunks

## Testing

Run BotAdvisor tests from the `backend/botadvisor` directory so they stay isolated from archived legacy tests.

```bash
cd backend/botadvisor
uv run pytest tests/unit -q
```

Targeted regression runs are also fine:

```bash
uv run pytest tests/unit/test_phase2_runtime_layout.py -q
uv run pytest tests/unit/test_embed_upsert.py -q
```

## Linting

From the backend root:

```bash
uv run ruff check botadvisor/app botadvisor/scripts botadvisor/tests
```

## Release Verification

Use this flow before push or deploy work that changes runtime behavior.

In terminal 1, start the canonical local runtime:

```bash
cd backend
uv run python -m botadvisor.scripts.dev up
```

In terminal 2, run the canonical readiness check:

```bash
cd backend
uv run python -m botadvisor.scripts.release_check
```

To validate the boot flow end-to-end with one command, use:

```bash
cd backend
uv run python -m botadvisor.scripts.release_check --boot-smoke
```

The release check passes only when:

- `/health` returns HTTP `200`
- the top-level status is `healthy`
- `retrieval`, `vector_store`, and `llm_path` checks are all `healthy`

For a broader pre-release verification pass, also run:

```bash
cd backend
uv run pytest -q
uv run ruff check botadvisor/app botadvisor/scripts botadvisor/tests
```

## Retrieval Evaluation

Use this flow before expanding the backend into tool calling or more advanced
agent orchestration.

```bash
cd backend
uv run python -m botadvisor.scripts.evaluate_retrieval --prepare-corpus
```

This command:

- rebuilds the checked-in evaluation corpus in the canonical Weaviate collection
- runs the gold query set against the live retrieval service
- writes local JSON and Markdown artifacts under `backend/.tmp/retrieval_evaluation`
- compares the run against `botadvisor/evaluation/baseline/retrieval_baseline.json`

## Storage Configuration

Canonical artifact storage is selected through environment variables in `botadvisor.app.core.config`.

- `STORAGE_BACKEND=local` uses `STORAGE_LOCAL_PATH`
- `STORAGE_BACKEND=minio` uses `STORAGE_MINIO_ENDPOINT`, `STORAGE_MINIO_ACCESS_KEY`, `STORAGE_MINIO_SECRET_KEY`, `STORAGE_MINIO_BUCKET`, and `STORAGE_MINIO_SECURE`

Start from the backend root `.env.example` and override only what you need for local development.
Legacy Fitvise env keys are not part of the canonical `botadvisor` runtime and may be ignored.

## Local Runtime Pieces

- `.env.example` in the backend root provides the canonical local settings baseline
- `botadvisor/docker-compose.yaml` provisions the local Weaviate dependency
- `botadvisor.scripts.dev` is the canonical developer entrypoint

## Canonical Docs

Use these docs as the single source of truth:

- `docs/architecture.md`
- `docs/coding_conventions.md`
- `docs/product_scope.md`
- `docs/backlog.md`
- `docs/migration.md`
- `docs/release_hardening.md`
- `docs/tasks.md`

## Non-Canonical References

These are not current runtime references:

- `backend/README.md`
- `backend/deprecated/legacy_backend/app`
- `backend/deprecated/legacy_backend/tests`
- `backend/deprecated/legacy_backend/docs/*`
