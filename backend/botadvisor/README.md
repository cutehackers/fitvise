# BotAdvisor

Canonical Fitvise backend under active migration.

`backend/botadvisor` is the only future runtime target.
`backend/app` remains legacy source material and is not part of the canonical execution path.

## Current Scope

The runtime currently standardizes:

- shared package layout under `botadvisor.app`
- canonical settings in `botadvisor.app.core.config`
- script-first ingestion and embedding entrypoints
- structured logging and LangFuse tracing foundations
- module boundaries for `chat`, `ingestion`, `retrieval`, `storage`, and `observability`

The retriever and thin API are still being migrated.

## Working Directory

Run canonical BotAdvisor commands from the backend root:

```bash
cd /Users/junhyounglee/workspace/fitvise/backend
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

### API Server

```bash
uv run uvicorn botadvisor.app.main:app --reload
```

Current canonical endpoints:

- `GET /health`
- `POST /query`
- `POST /chat` returns `application/x-ndjson` streaming chunks

## Testing

Run BotAdvisor tests from the `backend/botadvisor` directory so they do not inherit the legacy `backend/tests` runtime setup.

```bash
cd /Users/junhyounglee/workspace/fitvise/backend/botadvisor
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

## Services

`docker-compose.yaml` currently provisions Weaviate for local development.

```bash
cd /Users/junhyounglee/workspace/fitvise/backend/botadvisor
docker compose up -d
```

## Canonical Docs

Use these docs as the single source of truth:

- `docs/architecture.md`
- `docs/coding_conventions.md`
- `docs/product_scope.md`
- `docs/backlog.md`
- `docs/migration.md`
- `docs/tasks.md`

## Non-Canonical References

These are not current runtime references:

- `backend/README.md`
- `backend/app`
- `backend/tests`
- `backend/docs/*` except where explicitly mirrored for migration context
