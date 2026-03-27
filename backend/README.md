# Fitvise Backend

`backend/botadvisor` is the only canonical backend runtime.

## Canonical Entry Point

Use the BotAdvisor runtime and docs from:

- `backend/botadvisor`
- `backend/botadvisor/docs/*`

Quick start:

```bash
cd backend
uv sync
uv run python -m botadvisor.scripts.setup_vector_store --help
uv run python -m botadvisor.scripts.ingest --help
uv run python -m botadvisor.scripts.embed_upsert --help
uv run uvicorn botadvisor.app.main:app --reload
```

## Archived Legacy Backend

Historical legacy assets now live under:

- `backend/deprecated/legacy_backend/app`
- `backend/deprecated/legacy_backend/docs`
- `backend/deprecated/legacy_backend/tests`
- `backend/deprecated/legacy_backend/scripts`

They are preserved only for migration reference and should not be used as live runtime inputs.
