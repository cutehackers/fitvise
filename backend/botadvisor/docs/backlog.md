# BotAdvisor Canonical Backlog

This is the canonical backlog for the BotAdvisor redesign.
Statuses reflect the actual codebase plus the new solo-developer quality bar.

## Status Definitions

- `done`: implemented and reusable in the future `botadvisor` runtime
- `partial`: implementation exists but needs migration, cleanup, or refactoring
- `todo`: core implementation is still missing
- `drop`: removed from current solo-developer scope

## Backlog

| ID | Title | Status | Evidence | Notes |
|----|-------|--------|----------|-------|
| A1 | Dependency baseline | `done` | `backend/pyproject.toml`, `backend/uv.lock` | Core dependencies for Docling, LangChain, LlamaIndex, LangFuse, Weaviate, Chroma exist |
| A2 | Repo cleanup | `partial` | `backend/deprecated`, `backend/app`, `backend/docs` | Cleanup started, but legacy runtime and docs still coexist |
| A3 | Document identity model | `done` | `backend/botadvisor/app/core/entity/document.py` | Lean document entity exists and is used by ingestion and storage |
| B1 | Docling ingestion script | `done` | `backend/botadvisor/scripts/ingest.py` | CLI, chunk generation, dedupe skip path, LangFuse trace are implemented |
| B2 | Embedding/upsert script | `done` | `backend/botadvisor/scripts/embed_upsert.py`, `backend/botadvisor/tests/unit/test_embed_upsert.py` | CLI and processing summary are implemented and tested |
| B3 | Vector store bootstrap | `done` | `backend/botadvisor/scripts/setup_vector_store.py`, `backend/botadvisor/tests/unit/test_setup_vector_store.py` | Canonical Weaviate bootstrap now lives inside `botadvisor` with an explicit CLI and summary contract |
| B4 | Storage + dedupe plumbing | `done` | `backend/botadvisor/app/storage/*`, `backend/botadvisor/tests/unit/test_storage_factory.py`, `backend/botadvisor/scripts/ingest.py` | Local and MinIO storage selection are centralized in the canonical runtime and reused by ingestion |
| C1 | Retriever contract & registry | `done` | `backend/botadvisor/app/retrieval/*`, `backend/botadvisor/app/core/entity/retriever_request.py` | Canonical retrieval service, config, and request contract are assembled inside `botadvisor` |
| C2 | LlamaIndex retriever | `done` | `backend/botadvisor/app/retrieval/factory.py`, `backend/botadvisor/app/retrieval/service.py` | The LlamaIndex retriever is migrated into `botadvisor` with Weaviate-first hybrid configuration |
| C3 | LangChain adapter tooling | `done` | `backend/botadvisor/app/retrieval/langchain_adapter.py` | Canonical LangChain retriever adapter now wraps the `botadvisor` retrieval service |
| D1 | LLM factory | `done` | `backend/botadvisor/app/llm/*`, `backend/botadvisor/tests/unit/test_llm_factory.py` | Canonical `botadvisor` now owns a small Ollama-first LLM factory and health path |
| D2 | Agent & prompt | `done` | `backend/botadvisor/app/chat/prompting.py`, `backend/botadvisor/app/chat/service.py`, `backend/botadvisor/tests/unit/test_chat_service.py` | Retrieval-aware prompt assembly and LLM-backed answer generation now live inside the canonical chat module |
| D3 | FastAPI surface | `done` | `backend/botadvisor/app/main.py`, `backend/botadvisor/app/api/*`, `backend/botadvisor/app/chat/*` | Canonical chat, query, and health endpoints now exist in `botadvisor` with thin router boundaries |
| E1 | LangFuse tracing | `done` | `backend/botadvisor/app/observability/langfuse.py`, `backend/botadvisor/app/chat/service.py`, `backend/botadvisor/scripts/*.py` | LangFuse is standardized across the canonical API and script entrypoints |
| E2 | Structured logging | `done` | `backend/botadvisor/app/observability/logging.py`, `backend/botadvisor/app/api/middleware.py` | Request correlation and structured logging are standardized for the canonical runtime |
| E3 | Dev ergonomics | `partial` | `backend/botadvisor/docker-compose.yaml`, `backend/botadvisor/README.md` | Core assets exist, but one-command developer flow is not yet fully canonical |
| F1 | Platform adapters | `drop` | scope decision | Only `filesystem` remains first-class in the current scope |
| F2 | Quality loop | `drop` | scope decision | Offline retrieval evaluation is deferred until the core runtime is stable |
| F3 | Legacy cleanup | `partial` | `backend/deprecated`, legacy runtime still active | Cleanup work is required before redesign completion |

## Priority Order

1. retrieval core
2. thin API runtime
3. observability and runtime polish
4. legacy archive and removal

## Completion Rule

No backlog item is complete until it satisfies:

- canonical location in `backend/botadvisor`
- acceptable tests or verification evidence
- alignment with `architecture.md`
- alignment with `coding_conventions.md`
