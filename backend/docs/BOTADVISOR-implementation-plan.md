# BotAdvisor Implementation Plan (per target structure)

This plan maps each file in `docs/BOTADVISOR-phase.md` to concrete implementation steps. Reuse legacy patterns only when they reduce work and fit the slim architecture (e.g., Pydantic settings, FastAPI wiring style). Avoid reintroducing legacy boilerplate or Airflow-era complexity.

## app/
### api/v1/chat.py
- Implement POST `/api/v1/chat` with streaming (Server-Sent Events or chunked response). Use FastAPI `StreamingResponse`.
- Request schema: user query, optional platform/source filter, max_tokens, temperature.
- Flow: validate -> build agent (from `agent/assembler.py`) -> invoke with retriever tool -> stream tokens; include citations metadata in final chunk.
- Error handling: graceful empty-context fallback message; log via `observability/logging.py`; trace via LangFuse callback.
- Reuse: existing FastAPI patterns in `app/api/v1` (request/response schemas, dependency injection style) if lean enough.

### api/v1/health.py
- Simple GET returning status of LLM readiness and vector store connectivity (ping via retriever).
- No dependencies beyond config and retriever health probe.

### api/deps.py
- Provide DI helpers: `get_settings()`, `get_langfuse()`, `get_llm_client()`, `get_retriever()`, `get_agent()`.
- Cache lightweight singletons with `lru_cache` where safe; avoid deep containers.
- Reuse: dependency style from current `app/api/v1/fitvise/deps.py` if minimal.

## agent/
### prompts/system_prompt.md
- Write concise fitness-coach system prompt with retrieval instructions and citation requirements.
- Keep variables for persona, tone, and citation formatting.

### tools/retriever.py
- Wrap LangChain-compatible retriever tool around `app/retrieval` adapter.
- Tool signature: accepts `query: str`, optional `platform: str`, `filters: dict`.
- Return structured documents with metadata (title, url/source, platform, score, chunk_id).
- Include tracing hook to LangFuse.

### assembler.py
- Build LangChain (>=1.0) tool-calling agent or LangGraph node.
- Select LLM via `app/llm/factory.py`; register retriever tool; inject system prompt.
- Enable streaming callbacks; propagate request id for logging.
- Provide `get_agent()` callable used by API and CLI scripts.
- Reuse: if legacy agent assembly exists and is lightweight, adapt; otherwise write fresh.

## retrieval/
### base.py
- Define `RetrieverRequest` (query, platform, filters) and `RetrieverResult` (content, score, metadata, source).
- Abstract class `BaseRetriever` with `async query(req: RetrieverRequest) -> list[RetrieverResult]`.
- Include registry hooks to register adapters by platform/source key.

### llama_index.py
- Implement `LlamaIndexRetriever(BaseRetriever)` using LlamaIndex vector store client (Weaviate/Chroma backend).
- Configure hybrid search and top-k, with optional rerank if available in LlamaIndex.
- Respect platform filter by metadata query; attach citations (source, doc_id, page, chunk_id).
- Reuse: any solid LlamaIndex usage from legacy retrieval components; drop unused complexity.

### registry.py
- Simple registry dict: platform/source -> retriever adapter.
- Provide `get_retriever(platform: str | None)` with default fallback.
- Register `LlamaIndexRetriever` as default; allow additional adapters from `adapters/`.

### adapters/filesystem.py
- Adapter that reads Docling-normalized artifacts from local path for ingestion/testing.
- Uses same interface as base retriever, but can shortcut to local index if present.

### adapters/web.py
- Skeleton adapter for HTTP-sourced documents (crawl/fetch upstream, then index via scripts).
- Minimal stub until web ingestion pipeline lands.

### adapters/gdrive.py
- Skeleton adapter for Google Drive-sourced documents; ensure auth placeholders and doc metadata normalization.

## llm/
### factory.py
- Choose between Ollama Cloud `gemini-3-flash` (default) and local Ollama based on env.
- Expose `get_chat_model(streaming=True/False)` returning LangChain ChatModel with timeouts/token limits from settings.
- Reuse: model selection patterns from current LLM service if concise; remove old metrics hooks in favor of LangFuse callbacks.

### clients/ollama_cloud.py
- Thin wrapper configuring base URL/token for Ollama Cloud; set model `gemini-3-flash`; expose streaming invoke.
- Handle retries/timeouts; surface usage metadata for logging.

### clients/ollama_local.py
- Local Ollama client config with model fallback (e.g., `gemma2:2b` or existing local model).
- Keep parity in interface with cloud client for easy switching.

## observability/
### langfuse.py
- Initialize LangFuse client/callback handler from env; provide accessor used across API, scripts, and agent tools.
- No global side effects beyond a singleton.

### logging.py
- Configure structured logging (json if env flag set); inject request id/context.
- Provide helper to bind request id for API and scripts.
- Reuse: borrow minimal logging setup from current project if lightweight.

## core/
### config.py
- Pydantic BaseSettings: LLM settings (cloud/local, model, timeouts), vector store (Weaviate/Chroma), LangFuse keys, logging mode.
- Include list parsing helpers for CORS if needed by FastAPI.
- Reuse: existing `app/core/config.py` patterns; trim unused fields.

### types.py
- Define lean `Document` dataclass with fields: `id` (uuid/slug), `source_id` (platform-native id), `platform`, `source_url`, `checksum` (content hash for dedupe), `size_bytes`, `mime_type`, `created_at`, optional `fingerprint`/`etag`. Keep serialization helpers and reuse across scripts/retriever/API.
- Shared lightweight DTOs/aliases (e.g., `Chunk`, `DocumentMetadata`) used across retrieval and API responses.

## storage/
### local_storage.py
- Filesystem backend to persist raw artifacts and normalized outputs. Compute checksum (SHA256) and skip writes when checksum already exists; return stored URI.
- Provide simple interface `save(content: bytes, meta: Document) -> StoredArtifact` and `exists(checksum)`.
- Reuse: minimal patterns from legacy local storage if clean; avoid heavy wrappers.

### minio_client.py
- MinIO-backed implementation of the same interface; bucket from env; presign option for retrieval.
- Implement checksum-based dedupe before upload; ensure idempotent writes.
- Include lightweight health check used by `/health`.

## scripts/
### ingest.py
- CLI: `--input` (path/glob/dir), `--out` (dest), `--platform` label.
- Use Docling to parse PDFs/Office docs -> normalized JSON/Parquet chunks with metadata (doc_id, page, section, platform) and compute checksum; persist raw artifacts via storage backend (local/MinIO) with dedupe.
- Log to console; optional LangFuse trace; emit skipped-duplicate count.
- Reuse: Docling patterns from legacy `rag_phase1` if present; otherwise implement fresh.

### embed_upsert.py
- CLI: `--in` (artifact path), `--store` (weaviate|chroma), `--batch-size`, `--model`.
- Build LlamaIndex ingestion pipeline; generate embeddings; idempotent upsert to vector store.
- Handle retries/backoff; emit counts for successes/failures.
- Reuse: any clean embedding/upsert code from legacy `rag_phase2` with minimal adjustments.

### setup_vector_store.py
- Initialize schema/collections for chosen store (Weaviate classes or Chroma collections).
- Safe to re-run; no destructive resets by default.

### eval_retrieval.py
- Offline quality loop: load sample queries + expected docs; run retriever; compute simple metrics (recall@k, MRR).
- Output markdown/JSON report.

### adapters/__init__.py
- Placeholder to allow shared helper functions for script adapters if needed.

## configs/
### .env.local
- Document required variables: `LLM_PROVIDER`, `OLLAMA_CLOUD_URL`, `OLLAMA_CLOUD_API_KEY`, `OLLAMA_LOCAL_URL`, `VECTOR_STORE`, `WEAVIATE_URL`, `LANGFUSE_SECRET`, `LANGFUSE_PUBLIC_KEY`, logging toggles.

### vectorstore/weaviate.yaml
- Class schema: fields for content, metadata (doc_id, chunk_id, platform, source_url, page, section), vector config; hybrid enabled.

### vectorstore/chroma.yaml
- Collection name, metadata schema expectations, persistence path.

### logging.yaml
- Minimal logging formatter/level; optional JSON mode flag.

## tests/
### unit/test_retrieval.py
- Cover platform filter, empty results, metadata preservation from `LlamaIndexRetriever`.
- Use AsyncMock for LlamaIndex client; no real IO.

### unit/test_agent.py
- Validate agent wiring: tool registration, prompt injection, handles missing context gracefully.
- Mock LLM to return deterministic responses.

### unit/test_llm_factory.py
- Assert env switching between cloud/local; timeout/token config propagation.

### integration/test_ingest_and_embed.py
- Run Docling ingest on sample file -> embed_upsert with in-memory/chroma; assert embeddings stored and retrievable.

### integration/test_chat_flow.py
- End-to-end: start agent with stub LLM, run chat route, ensure citations returned.

## Root-level
### docker-compose.yaml
- Services: Weaviate (or Chroma via volume), optional local Ollama; FastAPI app service.
- Include env file mount and healthchecks.

### README.md
- 5-minute path: clone -> `uv sync` -> `uv run python scripts/ingest.py ...` -> `uv run python scripts/embed_upsert.py ...` -> start API -> curl chat example.
- Note LangFuse setup and env switches for cloud/local LLM.
