# BotAdvisor Backlog

## Global Progress
```
Epic A Progress: ⬜⬜⬜ (0/3 tasks completed) - 0%
Epic B Progress: ⬜⬜⬜⬜ (0/4 tasks completed) - 0%
Epic C Progress: ⬜⬜⬜ (0/3 tasks completed) - 0%
Epic D Progress: ⬜⬜⬜ (0/3 tasks completed) - 0%
Epic E Progress: ⬜⬜⬜ (0/3 tasks completed) - 0%
Epic F Progress: ⬜⬜⬜ (0/3 tasks completed) - 0%
```

## Status Summary
```
| Status       | Count |
|--------------|-------|
| todo         | 19    |
| in_progress  | 0     |
| blocked      | 0     |
| review       | 0     |
| done         | 0     |
```

# Meta
- **Goal:** Redesign BotAdvisor into a solo-friendly, script-first RAG backend powered by Docling, LlamaIndex, LangChain, Ollama Cloud `gemini-3-flash`, and LangFuse.
- **Scope:** Ingestion/embedding scripts, flexible retriever adapter, LangChain agent API, lightweight storage/dedup logic, and LangFuse observability.
- **Version:** 1.0
- **Updated:** YYYY-MM-DD

# Epics Overview

---

## Epic A: Foundation & Dependencies
### Progress
⬜⬜⬜ (0/3 tasks completed) - 0%

### Backlog Items
- [ ] **A1. Dependency baseline**  
  Deliverable: `pyproject.toml`/`uv.lock` covering Docling, LlamaIndex, LangChain>=1.0, LangFuse, FastAPI, Weaviate/Chroma, Ollama.  
  Acceptance: fresh install, env var docs.
- [ ] **A2. Repo cleanup**  
  Deliverable: prune legacy modules while keeping `app/`, `scripts/`, `docs/`, `configs/`.  
  Acceptance: legacy pipeline archived/moved to `deprecated/` with migration notes.
- [ ] **A3. Document identity model**  
  Deliverable: lean `Document` dataclass (id, source_id, platform, source_url, checksum, size_bytes, mime_type, created_at).  
  Acceptance: reused to dedupe ingestion + enrich citations.

---

## Epic B: Script-First Data Path
### Progress
⬜⬜⬜⬜ (0/4 tasks completed) - 0%

### Backlog Items
- [ ] **B1. Docling ingestion script**  
  Deliverable: `scripts/ingest.py` that uses Docling to produce normalized chunks and persists raw artifacts to local storage/MinIO with checksum dedupe metadata (doc_id, page, section, platform).  
  Acceptance: CLI (`--input`, `--out`, `--platform`), dedupe log, LangFuse trace.
- [ ] **B2. Embedding/upsert script**  
  Deliverable: `scripts/embed_upsert.py` that runs LlamaIndex ingestion, generates embeddings, and idempotently upserts into Weaviate/Chroma with retries.  
  Acceptance: CLI options for store/model/batch; success/failure counts.
- [ ] **B3. Vector store bootstrap**  
  Deliverable: `scripts/setup_vector_store.py` to initialize collections/schemas without destructive resets.  
  Acceptance: re-runnable script; documented defaults.
- [ ] **B4. Storage + dedupe plumbing**  
  Deliverable: `app/storage/local_storage.py` and `app/storage/minio_client.py` with checksum-driven interface used by ingestion.  
  Acceptance: ingestion skips duplicates; storage backend selectable via env.

---

## Epic C: Flexible Retriever Layer
### Progress
⬜⬜⬜ (0/3 tasks completed) - 0%

### Backlog Items
- [ ] **C1. Retriever contract & registry**  
  Deliverable: `app/retrieval/base.py` defines `RetrieverRequest/Result`, `BaseRetriever`, and adapter registry keys (platform/source).  
  Acceptance: new platform only requires adapter registration.
- [ ] **C2. LlamaIndex retriever**  
  Deliverable: `app/retrieval/llama_index.py` implementing hybrid search, platform filters, and citation metadata.  
  Acceptance: handles empty results, attaches doc metadata.
- [ ] **C3. LangChain adapter tooling**  
  Deliverable: retriever exposed as LangChain `BaseRetriever`/Tool (LangChain >=1.0) via `app/retrieval/registry.py` + LangChain tool wrapper.  
  Acceptance: agent tool returning documents with citations.

---

## Epic D: Agentic Chat Surface
### Progress
⬜⬜⬜ (0/3 tasks completed) - 0%

### Backlog Items
- [ ] **D1. LLM factory**  
  Deliverable: `app/llm/factory.py` to select Ollama Cloud `gemini-3-flash` by default with local Ollama fallback, timeouts, and token limits from env.  
  Acceptance: single env switch; streaming enabled.
- [ ] **D2. Agent & prompt**  
  Deliverable: LangChain agent (or LangGraph) invoking retriever tool with fitness-coach system prompt (`agent/prompts/system_prompt.md`) and streaming callbacks; `agent/assembler.py` builds agent.  
  Acceptance: request-id propagation, LangFuse tracing, empty-context guardrails.
- [ ] **D3. FastAPI surface**  
  Deliverable: `app/api/v1/chat.py` (streaming SSE) and `app/api/v1/health.py`; minimal DI in `app/api/deps.py`.  
  Acceptance: streaming tokens, health endpoint ping uses retriever storage + LLM.

---

## Epic E: Observability & Ops
### Progress
⬜⬜⬜ (0/3 tasks completed) - 0%

### Backlog Items
- [ ] **E1. LangFuse tracing**  
  Deliverable: `app/observability/langfuse.py` wired into scripts, retriever, agent, API; env-driven keys.  
  Acceptance: ingestion/embedding/retriever/agent spans visible in LangFuse.
- [ ] **E2. Structured logging**  
  Deliverable: `app/observability/logging.py` providing request-id binding for API and scripts; optional JSON formatting set via `configs/logging.yaml`.  
  Acceptance: logs readable locally without external stack.
- [ ] **E3. Dev ergonomics**  
  Deliverable: `docker-compose.yaml` (vector store + optional local Ollama), `cmd.sh`/`Makefile` shortcuts, README 5-minute setup path.  
  Acceptance: dev stack starts with single command; README lists ingest/embed/serve steps.

---

## Epic F: Hardening & Extensibility
### Progress
⬜⬜⬜ (0/3 tasks completed) - 0%

### Backlog Items
- [ ] **F1. Platform adapters**  
  Deliverable: example adapters (`app/retrieval/adapters/filesystem.py`, `web.py`, `gdrive.py`) reading Docling outputs and computing Document metadata.  
  Acceptance: new adapter registered via registry without API changes.
- [ ] **F2. Quality loop**  
  Deliverable: `scripts/eval_retrieval.py` comparing platform/model combinations (recall@k, MRR) and emitting report.  
  Acceptance: offline report helps tune retriever/system prompt.
- [ ] **F3. Legacy cleanup**  
  Deliverable: archive or delete prior `rag_phase{1,2,3}` artifacts; update docs to point to new plan/scripts.  
  Acceptance: no dead code paths; docs mention migration.

---

# Backlog Items (Detailed)

## ID: A1
### Title
Dependency baseline

### Epic
Epic A

### Type
Task

### Priority
P1

### Status
todo

### Description
Define the minimal dependency set for Docling ingestion, LlamaIndex retrieval/embedding, LangChain >=1.0 agents, LangFuse tracing, FastAPI API, and vector store clients.

### Acceptance Criteria
- `pyproject.toml` and `uv.lock` include Docling, LlamaIndex, LangChain >=1.0, LangFuse, FastAPI, Weaviate or Chroma, Ollama.
- Fresh install works without manual pins or edits.
- Env vars documented in `configs/.env.example`.

### Inputs Required
- `pyproject.toml`
- `uv.lock`
- `configs/.env.local`

### Outputs Expected
- Updated dependency manifests
- Documented env vars

### Dependencies
- None

### Notes
Keep dependencies lean; avoid legacy or unused libraries.

## ID: A2
### Title
Repo cleanup

### Epic
Epic A

### Type
Task

### Priority
P2

### Status
todo

### Description
Remove or isolate legacy pipeline boilerplate to keep a focused redesign workspace while preserving reference artifacts if needed.

### Acceptance Criteria
- Only essential folders remain (`app/`, `scripts/`, `docs/`, `configs/`).
- Legacy artifacts archived under `deprecated/` with migration notes.
- No runtime references to legacy pipelines.

### Inputs Required
- Existing repo tree
- Legacy `rag_phase*` docs

### Outputs Expected
- Cleaned repo structure
- `deprecated/` notes

### Dependencies
- None

### Notes
Do not delete user-authored changes without explicit approval.

## ID: A3
### Title
Document identity model

### Epic
Epic A

### Type
Task

### Priority
P1

### Status
todo

### Description
Define a minimal `Document` model used across ingestion, storage, retrieval, and citations to support deduplication and traceability.

### Acceptance Criteria
- `Document` includes `id`, `source_id`, `platform`, `source_url`, `checksum`, `size_bytes`, `mime_type`, `created_at`.
- Model reused by ingestion and storage interfaces.
- Document metadata is attached to retriever results.

### Inputs Required
- `app/core/types.py`
- Ingestion script requirements

### Outputs Expected
- Lean `Document` dataclass/type
- Consistent metadata usage

### Dependencies
- A1

### Notes
Keep fields minimal; avoid legacy verbose entities.

## ID: B1
### Title
Docling ingestion script

### Epic
Epic B

### Type
Feature

### Priority
P1

### Status
todo

### Description
Implement `scripts/ingest.py` to parse documents with Docling, produce normalized chunks, and persist raw artifacts via local/MinIO storage with checksum dedupe.

### Acceptance Criteria
- CLI supports `--input`, `--out`, `--platform`.
- Outputs normalized chunk artifacts with doc metadata.
- Duplicate inputs skipped using checksum.

### Inputs Required
- Docling readers
- `app/storage/local_storage.py` or `app/storage/minio_client.py`
- `app/core/types.py`

### Outputs Expected
- `scripts/ingest.py`
- Deduped artifacts and chunk outputs

### Dependencies
- A1
- A3

### Notes
Emit skipped-duplicate count in logs.

## ID: B2
### Title
Embedding/upsert script

### Epic
Epic B

### Type
Feature

### Priority
P1

### Status
todo

### Description
Implement `scripts/embed_upsert.py` to generate embeddings via LlamaIndex and idempotently upsert into Weaviate or Chroma.

### Acceptance Criteria
- CLI supports store selection, model name, batch size.
- Idempotent upsert behavior confirmed.
- Retry/backoff on transient errors.

### Inputs Required
- LlamaIndex ingestion pipeline
- Vector store client configs

### Outputs Expected
- `scripts/embed_upsert.py`
- Embedded vectors stored with metadata

### Dependencies
- A1
- B1

### Notes
Align metadata schema with retriever expectations.

## ID: B3
### Title
Vector store bootstrap

### Epic
Epic B

### Type
Task

### Priority
P2

### Status
todo

### Description
Provide a safe schema/init script for Weaviate or Chroma collections.

### Acceptance Criteria
- Script is re-runnable without destructive resets.
- Default class/collection names documented.
- Aligns schema fields with `Document` and chunk metadata.

### Inputs Required
- `configs/vectorstore/weaviate.yaml`
- `configs/vectorstore/chroma.yaml`

### Outputs Expected
- `scripts/setup_vector_store.py`

### Dependencies
- A1
- A3

### Notes
No destructive resets unless explicitly requested.

## ID: B4
### Title
Storage + dedupe plumbing

### Epic
Epic B

### Type
Feature

### Priority
P1

### Status
todo

### Description
Implement local storage and MinIO clients with checksum-based dedupe for raw artifact persistence.

### Acceptance Criteria
- `app/storage/local_storage.py` and `app/storage/minio_client.py` share a simple interface.
- Dedupe check prevents duplicate artifact writes.
- Storage backend selectable via env.

### Inputs Required
- Storage backend requirements
- `app/core/types.py`

### Outputs Expected
- Storage modules
- Dedupe logic used by ingestion

### Dependencies
- A1
- A3

### Notes
Ensure checksum is computed on raw bytes before write.

## ID: C1
### Title
Retriever contract & registry

### Epic
Epic C

### Type
Task

### Priority
P1

### Status
todo

### Description
Define retriever request/response types and a registry for platform adapters.

### Acceptance Criteria
- `BaseRetriever` and registry interface exist.
- Adapter registration by platform/source key.
- Default retriever fallback works when no platform provided.

### Inputs Required
- `app/retrieval/base.py`
- `app/retrieval/registry.py`

### Outputs Expected
- Registry and base classes

### Dependencies
- A3

### Notes
Keep the interface minimal and async-friendly.

## ID: C2
### Title
LlamaIndex retriever

### Epic
Epic C

### Type
Feature

### Priority
P1

### Status
todo

### Description
Build LlamaIndex-based retriever with hybrid search and platform filters, returning citation metadata.

### Acceptance Criteria
- Hybrid search enabled where supported.
- Metadata contains doc id, source, page/section.
- Empty results handled gracefully.

### Inputs Required
- LlamaIndex client setup
- Vector store schema

### Outputs Expected
- `app/retrieval/llama_index.py`

### Dependencies
- B2
- C1

### Notes
Align retrieval metadata with agent citations.

## ID: C3
### Title
LangChain adapter tooling

### Epic
Epic C

### Type
Feature

### Priority
P2

### Status
todo

### Description
Expose retriever as LangChain tool and BaseRetriever adapter for tool-calling agents.

### Acceptance Criteria
- Tool returns document list with metadata.
- Compatible with LangChain >=1.0 tool calling.
- Works with platform filter inputs.

### Inputs Required
- `app/agent/tools/retriever.py`
- `app/retrieval/registry.py`

### Outputs Expected
- LangChain tool adapter

### Dependencies
- C1
- C2

### Notes
Include LangFuse callback support.

## ID: D1
### Title
LLM factory

### Epic
Epic D

### Type
Task

### Priority
P1

### Status
todo

### Description
Create a model factory that defaults to Ollama Cloud `gemini-3-flash` with local Ollama fallback.

### Acceptance Criteria
- Env flag switches cloud/local.
- Streaming enabled and timeouts configurable.
- Model choice surfaced in logs.

### Inputs Required
- `app/llm/factory.py`
- `app/core/config.py`

### Outputs Expected
- Factory module
- Cloud/local client wiring

### Dependencies
- A1

### Notes
Keep interfaces consistent across cloud/local clients.

## ID: D2
### Title
Agent & prompt

### Epic
Epic D

### Type
Feature

### Priority
P1

### Status
todo

### Description
Implement LangChain agent assembly with retriever tool, system prompt, and streaming callbacks.

### Acceptance Criteria
- `agent/assembler.py` builds agent with tool registration.
- `agent/prompts/system_prompt.md` defines fitness-coach behavior.
- Empty-context guardrail message in place.

### Inputs Required
- `app/agent/assembler.py`
- `app/agent/prompts/system_prompt.md`
- Retriever tool

### Outputs Expected
- Agent assembly
- System prompt

### Dependencies
- C3
- D1

### Notes
Use LangFuse callbacks for tracing.

## ID: D3
### Title
FastAPI surface

### Epic
Epic D

### Type
Feature

### Priority
P1

### Status
todo

### Description
Implement minimal FastAPI endpoints for chat and health with streaming responses.

### Acceptance Criteria
- POST `/api/v1/chat` streams responses.
- GET `/health` checks retriever storage + LLM availability.
- DI wiring stays lightweight.

### Inputs Required
- `app/api/v1/chat.py`
- `app/api/v1/health.py`
- `app/api/deps.py`

### Outputs Expected
- API endpoints

### Dependencies
- D2
- E1

### Notes
Prefer SSE or chunked streaming; keep schema minimal.

## ID: E1
### Title
LangFuse tracing

### Epic
Epic E

### Type
Task

### Priority
P2

### Status
todo

### Description
Wire LangFuse callbacks across scripts, retriever, and API flows.

### Acceptance Criteria
- LangFuse handler initialized from env.
- Traces visible for ingest, embed, retrieve, and chat.
- No hard dependency when LangFuse disabled.

### Inputs Required
- `app/observability/langfuse.py`
- LangFuse keys in env

### Outputs Expected
- Callback handler module
- Tracing integrated in key workflows

### Dependencies
- A1

### Notes
Fail open if LangFuse keys missing.

## ID: E2
### Title
Structured logging

### Epic
Epic E

### Type
Task

### Priority
P2

### Status
todo

### Description
Implement structured logging with request-id binding for API and scripts.

### Acceptance Criteria
- Logs include request id when available.
- Optional JSON mode via config.
- No external logging stack required.

### Inputs Required
- `app/observability/logging.py`
- `configs/logging.yaml`

### Outputs Expected
- Logging helpers
- Configurable formatting

### Dependencies
- A1

### Notes
Keep default log format simple for local dev.

## ID: E3
### Title
Dev ergonomics

### Epic
Epic E

### Type
Improvement

### Priority
P3

### Status
todo

### Description
Provide a simple dev stack and shortcuts to run scripts and API quickly.

### Acceptance Criteria
- `docker-compose.yaml` includes vector store and optional local Ollama.
- `cmd.sh` or `Makefile` provides standard commands.
- README includes 5-minute setup path.

### Inputs Required
- `docker-compose.yaml`
- `cmd.sh` or `Makefile`
- `README.md`

### Outputs Expected
- Dev tooling and docs

### Dependencies
- B1
- B2

### Notes
Keep compose minimal to avoid maintenance overhead.

## ID: F1
### Title
Platform adapters

### Epic
Epic F

### Type
Feature

### Priority
P3

### Status
todo

### Description
Provide example adapters for filesystem, web, and Google Drive sources using Docling output format.

### Acceptance Criteria
- Adapter files exist and register with registry.
- Each adapter enriches Document metadata consistently.
- New adapter requires no API changes.

### Inputs Required
- `app/retrieval/adapters/`
- `app/retrieval/registry.py`

### Outputs Expected
- Adapter implementations

### Dependencies
- C1

### Notes
Initial adapters can be thin stubs with clear TODOs.

## ID: F2
### Title
Quality loop

### Epic
Epic F

### Type
Improvement

### Priority
P3

### Status
todo

### Description
Add offline retrieval evaluation to compare models/platforms and surface quality metrics.

### Acceptance Criteria
- `scripts/eval_retrieval.py` runs with sample queries and outputs metrics.
- Report saved as JSON/Markdown.
- Metrics include recall@k and MRR.

### Inputs Required
- Sample queries/expected docs
- Retriever interface

### Outputs Expected
- Evaluation script
- Report artifact

### Dependencies
- C2

### Notes
Keep metrics simple for maintainability.

## ID: F3
### Title
Legacy cleanup

### Epic
Epic F

### Type
Task

### Priority
P2

### Status
todo

### Description
Archive or remove legacy rag phase artifacts and update docs to point to redesign.

### Acceptance Criteria
- `rag_phase1.md`, `rag_phase2.md`, `rag_phase3.md` marked deprecated or archived.
- Docs point to `BOTADVISOR-BACKLOGS.md` and `BOTADVISOR-phase.md`.
- No dead code paths remain.

### Inputs Required
- Legacy docs
- New redesign docs

### Outputs Expected
- Deprecated markers or archived files

### Dependencies
- A2

### Notes
Avoid deleting history needed for reference.


# How to Update This Backlog
- Keep progress bars and counts accurate.
- Maintain checkbox state (`[ ]` or `[x]`).
- Use the template above for new items.