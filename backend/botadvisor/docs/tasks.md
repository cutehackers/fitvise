# BotAdvisor Refinement Tasks

> This task list turns the redesign into execution work for the canonical `backend/botadvisor` runtime.

## Phase 1: Canonical Documentation

- [ ] Create and keep `backend/botadvisor/docs/architecture.md` as the single architecture reference
- [ ] Create and keep `backend/botadvisor/docs/coding_conventions.md` as the code quality gate
- [ ] Create and keep `backend/botadvisor/docs/product_scope.md` as the scope boundary
- [ ] Reclassify backlog into `backend/botadvisor/docs/backlog.md`
- [ ] Define migration steps in `backend/botadvisor/docs/migration.md`

## Phase 2: Core Runtime Layout

- [x] Add a canonical `core` settings module for `botadvisor`
- [x] Create canonical module boundaries for `ingestion`, `retrieval`, `chat`, `storage`, and `observability`
- [x] Remove ad hoc path and import hacks where possible
- [x] Move runtime-only design decisions out of the legacy backend

## Phase 3: Retrieval Core

- [ ] Port the reusable LlamaIndex retriever behavior from legacy code into `backend/botadvisor`
- [ ] Make Weaviate the default canonical retrieval backend
- [ ] Implement hybrid-search-aware retriever configuration
- [ ] Add citation-preserving retrieval results
- [ ] Add a thin LangChain retriever adapter or tool boundary

## Phase 4: Thin API

- [ ] Add canonical `chat` API endpoints inside `backend/botadvisor`
- [ ] Add canonical `health` endpoints inside `backend/botadvisor`
- [ ] Keep routers thin and push orchestration into focused services
- [ ] Reuse only the behavior that is worth preserving from legacy chat endpoints

## Phase 5: Observability And Runtime Quality

- [ ] Standardize LangFuse usage across scripts and API code
- [ ] Standardize structured logging and request correlation
- [ ] Define runtime health checks for vector store, retriever, and LLM path
- [x] Ensure `README.md` reflects the canonical runtime, not the historical layout

## Phase 6: Storage And Indexing

- [ ] Canonicalize storage backend selection for local and MinIO modes
- [ ] Port or rewrite vector-store bootstrap into `backend/botadvisor/scripts/setup_vector_store.py`
- [ ] Verify script contracts for ingest, embed/upsert, and bootstrap remain explicit and testable

## Phase 7: Legacy Isolation

- [ ] Identify every runtime feature still coming from `backend/app`
- [ ] Port only the behavior needed into `backend/botadvisor`
- [ ] Mark remaining old docs as superseded
- [ ] Archive `backend/app` and obsolete docs once the canonical runtime replaces them

## Phase 8: Quality Sweep

- [ ] Break up oversized files that violate single responsibility
- [ ] Remove dead wrappers, dead abstractions, and speculative extension points
- [ ] Add focused tests around migrated retrieval and API seams
- [ ] Verify all surviving code meets the coding conventions
