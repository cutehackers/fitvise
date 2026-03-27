# BotAdvisor Refinement Tasks

> This task list turns the redesign into execution work for the canonical `backend/botadvisor` runtime.

## Phase 1: Canonical Documentation

- [x] Create and keep `backend/botadvisor/docs/architecture.md` as the single architecture reference
- [x] Create and keep `backend/botadvisor/docs/coding_conventions.md` as the code quality gate
- [x] Create and keep `backend/botadvisor/docs/product_scope.md` as the scope boundary
- [x] Reclassify backlog into `backend/botadvisor/docs/backlog.md`
- [x] Define migration steps in `backend/botadvisor/docs/migration.md`

## Phase 2: Core Runtime Layout

- [x] Add a canonical `core` settings module for `botadvisor`
- [x] Create canonical module boundaries for `ingestion`, `retrieval`, `chat`, `storage`, and `observability`
- [x] Remove ad hoc path and import hacks where possible
- [x] Move runtime-only design decisions out of the legacy backend

## Phase 3: Retrieval Core

- [x] Port the reusable LlamaIndex retriever behavior from legacy code into `backend/botadvisor`
- [x] Make Weaviate the default canonical retrieval backend
- [x] Implement hybrid-search-aware retriever configuration
- [x] Add citation-preserving retrieval results
- [x] Add a thin LangChain retriever adapter or tool boundary

## Phase 4: Thin API

- [x] Add canonical `chat` API endpoints inside `backend/botadvisor`
- [x] Add canonical `health` endpoints inside `backend/botadvisor`
- [x] Keep routers thin and push orchestration into focused services
- [x] Reuse only the behavior that is worth preserving from legacy chat endpoints

## Phase 5: Observability And Runtime Quality

- [x] Standardize LangFuse usage across scripts and API code
- [x] Standardize structured logging and request correlation
- [x] Define runtime health checks for vector store, retriever, and LLM path
- [x] Ensure `README.md` reflects the canonical runtime, not the historical layout

## Phase 6: Storage And Indexing

- [x] Canonicalize storage backend selection for local and MinIO modes
- [x] Port or rewrite vector-store bootstrap into `backend/botadvisor/scripts/setup_vector_store.py`
- [x] Verify script contracts for ingest, embed/upsert, and bootstrap remain explicit and testable

## Phase 7: Legacy Isolation

- [x] Identify every runtime feature still coming from `backend/app`
- [x] Port only the behavior needed into `backend/botadvisor`
- [x] Mark remaining old docs as superseded
- [x] Archive `backend/app` and obsolete docs once the canonical runtime replaces them

## Phase 8: Quality Sweep

- [x] Break up oversized files that violate single responsibility
- [x] Remove dead wrappers, dead abstractions, and speculative extension points
- [x] Add focused tests around migrated retrieval and API seams
- [x] Verify all surviving code meets the coding conventions

## Phase 9: Dev Ergonomics Completion

- [x] Add a canonical local runtime command that brings up dependencies and the API server
- [x] Add a canonical backend-root `.env.example` aligned with runtime settings
- [x] Update the README and backlog to describe the one-command developer workflow
