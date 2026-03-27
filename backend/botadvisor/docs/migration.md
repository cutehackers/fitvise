# BotAdvisor Migration Strategy

## Migration Goal

Move Fitvise backend ownership from the legacy `backend/app` runtime to the canonical `backend/botadvisor` runtime without losing useful implementation knowledge.

## Source And Target

- source: `backend/app`, `backend/scripts`, `backend/docs`
- target: `backend/botadvisor`

## Migration Rules

- Do not import legacy runtime modules into final `botadvisor` code.
- Port behavior, not architecture.
- Migrate only code that still helps the solo-developer target.
- Archive historical docs instead of keeping them as live references.

## Legacy Assets To Mine

### Reusable Implementation Sources

- `backend/app/infrastructure/llm/services/llama_index_retriever.py`
- `backend/app/infrastructure/llm/services/rag_orchestrator.py`
- `backend/app/api/v1/fitvise/chat.py`
- `backend/scripts/setup_weaviate_schema.py`

### Legacy Docs To Absorb

- `backend/docs/BOTADVISOR-structure.md`
- `backend/docs/BOTADVISOR-implementation-plan.md`
- `backend/docs/technical_architecture.md`
- `backend/docs/llm_architecture.md`
- `backend/docs/rag_phase1.md`
- `backend/docs/rag_phase2.md`
- `backend/docs/rag_phase3.md`
- `backend/docs/rag_phase4.md`
- `backend/docs/rag_phase5_core_presentation.md`
- `backend/docs/rag_phase5_core_tests.md`

## Absorption Map

- product and scope intent moves into `product_scope.md`
- architecture intent moves into `architecture.md`
- coding quality expectations move into `coding_conventions.md`
- status and implementation reality move into `backlog.md`
- execution order moves into `tasks.md`

## Archive Plan

### Stage 1

- mark old docs as superseded
- stop treating `backend/docs` as a planning source of truth

### Stage 2

- migrate reusable legacy code into `backend/botadvisor`
- add tests in canonical locations
- remove runtime dependencies on `backend/app`

### Stage 3

- move old docs into `deprecated/docs`
- archive or remove `backend/app`
- keep only small migration notes for future reference

## Removal Criteria For `backend/app`

`backend/app` can be archived when:

- `botadvisor` provides ingestion, embed/upsert, retrieval, and chat or query endpoints
- health and observability exist in `botadvisor`
- no runtime command depends on `backend/app`
- critical tests for the canonical path exist

## Risks

- copying legacy abstractions wholesale will recreate the same maintenance problem
- leaving both runtimes active too long will confuse future work
- calling partially migrated code `done` will cause backlog drift again
