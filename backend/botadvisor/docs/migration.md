# BotAdvisor Migration Strategy

## Migration Goal

Move Fitvise backend ownership from the legacy backend runtime to the canonical `backend/botadvisor` runtime without losing useful implementation knowledge.

## Source And Target

- source: `backend/deprecated/legacy_backend/app`, `backend/deprecated/legacy_backend/scripts`, `backend/deprecated/legacy_backend/docs`
- target: `backend/botadvisor`

## Migration Rules

- Do not import legacy runtime modules into final `botadvisor` code.
- Port behavior, not architecture.
- Migrate only code that still helps the solo-developer target.
- Archive historical docs instead of keeping them as live references.

## Legacy Assets To Mine

### Reusable Implementation Sources

- `backend/deprecated/legacy_backend/app/infrastructure/llm/services/llama_index_retriever.py`
- `backend/deprecated/legacy_backend/app/infrastructure/llm/services/rag_orchestrator.py`
- `backend/deprecated/legacy_backend/app/api/v1/fitvise/chat.py`
- `backend/deprecated/legacy_backend/scripts/setup_weaviate_schema.py`

### Legacy Docs To Absorb

- `backend/deprecated/legacy_backend/docs/BOTADVISOR-structure.md`
- `backend/deprecated/legacy_backend/docs/BOTADVISOR-implementation-plan.md`
- `backend/deprecated/legacy_backend/docs/technical_architecture.md`
- `backend/deprecated/legacy_backend/docs/llm_architecture.md`
- `backend/deprecated/legacy_backend/docs/rag_phase1.md`
- `backend/deprecated/legacy_backend/docs/rag_phase2.md`
- `backend/deprecated/legacy_backend/docs/rag_phase3.md`
- `backend/deprecated/legacy_backend/docs/rag_phase4.md`
- `backend/deprecated/legacy_backend/docs/rag_phase5_core_presentation.md`
- `backend/deprecated/legacy_backend/docs/rag_phase5_core_tests.md`

## Absorption Map

- product and scope intent moves into `product_scope.md`
- architecture intent moves into `architecture.md`
- coding quality expectations move into `coding_conventions.md`
- status and implementation reality move into `backlog.md`
- execution order moves into `tasks.md`

## Archive Plan

### Stage 1

- mark old docs as superseded
- stop treating archived legacy docs as a planning source of truth

### Stage 2

- migrate reusable legacy code into `backend/botadvisor`
- add tests in canonical locations
- remove runtime dependencies on the live legacy runtime

## Phase 7 Inventory

The last runtime behaviors that still materially depended on the legacy backend were:

- LLM client selection and health behavior from `app/infrastructure/external_services/ml_services/llm_services/ollama_service.py`
- retrieval-aware prompt and answer orchestration from `app/application/use_cases/chat/rag_chat_use_case.py`
- historical planning and implementation docs under `backend/deprecated/legacy_backend/docs`

Current Phase 7 status:

- canonical LLM path now lives in `backend/botadvisor/app/llm`
- canonical retrieval-aware prompt assembly now lives in `backend/botadvisor/app/chat/prompting.py`
- `backend/deprecated/legacy_backend/docs/*.md` are superseded historical references only
- the legacy backend runtime is archived under `backend/deprecated/legacy_backend`

### Stage 3

- move old docs into the legacy archive
- archive or remove the live legacy runtime tree
- keep only small migration notes for future reference

## Removal Criteria For The Legacy Backend

The legacy backend can be archived when:

- `botadvisor` provides ingestion, embed/upsert, retrieval, and chat or query endpoints
- health and observability exist in `botadvisor`
- no runtime command depends on the live legacy tree
- critical tests for the canonical path exist

## Risks

- copying legacy abstractions wholesale will recreate the same maintenance problem
- leaving both runtimes active too long will confuse future work
- calling partially migrated code `done` will cause backlog drift again
