# BotAdvisor Backlog

> Superseded by `backend/botadvisor/docs/backlog.md`.
> This file remains as a compatibility mirror because older project notes and prompts still reference it.

## Current Summary

The current source of truth for planning is `backend/botadvisor/docs/`.
Backlog status is reclassified using the following meanings:

- `done`: implemented and reusable in the future `botadvisor` runtime
- `partial`: implementation exists, but it does not yet satisfy the target structure or code-quality bar
- `todo`: core implementation is still missing
- `drop`: intentionally removed from solo-developer scope

## Progress

| Epic | Status |
|------|--------|
| Epic A: Foundation & Dependencies | `done`, `partial`, `done` |
| Epic B: Script-First Data Path | `done`, `done`, `partial`, `partial` |
| Epic C: Flexible Retriever Layer | `partial`, `partial`, `partial` |
| Epic D: Agentic Chat Surface | `partial`, `partial`, `partial` |
| Epic E: Observability & Ops | `partial`, `partial`, `partial` |
| Epic F: Hardening & Extensibility | `drop`, `drop`, `partial` |

## Reclassified Items

- `A1 Dependency baseline`: `done`
- `A2 Repo cleanup`: `partial`
- `A3 Document identity model`: `done`
- `B1 Docling ingestion script`: `done`
- `B2 Embedding/upsert script`: `done`
- `B3 Vector store bootstrap`: `partial`
- `B4 Storage + dedupe plumbing`: `partial`
- `C1 Retriever contract & registry`: `partial`
- `C2 LlamaIndex retriever`: `partial`
- `C3 LangChain adapter tooling`: `partial`
- `D1 LLM factory`: `partial`
- `D2 Agent & prompt`: `partial`
- `D3 FastAPI surface`: `partial`
- `E1 LangFuse tracing`: `partial`
- `E2 Structured logging`: `partial`
- `E3 Dev ergonomics`: `partial`
- `F1 Platform adapters`: `drop`
- `F2 Quality loop`: `drop`
- `F3 Legacy cleanup`: `partial`

## Why The Status Changed

- Several items were previously left as `todo` even though reusable implementation already exists in `backend/botadvisor` or `backend/app`.
- Those items are not all marked `done`, because the new solo-developer target requires stricter code quality, tighter module boundaries, and a single canonical runtime in `backend/botadvisor`.
- Any item that depends on `backend/app` but is still useful as migration source material is marked `partial`, not `todo`.

## Next Location

Use the following files for current planning and execution:

- `backend/botadvisor/docs/architecture.md`
- `backend/botadvisor/docs/coding_conventions.md`
- `backend/botadvisor/docs/product_scope.md`
- `backend/botadvisor/docs/backlog.md`
- `backend/botadvisor/docs/migration.md`
- `backend/botadvisor/docs/tasks.md`
