# BotAdvisor Product Scope

## Objective

Build a maintainable Fitvise backend for a solo developer by converging all future backend work into `backend/botadvisor`.

## In Scope

- document ingestion from `filesystem`
- raw artifact storage with dedupe support
- embedding and vector upsert
- Weaviate-first retrieval using LlamaIndex
- hybrid search as a core requirement
- thin chat and health APIs
- LangFuse tracing
- structured logging
- migration path away from legacy `backend/app`

## Initial Production Defaults

- vector store: `Weaviate`
- retrieval engine: `LlamaIndex`
- orchestration: `LangChain`
- primary source adapter: `filesystem`
- observability: `LangFuse` plus structured logs

## Deprioritized Or Deferred

- `web` and `gdrive` as first-class adapters
- LangGraph as a default orchestration layer
- generalized multi-provider plugin systems
- broad analytics or offline evaluation systems
- large operational stacks beyond lightweight health and tracing

## Explicit Non-Goals

- preserving every legacy abstraction from `backend/app`
- supporting many runtime modes equally from day one
- maximizing configurability at the cost of clarity
- building a framework-like backend before the core RAG path is clean

## Definition Of “Simple Enough For Solo Development”

The project is simple enough when:

- one canonical runtime path exists
- retrieval and chat flows can be explained in a few modules
- scripts stay operationally useful without hidden orchestration
- health and traces are easy to inspect
- legacy code is archived instead of coexisting as a second backend

## Scope Decisions For Existing Backlog

- retrieval remains core and is not optional
- observability remains core and is not optional
- multi-adapter extensibility is deferred
- advanced evaluation is deferred
- legacy cleanup is required before the redesign is considered complete
