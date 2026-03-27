# BotAdvisor Architecture

## Purpose

BotAdvisor is the future canonical Fitvise backend.
It replaces the legacy `backend/app` runtime with a maintainable hybrid modular monolith built for a solo developer.

The system keeps the following first-class capabilities:

- document ingestion
- embedding and vector upsert
- LlamaIndex-based hybrid retrieval on Weaviate
- thin chat and query APIs
- observability through LangFuse and structured logging

## Architectural Principles

- `backend/botadvisor` is the only long-term runtime target.
- `backend/app` is migration source material, not the future architecture.
- Retrieval is the core product capability. Everything else serves retrieval quality and safe delivery.
- LangChain is used for orchestration and tool-calling, not as the retrieval core.
- Weaviate is the default production vector store because hybrid search is a hard requirement.
- Scripts remain first-class operational entry points for ingestion and indexing.
- API remains thin. Business logic belongs in focused modules, not routers.

## Canonical Shape

BotAdvisor follows a hybrid modular monolith:

- shared foundations live in small common modules
- product behavior lives in focused feature modules
- runtime composition happens at explicit boundaries

### Shared Foundation Modules

- `core`
  lightweight settings, domain DTOs, shared types, and errors
- `observability`
  LangFuse and structured logging
- `storage`
  raw artifact persistence and dedupe support

### Feature Modules

- `ingestion`
  document loading, normalization, chunk creation
- `retrieval`
  LlamaIndex retriever contract, Weaviate integration, metadata filtering, citations
- `chat`
  query orchestration, prompt assembly, response shaping
- `platform`
  source-specific adapters, with `filesystem` as the only first-class adapter in the initial scope

## Dependency Direction

- routers depend on chat or health services
- chat services depend on retrieval and llm services
- retrieval depends on vector-store adapters and shared contracts
- ingestion depends on storage and shared contracts
- observability is imported by runtime edges and services, but never owns business logic
- no feature module may depend on legacy `backend/app`

## Runtime Model

### Script Runtime

Scripts are canonical operational entry points for:

- ingestion
- embedding and upsert
- vector-store bootstrap

This keeps operational flows explicit and easy to test.

### API Runtime

The API surface should stay minimal:

- `GET /health`
- `POST /chat`
- optional `POST /query` if it improves separation from chat

The API must not absorb ingestion or maintenance behavior.

## Retrieval Core

Retrieval is built on:

- `LlamaIndex` as the retrieval engine
- `Weaviate` as the default vector store
- metadata-aware retrieval with citations
- hybrid search as a design requirement

LangChain consumes retrieval through a thin adapter or tool boundary.
LangChain must not own the retriever implementation.

## LLM Integration

The LLM path should be simple:

- a small factory selects the active chat model
- default target can remain Ollama Cloud
- local fallback is optional, not the primary path
- chat orchestration is thin and retrieval-aware

LangGraph is deferred until a real multi-step graph is required.
Initial orchestration should stay LangChain-first.

## Observability

Observability is mandatory but lightweight:

- LangFuse traces for script and API execution
- structured logging with request correlation
- health endpoints for the runtime core

Grafana, Prometheus, and wider ops stacks are not required for the initial solo-developer operating model.

## Legacy Strategy

`backend/app` currently contains reusable implementation ideas for:

- LlamaIndex retriever wrapping
- RAG chat orchestration
- Weaviate bootstrap and schema management
- health endpoints and service wiring

These are migration sources only.
They should be reimplemented or selectively ported into `backend/botadvisor` under the new quality bar, then archived.

## Out of Scope For The Initial Canonical Runtime

- multiple first-class source adapters
- generalized plugin-like extension systems
- over-layered dependency injection containers
- large clean-architecture boilerplate trees
- heavyweight orchestration frameworks by default

## Success Condition

The architecture is successful when:

- `backend/botadvisor` alone can ingest, index, retrieve, and answer
- retrieval uses LlamaIndex plus Weaviate hybrid search
- the API is thin and understandable
- legacy code is no longer part of the live runtime path
- the codebase remains small enough for a solo developer to reason about quickly
