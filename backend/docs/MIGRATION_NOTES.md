# Migration Notes: Fitvise → BotAdvisor

## Overview

This document describes the migration from the legacy Fitvise RAG pipeline to the new BotAdvisor architecture.

## Deprecated Files

The following files have been moved to `backend/deprecated/`:

### Docker & Infrastructure
- `docker-compose.yml` → `deprecated/docker-compose.yml`
- `prometheus.yml` → `deprecated/prometheus.yml`
- `grafana/` → `deprecated/grafana/`

### Legacy Code
- `app/pipeline/` → `deprecated/pipeline/`
- `app/infrastructure/orchestration/` → `deprecated/infrastructure/orchestration/`
- `scripts/build_rag_pipeline.py` → `deprecated/build_rag_pipeline.py`

### Documentation
- `docs/rag_phase1.md` → `deprecated/old_docs/rag_phase1.md`
- `docs/rag_phase2.md` → `deprecated/old_docs/rag_phase2.md`
- `docs/rag_phase3.md` → `deprecated/old_docs/rag_phase3.md`
- `docs/agentic-rag-backlogs.md` → `deprecated/old_docs/agentic-rag-backlogs.md`
- `docs/rag_backlogs.md` → `deprecated/old_docs/rag_backlogs.md`

## New BotAdvisor Structure

```
backend/botadvisor/
├── app/
│   ├── core/              # Document model (NEW)
│   ├── storage/           # Storage backends (NEW)
│   ├── retrieval/         # Retrieval layer (NEW)
│   ├── agent/             # LangChain agent (NEW)
│   ├── observability/     # LangFuse + logging (NEW)
│   └── api/v2/            # BotAdvisor API (NEW)
├── scripts/
│   ├── ingest.py          # Docling-based ingestion (NEW)
│   ├── embed_upsert.py    # LlamaIndex embedding (NEW)
│   ├── setup_vector_store.py  # Weaviate setup (NEW)
│   └── eval_retrieval.py  # Evaluation script (NEW)
├── configs/
│   ├── .env.example       # Environment variables (NEW)
│   └── logging.yaml       # Logging config (NEW)
├── docker-compose.yaml    # Weaviate + LangFuse (NEW)
└── README.md
```

## Module Mapping

| Legacy Module | New Module | Notes |
|--------------|-----------|-------|
| `app/pipeline/` | `botadvisor/scripts/` | Pipeline → Script-first |
| `app/infrastructure/llm/` | `botadvisor/app/retrieval/` + `botadvisor/app/agent/` | Refactored |
| `app/infrastructure/storage/` | `botadvisor/app/storage/` | Simplified |
| `app/domain/entities/document.py` | `botadvisor/app/core/types.py` | Lean Document model |
| `app/api/v1/fitvise/` | `botadvisor/app/api/v2/` | New API endpoints |
| `scripts/build_rag_pipeline.py` | `scripts/ingest.py` + `scripts/embed_upsert.py` | Split for flexibility |

## Key Changes

### 1. Architecture
- **Legacy**: Monolithic pipeline with complex orchestration
- **New**: Script-first approach with modular components

### 2. LLM
- **Legacy**: Local Ollama
- **New**: Ollama Cloud (`gemini-3-flash`) only

### 3. Observability
- **Legacy**: Prometheus + Grafana (deprecated)
- **New**: LangFuse (tracing + metrics)

### 4. Storage
- **Legacy**: Complex abstraction layers
- **New**: Simple local/MinIO backends

### 5. Ingestion
- **Legacy**: Custom chunking
- **New**: Docling-based with checksum deduplication

### 6. Retrieval
- **Legacy**: Custom implementation
- **New**: LlamaIndex with platform adapters

### 7. API
- **Legacy**: `app/api/v1/fitvise/` (Fitvise-specific)
- **New**: `botadvisor/app/api/v2/` (BotAdvisor-specific)

## Compatibility Fence

**Rule**: No imports from `deprecated/` in new code.

When porting patterns from legacy:
1. Copy the pattern, don't import
2. Adapt to new architecture
3. Simplify if possible
4. Update documentation

## Migration Steps

### For Existing Data

If you have existing Weaviate data:
```bash
# Export from legacy collection
# Import to new BotAdvisor collection
```

### For API Clients

Legacy API: `http://localhost:8000/api/v1/fitvise/chat`
New API: `http://localhost:8000/api/v2/chat`

### For Scripts

Legacy: `scripts/build_rag_pipeline.py`
New: `scripts/ingest.py` + `scripts/embed_upsert.py`

## Environment Variables

| Legacy Variable | New Variable | Notes |
|----------------|--------------|-------|
| `OLLAMA_HOST` | `OLLAMA_CLOUD_API_KEY` | Now uses cloud API |
| `PROMETHEUS_ENABLED` | `LANGFUSE_PUBLIC_KEY` | Changed observability |
| `GRAFANA_ENABLED` | `LANGFUSE_SECRET_KEY` | Changed observability |

## Docker Services

| Legacy Service | New Service | Status |
|----------------|--------------|--------|
| Weaviate | Weaviate | ✅ Kept |
| Prometheus | - | ❌ Deprecated |
| Grafana | - | ❌ Deprecated |
| Backend (Fitvise) | - | ❌ Deprecated |
| PostgreSQL | LangFuse Cloud | ⚠️ Changed to cloud |

Note: LangFuse is now accessed via [LangFuse Cloud](https://cloud.langfuse.com) - no local database required.

## Testing

After migration:
```bash
# Start services
cd botadvisor
docker-compose up -d

# Test vector store
uv run python scripts/setup_vector_store.py

# Test ingestion
uv run python scripts/ingest.py --input test_docs --out test_output --platform filesystem

# Test embedding
uv run python scripts/embed_upsert.py --input test_output

# Test API
curl -X POST http://localhost:8000/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

## Rollback

If you need to rollback:
```bash
# Stop BotAdvisor
cd botadvisor
docker-compose down

# Restore legacy docker-compose
cp deprecated/docker-compose.yml .
docker-compose up -d
```

## References

- [BotAdvisor Backlog](BOTADVISOR-BACKLOGS.md)
- [BotAdvisor Structure](BOTADVISOR-structure.md)
- [BotAdvisor Implementation Plan](BOTADVISOR-implementation-plan.md)

## Questions?

If you encounter issues during migration:
1. Check `deprecated/` for legacy implementation details
2. Consult `BOTADVISOR-BACKLOGS.md` for task details
3. Review `botadvisor/README.md` for setup instructions
