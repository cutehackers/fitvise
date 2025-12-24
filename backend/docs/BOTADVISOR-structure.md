# BotAdvisor Redesign - Target Folder Structure

Purpose: filesystem layout produced when executing `docs/BOTADVISOR-BACKLOGS.md` — optimized for solo-friendly, script-first RAG with Docling ingestion, LlamaIndex retrieval, LangChain agent on Ollama Cloud `gemini-3-flash`, and LangFuse observability.

```
botadvisor/
├── app/
│   ├── api/                     # Thin FastAPI surface
│   │   ├── v1/
│   │   │   ├── chat.py          # POST /api/v1/chat (streaming)
│   │   │   └── health.py        # GET /health
│   │   └── deps.py              # Minimal dependency wiring
│   │
│   ├── agent/                   # LangChain agent assembly
│   │   ├── prompts/
│   │   │   └── system_prompt.md # Fitness coach system prompt
│   │   ├── tools/
│   │   │   └── retriever.py     # Retriever tool exposure
│   │   └── assembler.py         # Builds tool-calling agent / LangGraph node
│   │
│   ├── retrieval/               # LlamaIndex retriever + adapters
│   │   ├── base.py              # Contract + registry hooks
│   │   ├── llama_index.py       # Hybrid search, platform filters, citations
│   │   ├── registry.py          # Adapter registration keyed by platform/source
│   │   └── adapters/
│   │       ├── filesystem.py    # Local files/doc folders
│   │       ├── web.py           # HTTP crawl/scrape input
│   │       └── gdrive.py        # Google Drive example adapter
│   │
│   ├── llm/                     # Model selection
│   │   ├── factory.py           # Chooses Ollama Cloud gemini-3-flash or local Ollama
│   │   └── clients/
│   │       ├── ollama_cloud.py
│   │       └── ollama_local.py
│   │
│   ├── observability/           # LangFuse + logging
│   │   ├── langfuse.py          # Callback handler wiring
│   │   └── logging.py           # Structured logging utilities
│   │
│   ├── storage/                 # Object storage for raw artifacts (dedupe)
│   │   ├── local_storage.py     # Local filesystem backend with checksum
│   │   └── minio_client.py      # MinIO backend with checksum
│   │
│   └── core/
│       ├── config.py            # Pydantic settings (env-driven)
│       └── types.py             # Document dataclass (id, source_id, platform, source_url, checksum/hash, size_bytes, mime_type, created_at) + shared DTOs
│
├── scripts/                     # Script-first ingestion/embedding/evals
│   ├── ingest.py                # Docling -> normalized chunks
│   ├── embed_upsert.py      # LlamaIndex embedding -> vector store upsert
│   ├── setup_vector_store.py    # Schema/init for Weaviate/Chroma
│   ├── eval_retrieval.py        # Offline retrieval quality loop
│   └── adapters/                # Optional script helpers per platform
│       └── __init__.py
│
├── configs/
│   ├── .env.example             # Required env vars (LLM, LangFuse, vector store)
│   ├── vectorstore/
│   │   ├── weaviate.yaml        # Class/schema defaults
│   │   └── chroma.yaml          # Local dev defaults
│   └── logging.yaml             # Local logging config
│
├── tests/
│   ├── unit/
│   │   ├── test_retrieval.py    # Platform filter, empty-result handling
│   │   ├── test_agent.py        # Agent wiring + tool calls
│   │   └── test_llm_factory.py  # Cloud/local switching
│   └── integration/
│       ├── test_ingest_and_embed.py # Script smoke tests
│       └── test_chat_flow.py        # End-to-end chat with retriever tool
│
├── docker-compose.yaml          # Dev: vector store + optional local Ollama
├── README.md                    # 5-minute setup (scripts + API)
└── docs/
    ├── BOTADVISOR-BACKLOGS.md   # Work plan (source of truth)
    └── BOTADVISOR-phase.md      # This folder structure reference
```
