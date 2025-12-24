# BotAdvisor

A script-first RAG backend powered by Docling, LlamaIndex, LangChain, Ollama Cloud, and LangFuse.

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Ollama Cloud API key
- LangFuse account (optional)

### 1. Environment Setup

```bash
cd botadvisor
cp configs/.env.example .env
```

Edit `.env` and add your API keys:
```bash
OLLAMA_CLOUD_API_KEY=your_ollama_cloud_api_key
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 2. Start Services

```bash
docker-compose up -d
```

This starts:
- Weaviate (vector database) on port 8080

Note: LangFuse is accessed via [LangFuse Cloud](https://cloud.langfuse.com). No local setup required.

### 3. Initialize Vector Store

```bash
uv run python scripts/setup_vector_store.py
```

### 4. Ingest Documents

```bash
# Ingest from local filesystem
uv run python scripts/ingest.py \
  --input ./documents \
  --out ./data/chunks \
  --platform filesystem
```

### 5. Generate Embeddings

```bash
uv run python scripts/embed_upsert.py \
  --input ./data/chunks \
  --batch-size 32
```

### 6. Start API Server

```bash
uvicorn botadvisor.app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for API documentation.

## 📁 Project Structure

```
botadvisor/
├── app/
│   ├── core/              # Document model, types
│   ├── storage/           # local_storage, minio_client
│   ├── retrieval/         # retriever, registry, adapters
│   ├── agent/             # assembler, prompts
│   ├── observability/     # langfuse, logging
│   └── api/v2/            # FastAPI endpoints
├── scripts/
│   ├── ingest.py          # Docling ingestion
│   ├── embed_upsert.py    # LlamaIndex embedding
│   ├── setup_vector_store.py  # Initialize Weaviate
│   └── eval_retrieval.py  # Quality evaluation
├── configs/
│   ├── .env.example       # Environment variables
│   └── logging.yaml       # Logging configuration
├── docker-compose.yaml    # Weaviate + LangFuse
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_CLOUD_MODEL` | LLM model name | `gemini-3-flash` |
| `WEAVIATE_HOST` | Weaviate host | `localhost` |
| `WEAVIATE_PORT` | Weaviate port | `8080` |
| `STORAGE_BACKEND` | Storage backend | `local` |
| `EMBEDDING_MODEL_NAME` | Embedding model | `Alibaba-NLP/gte-multilingual-base` |

### Supported Platforms

- **filesystem**: Local files
- **web**: Web pages (coming soon)
- **gdrive**: Google Drive (coming soon)

## 📊 Observability

### LangFuse Cloud

- Access: https://cloud.langfuse.com
- Sign up for free at: https://langfuse.com
- Add your public/secret keys to `.env` file

## 🛠️ Development

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
black --check .
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (deletes all data)
docker-compose down -v
```

## 📝 Backlog

See [../docs/BOTADVISOR-BACKLOGS.md](../docs/BOTADVISOR-BACKLOGS.md) for detailed task tracking.

## 🔄 Migration

See [../docs/MIGRATION_NOTES.md](../docs/MIGRATION_NOTES.md) for migration from legacy Fitvise code.

## 🤝 Contributing

1. Create a feature branch
2. Implement changes
3. Add tests
4. Submit PR

## 📄 License

MIT
