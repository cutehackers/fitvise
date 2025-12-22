# Fitvise Backend

> **AI-Powered Fitness API** - Intelligent workout planning and health guidance through advanced language models with RAG pipeline

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.11.7-red.svg)](https://pydantic-docs.helpmanual.io/)
[![Testing](https://img.shields.io/badge/pytest-comprehensive-green.svg)](https://pytest.org/)

## 🌟 Overview

Fitvise Backend is a sophisticated REST API that leverages Large Language Models (LLMs) to provide personalized fitness coaching, workout planning, and health guidance. Built with modern Python technologies and Domain-Driven Design principles, it features a production-grade RAG (Retrieval-Augmented Generation) pipeline for intelligent document processing and retrieval.

### Key Features

🤖 **AI-Powered Fitness Coaching** - Generate personalized workout plans, nutrition advice, and exercise recommendations
🏗️ **Domain-Driven Design** - Modular FastAPI structure with clear separation of concerns and DDD principles
📚 **RAG Pipeline** - Multi-phase document ingestion, embedding generation, and retrieval with comprehensive orchestration
📊 **Health Monitoring** - Comprehensive health checks and service availability monitoring
🔒 **Type Safety** - Full Pydantic validation for requests and responses with comprehensive data modeling
⚡ **High Performance** - Async HTTP client with connection pooling, timeout management, and batch operations
🧪 **Comprehensive Testing** - Unit, integration, and E2E tests with 30+ tests for RAG pipeline phases
📚 **Auto-Generated Documentation** - Interactive Swagger UI and ReDoc documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LLM service (e.g., [Ollama](https://ollama.ai/))
- SQLite or PostgreSQL for data persistence
- Weaviate for vector embeddings and semantic search

### Installation

0. **Python dependency management**:
```bash
pip freeze > requirements.txt
```

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd fitvise/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
# Create .env file
cat > .env << EOF
APP_NAME=Fitvise Backend API
APP_VERSION=1.0.0
APP_DESCRIPTION=AI-powered fitness API

# LLM Configuration
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
LLM_TIMEOUT=30

# API Configuration
ENVIRONMENT=local
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
API_V1_PREFIX=/api/v1

# Database Configuration
DATABASE_URL=sqlite+aiosqlite:///./fitvise.db
# For PostgreSQL: DATABASE_URL=postgresql+asyncpg://user:password@localhost/fitvise

# Vector Store Configuration (Weaviate)
WEAVIATE_HOST=http://localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051

# RAG Pipeline Configuration
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=100
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# CORS Configuration
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=*

# Logging
LOG_LEVEL=INFO
EOF
```

3. **Start the server**:
```bash
python run.py
# Or with uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the API**:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/workout/health

## 📚 API Documentation

### Core Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/v1/workout/health` | Service health monitoring |
| `POST` | `/api/v1/workout/prompt` | AI fitness prompts |
| `GET` | `/api/v1/workout/models` | Available model information |

### Example Usage

**Generate a workout plan**:
```bash
curl -X POST 'http://localhost:8000/api/v1/workout/prompt' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Create a 30-minute upper body workout for beginners",
    "context": "I have dumbbells and limited time",
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

**Health check**:
```bash
curl -X GET 'http://localhost:8000/api/v1/workout/health'
```

For detailed API documentation, see [API.md](API.md) or visit `/docs` when running.

## 🏗️ Architecture

### Project Structure

```
app/
├── api/v1/                      # API version 1
│   ├── endpoints/               # Endpoint implementations
│   │   └── workout.py           # Workout-related endpoints
│   ├── schemas.py               # Pydantic request/response models
│   └── router.py                # API router aggregation
├── application/                 # Application layer (use cases)
│   ├── use_cases/               # Business logic orchestration
│   └── dto/                     # Data transfer objects
├── core/
│   ├── config.py                # Configuration management
│   └── settings.py              # Environment settings
├── domain/                      # Domain layer (entities, value objects)
│   ├── entities/                # Core domain entities
│   ├── value_objects/           # Immutable domain concepts
│   ├── repositories/            # Repository interfaces
│   └── exceptions.py            # Domain-specific exceptions
├── infrastructure/              # Infrastructure layer
│   ├── repositories/            # Repository implementations
│   │   ├── container.py         # RepositoryContainer for DI
│   │   ├── dependencies.py      # FastAPI dependencies
│   │   └── implementations/     # Concrete repository implementations
│   ├── database/                # Database setup and models
│   ├── services/                # External service integrations
│   │   ├── llm_service.py       # LLM API client
│   │   ├── embedding_service.py # Embedding generation
│   │   └── vector_store.py      # Vector database integration
│   └── migrations/              # Database migrations
├── pipeline/                    # RAG Pipeline
│   ├── workflow.py              # Pipeline orchestrator
│   ├── config.py                # Pipeline configuration
│   └── phases/                  # Pipeline phases
│       ├── rag_infrastructure_task.py
│       ├── rag_ingestion_task.py
│       └── rag_embedding_task.py
├── schemas/                     # Shared Pydantic models
│   ├── chat.py                  # LLM request/response schemas
│   └── search.py                # Search-related schemas
└── main.py                      # FastAPI application
```

### Core Components

**RAG Pipeline** (`app/pipeline/`)
- **Phase 1 (Infrastructure)**: Initializes vector databases and validates external services
- **Phase 2 (Ingestion)**: Discovers documents, chunks with semantic strategies, stores in repository
- **Phase 3 (Embedding)**: Generates embeddings using sentence transformers, deduplicates, stores in vector DB
- **Orchestrator**: Coordinates phases with shared state and comprehensive error handling

**Domain Layer** (`app/domain/`)
- **Entities**: Document, Chunk, Embedding, ProcessingJob with unique identities
- **Value Objects**: ChunkMetadata, DocumentMetadata, ChunkLoadPolicy (immutable)
- **Repositories**: Abstract interfaces for data persistence
- **Exceptions**: Domain-specific exceptions for error handling

**Application Layer** (`app/application/`)
- **Use Cases**: Business logic orchestration for fitness coaching and RAG operations
- **DTOs**: Data transfer objects for internal communication
- **Query Handlers**: Query processing for read operations

**Infrastructure Layer** (`app/infrastructure/`)
- **RepositoryContainer**: Dependency injection pattern for repositories
- **Database**: SQLAlchemy async session management with SQLite/PostgreSQL support
- **LLM Service** (`app/infrastructure/services/llm_service.py`)
  - Async HTTP client for LLM API communication
  - Request/response processing with error handling
  - Token usage tracking and performance metrics
  - Health monitoring and timeout management

**Configuration** (`app/core/config.py`, `app/core/settings.py`)
- Pydantic-based settings management
- Environment variable loading with comprehensive validation
- Type-safe configuration access
- CORS and security settings
- Database and vector store configuration

**API Endpoints** (`app/api/v1/endpoints/`)
- RESTful endpoint implementations following REST principles
- Request validation and response formatting with Pydantic
- Error handling with appropriate HTTP status codes
- Health monitoring and service status reporting
- Dependency injection using FastAPI's Depends()

## 🔧 Configuration

### Environment Variables

#### **Application & API**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_NAME` | Application name | - | ✅ |
| `APP_VERSION` | API version | - | ✅ |
| `ENVIRONMENT` | Deployment environment (local, staging, production) | `local` | ✅ |
| `DEBUG` | Debug mode | `false` | ✅ |
| `API_HOST` | Server host | `0.0.0.0` | ✅ |
| `API_PORT` | Server port | `8000` | ✅ |
| `API_V1_PREFIX` | API v1 prefix | `/api/v1` | ✅ |

#### **LLM Integration**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_BASE_URL` | LLM service URL | - | ✅ |
| `LLM_MODEL` | Model identifier | - | ✅ |
| `LLM_TEMPERATURE` | Response creativity (0.0-2.0) | `0.7` | ✅ |
| `LLM_MAX_TOKENS` | Max response length | `1000` | ✅ |
| `LLM_TIMEOUT` | Request timeout (seconds) | `30` | ✅ |

#### **Database**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | Database connection URL | `sqlite+aiosqlite:///./fitvise.db` | ✅ |

#### **Vector Store & Embeddings**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `WEAVIATE_HOST` | Weaviate instance host | `http://localhost` | ✅ |
| `WEAVIATE_PORT` | Weaviate REST API port | `8080` | ✅ |
| `WEAVIATE_GRPC_PORT` | Weaviate gRPC port | `50051` | ✅ |
| `EMBEDDING_MODEL` | Sentence transformer model name | `sentence-transformers/all-MiniLM-L6-v2` | ✅ |

#### **RAG Pipeline**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `RAG_CHUNK_SIZE` | Document chunk size in tokens | `500` | ✅ |
| `RAG_CHUNK_OVERLAP` | Chunk overlap in tokens | `100` | ✅ |

#### **CORS & Security**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CORS_ORIGINS` | Allowed CORS origins | `*` | ✅ |
| `CORS_ALLOW_CREDENTIALS` | Allow credentials in CORS | `true` | ✅ |
| `CORS_ALLOW_METHODS` | Allowed HTTP methods | `GET,POST,PUT,DELETE,OPTIONS` | ✅ |
| `CORS_ALLOW_HEADERS` | Allowed headers | `*` | ✅ |

#### **Logging**
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOG_LEVEL` | Logging level | `INFO` | ✅ |

### LLM Service Setup

The API requires a compatible LLM service. For development, we recommend [Ollama](https://ollama.ai/):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start the service (runs on localhost:11434)
ollama serve
```

## 📊 Features

### Health Monitoring

- **Service Health**: `/health` endpoint monitors API and LLM service status
- **Status Levels**: healthy, degraded, unhealthy
- **Metrics**: Response times, token usage, success rates
- **Logging**: Structured logging with request tracing

### Error Handling

- **Input Validation**: Comprehensive request validation with Pydantic
- **HTTP Status Codes**: Appropriate status codes for different error types
- **Error Response Format**: Consistent error structure across all endpoints
- **Timeout Management**: Graceful handling of LLM service timeouts

### Performance

- **Async Operations**: Non-blocking I/O for high concurrency
- **Connection Pooling**: Efficient HTTP client with connection reuse
- **Request Timeouts**: Configurable timeouts to prevent hanging requests
- **Resource Limits**: Configurable limits for tokens and request size

## 🧪 Testing

### Test Structure

All tests are located in the `/tests/` directory organized by type:

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── pipeline/                  # RAG pipeline phase tests
│   │   ├── test_rag_embedding_task.py      # Phase 3: 30+ embedding tests
│   │   └── test_rag_ingestion_task.py      # Phase 2: Ingestion tests
│   ├── application/               # Use case and business logic tests
│   ├── infrastructure/            # Repository and service tests
│   ├── domain/                    # Entity and value object tests
│   └── ...
├── integration/                   # Integration tests for component interactions
├── e2e/                          # End-to-end tests for complete workflows
├── fixtures/                      # Shared test fixtures and sample data
├── utils/                         # Test utilities and helpers
├── performance/                   # Performance and load tests
└── conftest.py                   # Global pytest configuration
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/unit/pipeline/test_rag_embedding_task.py -v

# Run specific test class
pytest tests/unit/pipeline/test_rag_embedding_task.py::TestEmbeddingExecution -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html

# Run with detailed output
pytest tests/ -vv --tb=short

# Run only fast tests (exclude slow tests)
pytest tests/ -m "not slow"

# Run with specific markers
pytest tests/ -m "integration"  # Only integration tests
pytest tests/ -m "unit"         # Only unit tests
```

### Test Organization Rules

✅ **Unit tests**: `/tests/unit/` - Test individual components in isolation
✅ **Integration tests**: `/tests/integration/` - Test component interactions
✅ **E2E tests**: `/tests/e2e/` - Test complete user workflows
✅ **Pipeline tests**: `/tests/unit/pipeline/` - RAG pipeline phase tests
❌ **Never** place tests inside `app/` directory - they belong in `/tests/`

### Recent Test Additions

- `tests/unit/pipeline/test_rag_embedding_task.py` - 30+ comprehensive tests for embedding generation phase
- `tests/unit/pipeline/test_rag_ingestion_task.py` - Tests for document ingestion phase
- All new pipeline tests follow TDD with comprehensive mocking and async support

### Manual Testing

```python
import httpx
import asyncio

async def test_fitness_api():
    async with httpx.AsyncClient() as client:
        # Health check
        health = await client.get("http://localhost:8000/api/v1/workout/health")
        print(f"Health: {health.json()}")

        # Fitness prompt
        response = await client.post(
            "http://localhost:8000/api/v1/workout/prompt",
            json={
                "query": "Quick 15-minute morning workout routine",
                "context": "Beginner level, no equipment needed"
            }
        )
        print(f"Workout: {response.json()['response']}")

asyncio.run(test_fitness_api())
```

### Example Test Client

```bash
python api_example.py
```

## 🔌 Dependency Injection & InfraContainer

The application uses the **InfraContainer** pattern for flexible dependency injection that works in FastAPI endpoints, scripts, and pipeline phases.

### Using in FastAPI Endpoints

```python
from fastapi import Depends
from app.di.containers.infra_container import InfraContainer
from app.domain.repositories import DocumentRepository

@router.post("/documents")
async def create_document(
    # Inject InfraContainer and access repositories
    container: InfraContainer = Depends(get_infra_container),
):
    document = await container.document_repository().save(new_document)
    return document

@router.get("/documents")
async def list_documents(
    # Inject InfraContainer for multiple repositories
    container: InfraContainer = Depends(get_infra_container),
):
    documents = await container.document_repository().find_all()
    sources = await container.data_source_repository().find_all()
    return {"documents": documents, "sources": sources}
```

### Using in Scripts and Pipeline

```python
import asyncio
from app.di.containers.infra_container import InfraContainer
from app.infrastructure.database.database import AsyncSessionLocal
from app.pipeline.phases.rag_ingestion_task import IngestionPhase

async def run_pipeline():
    """Run RAG pipeline with InfraContainer."""
    container = InfraContainer()

    async with AsyncSessionLocal() as session:
        container.db_session.override(session)

        # Pass to pipeline phases
        phase = IngestionPhase(
            document_repository=container.document_repository(),
            data_source_repository=container.data_source_repository(),
        )

        await phase.execute(spec)
        await session.commit()

asyncio.run(run_pipeline())
```

### Configuration-Based Repository Selection

The container automatically selects repository implementation based on `database_type` configuration:

- **In-Memory Mode**: `default` (InMemoryDocumentRepository, ideal for testing)
- **SQLite**: `aiosqlite` (SQLAlchemyDocumentRepository)
- **PostgreSQL**: `asyncpg` (SQLAlchemyDocumentRepository)
- **MySQL**: `aiomysql` or `asyncmy` (SQLAlchemyDocumentRepository)

## 🎯 Critical Development Patterns

### Async/Await Usage

✅ **All database operations are async** - Use `await` with repository methods
✅ **All external service calls are async** - LLM, embedding services, vector stores
✅ **Use AsyncMock in tests** - Not Mock, for async methods
✅ **AsyncSession context managers** - Properly manage SQLAlchemy async sessions
❌ **Never use sync operations in async functions** - Blocks event loop

### RAG Pipeline Testing

✅ **Mock external dependencies** - Embedding services, vector stores, repositories
✅ **Use real dataclasses** - EmbeddingResult, RagEmbeddingTaskReport are real models
✅ **Test success and failure paths** - Especially Phase 2 → Phase 3 handover
✅ **Validate chunk availability** - Phase 3 assumes Phase 2 completed successfully
✅ **Track deduplication stats** - Validate duplicate removal calculations

### Repository Access Patterns

✅ **Use InfraContainer for DI** - Provides consistent repository access
✅ **Access via FastAPI Depends** - `Depends(get_infra_container)` in endpoints
✅ **Manual creation in scripts** - With Settings and AsyncSession
✅ **Mock in tests with AsyncMock** - Realistic return values

### Error Handling

✅ **Domain exceptions** - ChunkingError, EmbeddingGenerationError, etc.
✅ **Custom pipeline exceptions** - EmbeddingPipelineError, IngestionPipelineError
✅ **Catch and log at boundaries** - FastAPI routes, pipeline orchestrator
✅ **Return structured errors** - Always include error details for debugging

## 📋 Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Development server (with auto-reload)
python run.py

# Alternative with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (without reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing and Validation

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test module
pytest tests/unit/pipeline/test_rag_embedding_task.py -v

# Run linting and type checking
mypy app/
pylint app/
black --check app/

# Format code
black app/
```

## 🔐 Security

### Input Validation

- Query length limited to 1000 characters
- Context length limited to 2000 characters  
- Parameter ranges validated (temperature: 0.0-2.0, tokens: 50-2000)
- Type validation with Pydantic models

### Production Considerations

- **CORS Configuration**: Configure specific origins for production
- **Rate Limiting**: Implement request throttling
- **Authentication**: Add API key or JWT authentication
- **HTTPS**: Use TLS encryption in production
- **Input Sanitization**: Additional validation for user inputs

## 📈 Performance & Monitoring

### Metrics

- **Token Usage**: Track prompt and completion tokens
- **Response Times**: Monitor generation duration
- **Success Rates**: Track successful vs failed requests
- **Error Patterns**: Identify common failure modes

### Logging

- **Structured Logging**: JSON format for production
- **Request Tracing**: Unique request IDs for debugging
- **Performance Metrics**: Response time and resource usage
- **Error Tracking**: Comprehensive error logging

## 📊 API Data Modeling

All API data models follow a consistent naming and structure convention for clarity and maintainability.

### Naming Patterns

#### Schema/Model Types
- **Request Models**: `{Action}{Resource}Request` (e.g., `PromptRequest`, `EmbedChunksRequest`)
- **Response Models**: `{Action}{Resource}Response` (e.g., `PromptResponse`, `EmbedChunksResponse`)
- **Input Models**: `{Resource}Input` (e.g., `ChunkInput`)
- **Result Models**: `{Resource}Result` (e.g., `EmbeddingResult`)
- **Configuration Models**: `{Service}Config` (e.g., `EmbeddingConfig`)

#### Field Naming
- **Case**: All fields use `snake_case` (e.g., `processing_time_ms`)
- **Time fields**: Use `_ms` for milliseconds, `_seconds` for longer durations
- **Count fields**: Use `_count` suffix (e.g., `total_chunks`, `failed_count`)
- **ID fields**: Use `_id` suffix (e.g., `chunk_id`, `session_id`)
- **Boolean flags**: Use `is_`, `has_`, `can_` prefix (e.g., `is_successful`, `has_metadata`)

### Schema Design Principles

#### Request Model Structure
```python
class PromptRequest(BaseModel):
    """Request model for LLM prompt operation."""

    # Required fields first
    query: str = Field(..., min_length=1, max_length=1000, description="User query")

    # Optional fields with sensible defaults
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="Max response length")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "query": "Create a 30-minute workout",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
```

#### Response Model Structure
```python
class PromptResponse(BaseModel):
    """Response model for LLM prompt operation."""

    # Success/error status
    success: bool = Field(description="Whether operation was successful")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Core response data
    response: str = Field(description="Generated response from LLM")

    # Metadata
    processing_time_ms: float = Field(ge=0, description="Processing time in milliseconds")
    tokens_used: Dict[str, int] = Field(default_factory=dict, description="Token usage breakdown")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

### Field Validation Guidelines

✅ **String Fields**: Always include `min_length` and `max_length`
✅ **Numeric Fields**: Include `ge`/`le` constraints and `description`
✅ **List Fields**: Include `min_items`/`max_items` when applicable
✅ **Optional Fields**: Provide sensible defaults
✅ **Documentation**: Every field must have a descriptive `description`
✅ **Examples**: Include realistic examples via `json_schema_extra`

### File Organization

```
app/
├── schemas/                        # Shared schemas
│   ├── chat.py                    # LLM request/response schemas
│   └── search.py                  # Search-related schemas
├── api/v1/{module}/
│   └── schemas.py                 # Module-specific schemas
└── application/dto/
    └── {service}_dto.py           # Internal DTOs (not exposed to API)
```

## 🏛️ Domain-Driven Design (DDD) Architecture

The codebase follows DDD principles with clear domain boundaries and layers:

### Domain Layer (`app/domain/`)

**Entities**: Objects with unique identity that persist over time
- `Document`: Core domain entity for documents
- `Chunk`: Text chunks from documents with metadata
- `Embedding`: Vector embeddings of text chunks
- `ProcessingJob`: Background processing task

**Value Objects**: Immutable domain concepts without unique identity
- `ChunkMetadata`: Immutable chunk metadata
- `DocumentMetadata`: Immutable document metadata
- `ChunkLoadPolicy`: Strategy for loading chunks with fallback handling

**Repositories**: Abstract interfaces for data persistence
```python
class DocumentRepository(Protocol):
    """Repository interface for documents."""

    async def save(self, document: Document) -> Document: ...
    async def find_by_id(self, document_id: str) -> Optional[Document]: ...
    async def find_all(self) -> List[Document]: ...
    async def delete(self, document_id: str) -> bool: ...
```

**Exceptions**: Domain-specific exceptions
- `DocumentNotFoundError`: Document not found in repository
- `ChunkingError`: Error during document chunking
- `EmbeddingGenerationError`: Error generating embeddings
- `ValidationError`: Domain object validation failure

### Application Layer (`app/application/`)

**Use Cases**: Business logic orchestration
- Coordinate domain objects to implement business processes
- Translate external requests to domain operations
- Handle transaction boundaries and error recovery

**DTOs**: Data transfer objects for internal communication
- Shield domain objects from external changes
- Structure data for specific use cases
- Provide type-safe communication between layers

### Infrastructure Layer (`app/infrastructure/`)

**Repository Implementations**: Concrete implementations of repository interfaces
- SQLAlchemy ORM models for database persistence
- In-memory repositories for testing
- Vector database integration for embeddings

**External Services**:
- `LLMService`: Communicates with external LLM APIs
- `EmbeddingService`: Generates embeddings using transformers
- `VectorStore`: Manages vector database operations

**Database**: SQLAlchemy async session management and models

### API Layer (`app/api/`)

**Schemas**: Pydantic models for HTTP request/response validation
**Routes**: HTTP endpoint handlers with FastAPI
**Dependencies**: FastAPI dependency injection for repository access

## 🤝 Contributing

### Development Guidelines

- **Type Hints**: All functions must have type annotations
- **Async/Await**: Use async for all I/O operations (database, HTTP, services)
- **Pydantic Models**: Use for all request/response structures with comprehensive validation
- **Error Handling**: Domain-specific exceptions with proper error context
- **Documentation**: Update API.md for endpoint changes and CLAUDE.md for architecture changes

### DDD Principles

- **Domain Layer**: Keep business logic in domain entities and use cases
- **Repository Pattern**: Use abstract repository interfaces for data access
- **Value Objects**: Use immutable value objects for domain concepts
- **Domain Exceptions**: Use domain-specific exceptions, not generic exceptions

### RAG Pipeline Development

- **Phase Dependencies**: Validate phase handover (Phase 2 → Phase 3)
- **Chunk Management**: Use ChunkLoadPolicy for fallback strategies
- **Embedding Tracking**: Track deduplication and embedding statistics
- **Error Recovery**: Implement graceful error handling with meaningful messages

### Testing Requirements

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions (especially phase handover)
- **Test Organization**: Place all tests in `/tests/` directory, never in `/app/`
- **Async Testing**: Use `@pytest.mark.asyncio` and AsyncMock for async code
- **Coverage**: Maintain ≥80% unit test coverage

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for all public functions and modules
- Maintain consistent error handling patterns across layers
- Use proper type hints including Optional, Union, Dict, List

## 📚 Additional Resources

- **[API Documentation](API.md)** - Detailed endpoint documentation
- **[CLAUDE.md](CLAUDE.md)** - Comprehensive development guide with architecture details
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - Framework documentation
- **[Pydantic Documentation](https://docs.pydantic.dev/)** - Validation and serialization
- **[SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)** - Async database patterns
- **[Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)** - DDD principles
- **Interactive API Docs** - Available at `/docs` when running
- **ReDoc** - Alternative API documentation at `/redoc` when running

## 📄 License

This project is part of the Fitvise fitness application suite.

---

**Built with ❤️ for intelligent fitness coaching**
