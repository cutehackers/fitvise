# FITVISE, AI powered chatbot

This file provides guidance when working with code in this repository.

## Project Structure & Module Organization
Fitvise backend is centered in `app/`, with HTTP interfaces in `app/api/v1`, orchestration and services in `app/application`, domain logic in `app/domain`, infrastructure adapters in `app/infrastructure`, and shared schemas in `app/schemas`. `main.py` wires FastAPI, while `models/` keeps persisted ML assets and `docs/` houses architectural notes. Tests live under `tests/` with `unit`, `integration`, `e2e`, and reusable fixtures inside `tests/fixtures`.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (or run `./boot.sh -i` to create `.venv` and sync automatically). Start the API via `uv run python run.py` or `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` for live reload, and verify configuration using `uv run python test_settings.py`. After setup, `./boot.sh` is the quickest way to bring the server up with consistent tooling.

## Coding Style & Naming Conventions
Code targets Python 3.11+, four-space indentation, and explicit typing on public interfaces. Apply `uv run black .` (120-character lines) and `uv run isort . --profile black`; keep modules snake_case, routers in `router.py`, and Pydantic models suffixed with `Schema`. Tests and fixtures should stay snake_case with filenames like `test_workout.py` and factory helpers in `tests/fixtures/*`.

## Testing Guidelines
Execute `uv run pytest` to run the suite; the included `tests/pytest.ini` enforces 80% coverage across `app`, emitting reports in `htmlcov/` and `coverage.xml`. Use markers to focus runs, e.g. `uv run pytest -m unit` or `uv run pytest tests/integration -m "integration and not external"`. Mirror production modules in the matching test package and document any required external services with `external` or `skip_ci` markers.

## Commit & Pull Request Guidelines
Commits follow Conventional style as seen in history (`feat:`, `refactor:`, `chore:`), so prefer a type, optional scope, and imperative summary such as `feat: add workout prompt endpoint`. Separate formatting-only changes and keep related work together for easier review. Pull requests should summarise behaviour changes, note affected endpoints or configs, list verification steps (`uv run pytest`, sample curl), and link the relevant issue or doc.

## Configuration & Security Tips
Environment variables belong in `.env`; use the template in `README.md`, keep secrets out of git, and run `uv run python test_settings.py` after edits to catch misconfiguration. Respect `ENVIRONMENT` flags to avoid exposing docs in production, and never commit generated artifacts (`htmlcov/`, `models/` exports) or API credentials.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

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

**Test Structure**:
All tests are located in the `/tests/` directory organized by type:
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── pipeline/            # Pipeline phase tests (RAG workflow)
│   ├── application/         # Use case and business logic tests
│   ├── infrastructure/      # Repository and service tests
│   ├── domain/              # Entity and value object tests
│   └── ...
├── integration/             # Integration tests for component interactions
├── e2e/                     # End-to-end tests for complete workflows
├── fixtures/                # Shared test fixtures and sample data
├── utils/                   # Test utilities and helpers
├── performance/             # Performance and load tests
└── conftest.py             # Global pytest configuration
```

**Running Tests**:
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/unit/pipeline/test_rag_embedding_task.py -v

# Run specific test class
pytest tests/unit/pipeline/test_rag_embedding_task.py::TestEmbeddingExecution -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run with detailed output and short tracebacks
pytest tests/ -vv --tb=short

# Run only fast tests (exclude slow tests)
pytest tests/ -m "not slow"
```

**Test Organization Rules**:
- ✅ **Unit tests**: `/tests/unit/` - Test individual components in isolation
- ✅ **Integration tests**: `/tests/integration/` - Test component interactions
- ✅ **E2E tests**: `/tests/e2e/` - Test complete user workflows
- ✅ **Pipeline tests**: `/tests/unit/pipeline/` - RAG pipeline phase tests
- ❌ **Never** place tests inside `app/` directory - they belong in `/tests/`

**Recent Test Additions**:
- `tests/unit/pipeline/test_rag_embedding_task.py` - 30 comprehensive tests for embedding generation phase
- `tests/unit/pipeline/test_rag_ingestion_task.py` - Tests for document ingestion phase
- All new pipeline tests follow TDD with comprehensive mocking and async support

### Environment Configuration
The service requires a comprehensive `.env` file. Copy the existing `.env` and modify as needed. Key configuration areas:
- **LLM Integration**: `LLM_BASE_URL`, `LLM_MODEL` (requires Ollama or compatible service)
- **API Settings**: `API_HOST`, `API_PORT`, CORS configuration
- **Database**: `DATABASE_URL` (SQLite by default)
- **Vector Store**: ChromaDB configuration for embeddings
- **Security**: JWT configuration, file upload limits

## Architecture Overview

This is a FastAPI-based AI fitness service with modular architecture, comprehensive LLM integration, and a production-grade RAG (Retrieval-Augmented Generation) pipeline.

### RAG Pipeline Architecture

The RAG pipeline orchestrates document processing, embedding, and retrieval in three distinct phases:

**Phase 1: Infrastructure Setup** (`RagInfrastructureTask`)
- Initializes vector databases and storage systems
- Validates external service connectivity
- Prepares system for data ingestion

**Phase 2: Document Ingestion** (`RagIngestionTask`)
- Discovers and retrieves documents from multiple sources (files, databases, APIs, web)
- Processes documents through multi-stage pipeline: extract → normalize → enrich → validate
- Chunks documents using semantic, recursive, or sentence-based strategies
- Stores processed documents in shared repository for Phase 3

**Phase 3: Embedding Generation** (`RagEmbeddingTask`)
- Validates chunk availability from Phase 2 (Task 2 → Task 3 handover)
- Generates embeddings using sentence transformer models
- Deduplicates chunks and stores embeddings in vector database
- Tracks statistics: documents processed, chunks deduped, embeddings stored

**Workflow Orchestration** (`RagWorkflow`)
- Coordinates all phases with shared repository state
- Provides unified execution interface with progress tracking
- Generates comprehensive execution reports
- Handles error recovery and phase dependencies

**Key Components**:
- `app/pipeline/workflow.py` - Main orchestrator
- `app/pipeline/phases/` - Individual phase implementations
- `app/pipeline/config.py` - Pipeline configuration
- `app/domain/entities/chunk_load_policy.py` - Chunk loading strategies for fallback handling

### Core Service Architecture
The service follows a layered architecture with clear separation of concerns:

**FastAPI Application** (`app/main.py`)
- Configures CORS, middleware, and global exception handling
- Implements lifespan management for LLM service connections
- Auto-generates OpenAPI documentation at `/docs` and `/redoc`

**Configuration Management** (`app/core/config.py`)
- Pydantic-based settings with comprehensive validation
- Supports extensive configuration through environment variables
- Includes property methods for list conversion (CORS origins, file types)
- Validates configuration constraints (port ranges, enum values)

**LLM Service Layer** (`app/services/llm_service.py`)
- Async HTTP client for LLM API communication with connection pooling
- Comprehensive error handling with timeout management
- Token usage tracking and performance metrics
- Health monitoring capabilities

### API Structure
The API follows RESTful principles with versioned endpoints:

**Router Hierarchy**:
- Main app includes v1 router with `/api/v1` prefix
- V1 router includes workout router with `/workout` prefix
- Final endpoints: `/api/v1/workout/{endpoint}`

**Request/Response Flow**:
1. FastAPI receives and validates requests using Pydantic schemas
2. Workout endpoints delegate to LLM service with proper error handling
3. LLM service formats requests for external API and processes responses
4. Structured responses include metadata (tokens, timing, model info)

### Data Models and Validation
Pydantic models provide comprehensive validation and auto-documentation:

**Request Models**:
- `WorkoutPromptRequest`: User queries with optional context and generation parameters
- Input validation includes length limits, parameter ranges, and type checking

**Response Models**:
- `WorkoutPromptResponse`: AI responses with metadata and success indicators
- `HealthResponse`: Service status with LLM availability monitoring
- `ApiErrorResponse`: Standardized error format with detail and codes

### External Dependencies
- **LLM Service**: Requires Ollama or compatible service running on configured URL
- **Vector Store**: ChromaDB for embeddings (configurable to FAISS)
- **Database**: SQLite by default, configurable to PostgreSQL/MySQL

### Domain-Driven Design (DDD) Structure

The codebase follows DDD principles with clear domain boundaries:

**Domain Layer** (`app/domain/`)
- **Entities**: Core domain objects (Document, Chunk, Embedding, ProcessingJob)
- **Value Objects**: Immutable domain concepts (ChunkMetadata, DocumentMetadata, ChunkLoadPolicy)
- **Repositories**: Abstract interfaces for data persistence
- **Exceptions**: Domain-specific exceptions for error handling

**Application Layer** (`app/application/`)
- **Use Cases**: Business logic orchestration
- **DTOs**: Data transfer objects for internal communication
- **Query Handlers**: Query processing for read operations

**Infrastructure Layer** (`app/infrastructure/`)
- **Repository Implementations**: Concrete repository implementations
- **External Services**: ML services, LLM integration, vector stores
- **Database**: SQLAlchemy models and async session management

**API Layer** (`app/api/`)
- **Schemas**: Pydantic models for request/response validation
- **Routes**: HTTP endpoints with proper error handling
- **Dependencies**: FastAPI dependency injection

### Key Design Patterns
- **Dependency Injection**: Services injected through FastAPI's dependency system
- **Repository Pattern**: Data access abstraction through RepositoryContainer
- **Domain-Driven Design**: Clear separation between domain, application, and infrastructure
- **Value Objects**: Immutable domain concepts with validation
- **Async Operations**: Non-blocking I/O throughout the stack
- **Configuration as Code**: All settings externalized to environment variables
- **Comprehensive Error Handling**: Graceful degradation with meaningful error messages
- **Health Monitoring**: Service and dependency health checking

### Repository Container Pattern

The application uses **RepositoryContainer** for dependency injection of data repositories. This pattern provides:

**Benefits**:
- ✅ Works in both FastAPI endpoints and standalone scripts
- ✅ Configuration-driven repository selection (in-memory vs database)
- ✅ Lazy initialization for optimal performance
- ✅ Easy testing with dependency overrides
- ✅ Clean property-based API

**Location**: `app/infrastructure/repositories/container.py`

**Usage in FastAPI Endpoints**:
```python
from fastapi import Depends
from app.infrastructure.repositories.dependencies import (
    get_repository_container,
    get_document_repository,
)
from app.domain.repositories import DocumentRepository

@router.post("/documents")
async def create_document(
    # Option 1: Inject specific repository
    repo: DocumentRepository = Depends(get_document_repository),
):
    document = await repo.save(new_document)
    return document

@router.get("/documents")
async def list_documents(
    # Option 2: Inject entire container for multiple repositories
    container: RepositoryContainer = Depends(get_repository_container),
):
    documents = await container.document_repository.find_all()
    sources = await container.data_source_repository.find_all()
    return {"documents": documents, "sources": sources}
```

**Usage in Scripts**:
```python
import asyncio
from app.core.settings import Settings
from app.infrastructure.repositories.container import RepositoryContainer
from app.infrastructure.database.database import AsyncSessionLocal

async def maintenance_script():
    """Example maintenance script using RepositoryContainer."""
    settings = Settings()

    # For database operations
    async with AsyncSessionLocal() as session:
        container = RepositoryContainer(settings, session)

        # Access repositories
        documents = await container.document_repository.find_all()
        for doc in documents:
            # Process documents
            pass

        await session.commit()

async def quick_test():
    """Quick test using in-memory repositories."""
    settings = Settings()
    settings.database_url = "sqlite:///:memory:"

    # No session needed for in-memory mode
    container = RepositoryContainer(settings)

    # Use repositories directly
    await container.document_repository.save(document)

if __name__ == "__main__":
    asyncio.run(maintenance_script())
```

**Usage in Pipeline Phases**:
```python
from app.pipeline.phases.rag_ingestion_task import IngestionPhase

async def run_pipeline():
    """Run pipeline with repository container."""
    settings = Settings()

    async with AsyncSessionLocal() as session:
        # Create container once for entire pipeline
        container = RepositoryContainer(settings, session)

        # Pass to pipeline phases
        phase = IngestionPhase(
            document_repository=container.document_repository,
            data_source_repository=container.data_source_repository,
        )

        await phase.execute(spec)
        await session.commit()
```

**Testing with Container**:
```python
from fastapi.testclient import TestClient
from app.main import app
from app.infrastructure.dependencies import get_repository_container

def test_endpoint_with_mock_container():
    """Test endpoint by mocking the entire container."""
    # Create mock container
    mock_container = Mock(spec=RepositoryContainer)
    mock_repo = AsyncMock()
    mock_container.document_repository = mock_repo

    # Override dependency
    async def override_container():
        yield mock_container

    app.dependency_overrides[get_repository_container] = override_container

    # Test endpoint
    client = TestClient(app)
    response = client.post("/api/v1/documents/", json={...})

    # Verify mock was called
    assert mock_repo.save.called

    # Cleanup
    app.dependency_overrides.clear()
```

**Configuration-Based Selection**:

The container automatically selects repository implementation based on `DATABASE_URL`:

- **In-Memory Mode**: `sqlite:///:memory:` (no async driver)
  - Fast, no persistence
  - Ideal for testing and prototyping

- **Database Mode**: Async drivers detected automatically
  - `sqlite+aiosqlite:///` - SQLite with async support
  - `postgresql+asyncpg://` - PostgreSQL with async support
  - `mysql+aiomysql://` or `mysql+asyncmy://` - MySQL with async support

**See Also**:
- Container implementation: `app/infrastructure/repositories/container.py`
- FastAPI dependencies: `app/infrastructure/repositories/dependencies.py`
- Usage examples: `scripts/examples/use_container.py`
- Tests: `tests/unit/infrastructure/test_repository_container.py`

### Development Considerations
- The service expects an external LLM API (typically Ollama) for core functionality
- Configuration validation is extensive - use `test_settings.py` to verify setup
- All async operations use proper resource management (connection pooling, timeouts)
- CORS is configurable for different deployment environments
- Logging is structured with configurable levels and rotation

## API Data Modeling Style Guide

### Naming Conventions

#### **Schema/Model Naming Patterns**
The project follows a consistent naming convention for API data models:

- **Request Models**: `{Action}{Resource}Request` - e.g., `PromptRequest`, `EmbedChunksRequest`, `SearchRequest`
- **Response Models**: `{Action}{Resource}Response` - e.g., `PromptResponse`, `EmbedChunksResponse`, `SearchResponse`
- **Input Models**: `{Resource}Input` - e.g., `ChunkInput`, `TextItemInput`
- **Result Models**: `{Resource}Result` - e.g., `ChunkEmbeddingResult`, `BatchEmbedResult`
- **Filter Models**: `{Resource}Filter` - e.g., `SearchFilter`
- **Configuration Models**: `{Service}Config` - e.g., `EmbeddingConfig`, `WeaviateConfig`

#### **Field Naming Patterns**
- **Snake Case**: All fields use snake_case (e.g., `processing_time_ms`, `max_tokens`)
- **Descriptive Names**: Use clear, descriptive field names
- **Consistent Suffixes**:
  - Time fields: `_ms` (milliseconds), `_seconds` for longer durations
  - Count fields: `_count` (e.g., `total_chunks`, `failed_count`)
  - ID fields: `_id` (e.g., `chunk_id`, `session_id`)
  - Boolean flags: `is_`, `has_`, `can_` (e.g., `is_successful`, `has_metadata`)

#### **File Organization**
```
app/
├── schemas/                    # General schema models (chat, shared)
│   ├── chat.py                # Chat-related schemas
│   └── search.py              # Search-related schemas
├── api/v1/{module}/
│   └── schemas.py             # Module-specific schemas
└── application/dto/           # Internal DTOs (not exposed to API)
    └── {service}_dto.py
```

### Schema Design Principles

#### **Request Model Structure**
```python
class {Action}{Resource}Request(BaseModel):
    """Request model for {action} {resource} operation."""

    # Required fields first
    required_field: str = Field(..., description="Clear description")

    # Optional fields with sensible defaults
    optional_field: str = Field(default="default", description="Optional parameter")

    # Validation decorators
    @field_validator('field')
    @classmethod
    def validate_field(cls, v):
        """Validate field constraints."""
        return v
```

#### **Response Model Structure**
```python
class {Action}{Resource}Response(BaseModel):
    """Response model for {action} {resource} operation."""

    # Success/error status
    success: bool = Field(description="Whether operation was successful")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Core response data
    result_data: ResultType = Field(description="Primary response data")

    # Metadata
    processing_time_ms: float = Field(ge=0, description="Processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Field Design Guidelines

#### **Validation Rules**
- **String Fields**: Always include `min_length` and `max_length`
- **Numeric Fields**: Include `ge`/`le` constraints and `description`
- **List Fields**: Include `min_items`/`max_items` when applicable
- **Optional Fields**: Provide sensible defaults

#### **Documentation Standards**
- **Clear Descriptions**: Every field must have a descriptive `Field(..., description="...")`
- **Examples**: Include `example="..."` for representative values
- **Type Safety**: Use proper type hints with Optional[] for nullable fields

#### **Error Handling**
- **Structured Errors**: Include `success` flag and `error` message in responses
- **Validation Errors**: Use Pydantic validators with clear error messages
- **Consistent Format**: All error responses follow the same structure

### Module-Specific Patterns

#### **Chat/LLM Schemas** (`app/schemas/chat.py`)
- `PromptRequest`/`PromptResponse` - Core prompt interaction
- `ChatRequest`/`ChatResponse` - Streaming chat completion
- `ChatMessage` - Individual message structure
- `ApiErrorResponse` - Standard error format

#### **Embedding Schemas** (`app/api/v1/embeddings/schemas.py`)
- `EmbedChunksRequest`/`EmbedChunksResponse` - Document chunk embedding
- `EmbedQueryRequest`/`EmbedQueryResponse` - Query embedding with caching
- `BatchEmbedRequest`/`BatchEmbedResponse` - Bulk text embedding
- `SearchRequest`/`SearchResponse` - Similarity search in vector DB

#### **Search Schemas** (`app/schemas/search.py`)
- `SearchRequest`/`SearchResponse` - Core semantic search
- `SearchFilter` - Search filtering criteria
- `SimilarChunksRequest` - "More like this" functionality
- `SearchSuggestionsRequest`/`SearchSuggestionsResponse` - Autocomplete

### Best Practices

#### **Version Compatibility**
- **Backward Compatibility**: Don't remove fields in minor versions
- **Deprecated Fields**: Mark with `deprecated` parameter in Field()
- **Optional Additions**: New fields should be optional with defaults

#### **Performance Considerations**
- **Lazy Loading**: Use `Optional` for expensive-to-compute fields
- **Efficient Validation**: Keep validators simple and fast
- **Batch Operations**: Support bulk operations where appropriate

#### **Security**
- **Input Sanitization**: Validate all input parameters
- **Length Limits**: Prevent DoS attacks with max length constraints
- **Type Safety**: Use strict typing to prevent injection attacks

### Testing Requirements
- Every test should have a clear description of what it is testing
- Every test file should be placed under `tests` directory

#### **Schema Validation Tests**
```python
def test_request_validation():
    """Test request model validation."""
    # Valid request
    request = ModelSchema(valid_data)
    assert request.field == expected_value

    # Invalid request
    with pytest.raises(ValidationError):
        ModelSchema(invalid_data)
```

#### **Response Serialization Tests**
```python
def test_response_serialization():
    """Test response model serialization."""
    response = ResponseSchema(success=True, data=test_data)
    json_str = response.model_dump_json()
    assert "success" in json_str
```

### Documentation Standards

#### **Auto-Generated Docs**
- **OpenAPI Integration**: All schemas automatically appear in `/docs`
- **Example Values**: Include realistic examples in Field() definitions
- **Descriptions**: Clear, user-friendly descriptions for all fields

#### **Inline Documentation**
- **Class Docstrings**: Explain the purpose and usage of each model
- **Field Descriptions**: Explain what each field represents
- **Validator Comments**: Document complex validation logic

This style guide ensures consistency across all API endpoints while maintaining clarity and maintainability of the data models.

## LLM Integration Notes

The service is designed around external LLM APIs with fitness-specific system prompts. The LLM service handles:
- Request formatting for Ollama-compatible APIs
- Response parsing with token tracking
- Timeout and error management
- Health monitoring of the LLM service

When modifying LLM integration, ensure compatibility with the expected request/response format and maintain the existing error handling patterns.

## Critical Development Patterns

### Async/Await Usage
- **All database operations are async** - Use `await` with repository methods
- **All external service calls are async** - LLM service, embedding service, vector store operations
- **Use AsyncMock in tests** - Not Mock, for async methods
- **AsyncSession context managers** - Properly manage SQLAlchemy async sessions
- **Never use sync operations in async functions** - This will block event loop

### RAG Pipeline Testing
- **Mock external dependencies** - Embedding services, vector stores, repositories
- **Use real dataclasses** - EmbeddingResult, RagEmbeddingTaskReport are real models
- **Test both success and failure paths** - Especially Task 2 → Task 3 handover
- **Validate chunk availability** - Phase 3 assumes Phase 2 completed successfully
- **Track deduplication stats** - Validate duplicate removal calculations

### Repository Access Patterns
- **Use RepositoryContainer for dependency injection** - Provides consistent repository access
- **Access repositories via dependency injection** - `Depends(get_repository_container)`
- **In scripts, create RepositoryContainer manually** - With Settings and AsyncSession
- **In tests, mock repositories with AsyncMock** - Realistic return values

### Error Handling
- **Domain exceptions for business logic failures** - ChunkingError, EmbeddingGenerationError, etc.
- **Custom pipeline exceptions** - EmbeddingPipelineError, IngestionPipelineError
- **Catch and log at boundaries** - FastAPI routes, pipeline orchestrator
- **Return structured error responses** - Always include error details for debugging

### Testing Strategy
- **Use pytest with async support** - `@pytest.mark.asyncio` decorator
- **Conftest for shared fixtures** - Global fixtures in `conftest.py`, module-specific in local `conftest.py`
- **Organize by layer** - Tests mirror application structure (unit/pipeline, unit/domain, integration/api)
- **Test configuration in isolation** - Settings validation before service startup
- **Mock at boundaries** - Mock external services and repositories, not internal logic

### Performance Considerations
- **Connection pooling enabled** - AsyncPG for PostgreSQL, aiosqlite for SQLite
- **Batch operations preferred** - Processing documents in batches when possible
- **Lazy loading in repositories** - Repository patterns should fetch data on demand
- **Vector database indexing** - ChromaDB/Weaviate handle embedding search optimization
- **Monitor embedding model memory** - Sentence transformers can be memory-intensive