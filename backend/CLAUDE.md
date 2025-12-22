# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## NOTICE
This project is in developement and is going to actively update the code base without support backward compatibility.

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
- **Vector Store**: Weaviate configuration for semantic search and embeddings
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
- **Vector Store**: Weaviate for vector embeddings and semantic search
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

### Layered Container Architecture

이 프로젝트는 Clean Architecture 원칙에 따라 계층화된 DI container 구조를 사용합니다:

```
app/di/containers/
├── infra_container.py     # Infrastructure Layer
├── domain_container.py    # Domain Layer
└── container.py          # Application Layer (AppContainer)
```

#### Container Layer 설명

##### Infrastructure Container (`InfraContainer`)
- **담당**: 데이터베이스, 외부 서비스, 저장소
- **포함**: Weaviate, 데이터베이스 세션, 리포지토리, 임베딩 서비스
- **특징**: 외부 의존성과 가장 가까움

##### Domain Container (`DomainContainer`)
- **담당**: 비즈니스 로직, 도메인 서비스
- **포함**: ContextService, RetrievalService, SessionService 등
- **특징**: Infrastructure Container에 의존

##### Application Container (`AppContainer`)
- **담당**: 애플리케이션 오케스트레이션, 유스케이스
- **포함**: 모든 계층을 통합한 최상위 컨테이너
- **특징**: Domain Container와 Infrastructure Container에 의존

#### FastAPI Dependency 구조

```python
# app/api/v1/fitvise/deps.py

# ========================================
# INFRASTRUCTURE
# ========================================
@lru_cache()
def get_app_container() -> AppContainer:
    return AppContainer()

# ========================================
# DOMAIN
# ========================================
@lru_cache()
def get_session_service() -> SessionService:
    return SessionService()

# ========================================
# APPLICATION
# ========================================
async def get_llama_index_retriever(
    container: Annotated[AppContainer, Depends(get_app_container)]
) -> BaseRetriever:
    return container.llama_index_retriever
```

#### 사용법

```python
from fastapi import Depends
from app.api.v1.fitvise.deps import get_llama_index_retriever

@app.get("/search")
async def search(
    retriever: BaseRetriever = Depends(get_llama_index_retriever)
):
    # retriever는 자동으로 초기화되고 캐시됨
    results = await retriever.ainvoke("query")
    return results
```

### Key Design Patterns
- **Dependency Injection**: Services injected through FastAPI's dependency system
- **Repository Pattern**: Data access abstraction through RepositoryContainer
- **Domain-Driven Design**: Clear separation between domain, application, and infrastructure
- **Value Objects**: Immutable domain concepts with validation
- **Async Operations**: Non-blocking I/O throughout the stack
- **Configuration as Code**: All settings externalized to environment variables
- **Comprehensive Error Handling**: Graceful degradation with meaningful error messages
- **Health Monitoring**: Service and dependency health checking

### InfraContainer Pattern

The application uses **InfraContainer** for dependency injection of infrastructure services including data repositories. This pattern provides:

**Benefits**:
- ✅ Works in both FastAPI endpoints and standalone scripts
- ✅ Configuration-driven repository selection (in-memory vs database)
- ✅ Lazy initialization for optimal performance
- ✅ Easy testing with dependency overrides
- ✅ Clean property-based API

**Location**: `app/di/containers/infra_container.py`

**Usage in FastAPI Endpoints**:
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

**Usage in Scripts**:
```python
import asyncio
from app.di.containers.infra_container import InfraContainer
from app.infrastructure.database.database import AsyncSessionLocal

async def maintenance_script():
    """Example maintenance script using InfraContainer."""
    # Create InfraContainer
    container = InfraContainer()

    # For database operations
    async with AsyncSessionLocal() as session:
        container.db_session.override(session)

        # Access repositories
        documents = await container.document_repository().find_all()
        for doc in documents:
            # Process documents
            pass

        await session.commit()

async def quick_test():
    """Quick test using in-memory repositories."""
    # Create InfraContainer with default configuration
    container = InfraContainer()
    container.configs.database_type.override('default')

    # Use repositories directly
    await container.document_repository().save(document)

if __name__ == "__main__":
    asyncio.run(maintenance_script())
```

**Usage in Pipeline Phases**:
```python
from app.pipeline.phases.rag_ingestion_task import IngestionPhase
from app.di.containers.infra_container import InfraContainer

async def run_pipeline():
    """Run pipeline with InfraContainer."""
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
```

**Testing with Container**:
```python
from fastapi.testclient import TestClient
from app.main import app
from app.di.containers.infra_container import InfraContainer

def test_endpoint_with_mock_container():
    """Test endpoint by mocking the InfraContainer."""
    # Create mock container
    mock_container = InfraContainer()
    mock_repo = AsyncMock()
    # Override the repository provider
    mock_container.document_repository.override(mock_repo)

    # Override dependency
    async def override_container():
        return mock_container

    app.dependency_overrides[get_infra_container] = override_container

    # Test endpoint
    client = TestClient(app)
    response = client.post("/api/v1/documents/", json={...})

    # Verify mock was called
    assert mock_repo.save.called

    # Cleanup
    app.dependency_overrides.clear()
```

**Configuration-Based Selection**:

The container automatically selects repository implementation based on `database_type` configuration:

- **In-Memory Mode**: `default` (InMemoryDocumentRepository)
  - Fast, no persistence
  - Ideal for testing and prototyping

- **Database Mode**: Async drivers configured automatically
  - `aiosqlite` - SQLite with async support (SQLAlchemyDocumentRepository)
  - `asyncpg` - PostgreSQL with async support (SQLAlchemyDocumentRepository)
  - `aiomysql` or `asyncmy` - MySQL with async support (SQLAlchemyDocumentRepository)

**See Also**:
- Container implementation: `app/di/containers/infra_container.py`
- FastAPI dependencies: `app/di/bootstrap.py`
- Usage examples: `scripts/examples/use_container.py`
- Tests: Updated factory selection tests in repository test files

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
- **Use InfraContainer for dependency injection** - Provides consistent repository access
- **Access repositories via dependency injection** - `Depends(get_infra_container)`
- **In scripts, create InfraContainer manually** - With Settings and AsyncSession
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
- **Vector database indexing** - Weaviate handles embedding search optimization with advanced indexing strategies
- **Monitor embedding model memory** - Sentence transformers can be memory-intensive

### Python usage
- If you know the attribute name, access it directly. If you need dynamic behavior, consider a dictionary or a strategy pattern instead of relying on attribute lookup(getattr) for control flow. use `getattr` only when you need to access attributes dynamically and when you need default or fallback value.
- Avoid import statement using try-catch statement. It can cause circular import and unexpected behavior. declare import statement at the top of the file.
- Always keep the code clean and tidy by using type-safe and definite coding styles.
