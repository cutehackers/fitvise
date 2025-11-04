# CLAUDE.md

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
All tests are located in the `/backend/tests/` directory following this structure:
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── table_serialization/ # Table serialization module tests
│   │   ├── fixtures/        # Test data fixtures
│   │   │   └── sample_tables.py
│   │   ├── conftest.py      # Local test configuration
│   │   └── test_serializers.py
│   └── ...
├── integration/             # Integration tests
├── e2e/                     # End-to-end tests
├── fixtures/                # Shared test fixtures
├── utils/                   # Test utilities
└── conftest.py             # Global pytest configuration
```

**Running Tests**:
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/unit/table_serialization/

# Run with verbose output
pytest tests/unit/table_serialization/test_serializers.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Test configuration loading (legacy)
python test_settings.py

# Test API endpoints (legacy)
python api_example.py

# Test LLM service directly (legacy)
python example_usage.py
```

**Test Organization Rules**:
- ✅ **Unit tests**: `/tests/unit/` - Test individual components in isolation
- ✅ **Integration tests**: `/tests/integration/` - Test component interactions
- ✅ **E2E tests**: `/tests/e2e/` - Test complete user workflows
- ❌ **Never** place tests inside `app/` directory - they belong in `/tests/`

### Environment Configuration
The service requires a comprehensive `.env` file. Copy the existing `.env` and modify as needed. Key configuration areas:
- **LLM Integration**: `LLM_BASE_URL`, `LLM_MODEL` (requires Ollama or compatible service)
- **API Settings**: `API_HOST`, `API_PORT`, CORS configuration
- **Database**: `DATABASE_URL` (SQLite by default)
- **Vector Store**: ChromaDB configuration for embeddings
- **Security**: JWT configuration, file upload limits

## Architecture Overview

This is a FastAPI-based AI fitness service with modular architecture and comprehensive LLM integration.

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

### Key Design Patterns
- **Dependency Injection**: Services injected through FastAPI's dependency system
- **Async Operations**: Non-blocking I/O throughout the stack
- **Configuration as Code**: All settings externalized to environment variables
- **Comprehensive Error Handling**: Graceful degradation with meaningful error messages
- **Health Monitoring**: Service and dependency health checking

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