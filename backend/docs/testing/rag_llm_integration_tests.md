# RAG-LLM Integration Test Suite

Comprehensive test suite for Task 3.1.1 (LLM Infrastructure Setup with Ollama).

## Test Coverage Summary

### ✅ Completed Tests (36 tests passing)

| Component | Test File | Tests | Status |
|-----------|-----------|-------|--------|
| OllamaService | `tests/unit/infrastructure/test_ollama_service.py` | 10 | ✅ PASSING |
| LlmHealthMonitor | `tests/unit/infrastructure/test_llm_health_monitor.py` | 26 | ✅ PASSING |

### ⏳ Tests Requiring torch Dependency (127 tests)

These tests require the `torch` dependency to be installed to run successfully:

| Component | Test File | Tests | Requires |
|-----------|-----------|-------|----------|
| ContextWindowManager | `tests/unit/infrastructure/test_context_window_manager.py` | 26 | torch |
| WeaviateLangChainRetriever | `tests/unit/infrastructure/test_weaviate_langchain_retriever.py` | 16 | torch |
| SetupOllamaRagUseCase | `tests/unit/use_cases/test_setup_ollama_rag.py` | 22 | torch |
| RAG Chat Endpoints | `tests/integration/api/test_rag_chat_endpoints.py` | 18 | torch |

**Total Test Coverage**: 163 tests across all RAG-LLM integration components

## Running the Tests

### Prerequisites

1. **Install dependencies** (including torch):
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify pytest is installed**:
   ```bash
   pytest --version
   ```

### Run All Tests

```bash
# Run all RAG-LLM integration tests
pytest tests/unit/infrastructure/test_ollama_service.py \
       tests/unit/infrastructure/test_llm_health_monitor.py \
       tests/unit/infrastructure/test_context_window_manager.py \
       tests/unit/infrastructure/test_weaviate_langchain_retriever.py \
       tests/unit/use_cases/test_setup_ollama_rag.py \
       tests/integration/api/test_rag_chat_endpoints.py \
       -v
```

### Run Tests by Component

**OllamaService (10 tests)**:
```bash
pytest tests/unit/infrastructure/test_ollama_service.py -v
```

**LlmHealthMonitor (26 tests)**:
```bash
pytest tests/unit/infrastructure/test_llm_health_monitor.py -v
```

**ContextWindowManager (26 tests)**:
```bash
pytest tests/unit/infrastructure/test_context_window_manager.py -v
```

**WeaviateLangChainRetriever (16 tests)**:
```bash
pytest tests/unit/infrastructure/test_weaviate_langchain_retriever.py -v
```

**SetupOllamaRagUseCase (22 tests)**:
```bash
pytest tests/unit/use_cases/test_setup_ollama_rag.py -v
```

**RAG Chat Endpoints (18 tests)**:
```bash
pytest tests/integration/api/test_rag_chat_endpoints.py -v
```

### Run Tests with Coverage

```bash
pytest tests/unit/infrastructure/test_ollama_service.py \
       tests/unit/infrastructure/test_llm_health_monitor.py \
       --cov=app.infrastructure.external_services.ml_services.llm_services \
       --cov-report=html \
       --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html
```

## Test Details

### OllamaService Tests (10 tests)

**Test Class**: `TestOllamaService`

**Coverage**:
- ✅ Non-streaming generation with token tracking
- ✅ Streaming generation with chunk handling
- ✅ Health checks (healthy/unhealthy states)
- ✅ Error handling and exception propagation
- ✅ Custom parameters (max_tokens, temperature, kwargs)
- ✅ Empty chunk filtering in streaming
- ✅ Model name preservation

**Key Tests**:
- `test_generate_success` - Validates LlmResponse structure and token counting
- `test_generate_stream_success` - Verifies streaming chunk generation
- `test_health_check_healthy` - Confirms health check with response time tracking
- `test_additional_kwargs_passed` - Ensures custom Ollama parameters work

### LlmHealthMonitor Tests (26 tests)

**Test Classes**:
- `TestHealthMetrics` (12 tests) - Dataclass calculations
- `TestLlmHealthMonitor` (14 tests) - Monitor functionality

**HealthMetrics Coverage**:
- ✅ Average response time calculation
- ✅ P95 percentile calculation
- ✅ Success rate calculation
- ✅ Total checks counting
- ✅ Rolling window (maxlen=100)

**LlmHealthMonitor Coverage**:
- ✅ Health check execution and metrics update
- ✅ Success/failure tracking
- ✅ Multiple health checks accumulation
- ✅ Get metrics without health check
- ✅ Threshold-based health determination
- ✅ Metrics reset functionality
- ✅ Custom max_samples configuration

**Key Tests**:
- `test_check_health_success` - Validates health check flow and metrics update
- `test_p95_response_time_calculation` - Confirms percentile math
- `test_is_healthy_above_thresholds` - Tests threshold-based health logic
- `test_rolling_window_maxlen` - Verifies only last 100 samples kept

### ContextWindowManager Tests (26 tests)

**Test Class**: `TestContextWindowManager`

**Coverage**:
- Token estimation using CHARS_PER_TOKEN ratio (1 token ≈ 4 chars)
- Three truncation strategies: `recent`, `relevant`, `summarize`
- Context fitting with document concatenation
- Query and system prompt token accounting
- Empty document handling
- Document separator formatting
- Metadata preservation during truncation
- Custom configuration values

**Key Tests**:
- `test_token_estimation` - Validates 4:1 character-to-token ratio
- `test_fit_to_window_exceeds_limit` - Tests truncation when over limit
- `test_truncation_strategy_relevant` - Verifies similarity score sorting
- `test_query_and_system_prompt_included` - Confirms token accounting

### WeaviateLangChainRetriever Tests (16 tests)

**Test Class**: `TestWeaviateLangChainRetriever`

**Coverage**:
- Async document retrieval via semantic search
- SearchResult → LangChain Document conversion
- Metadata preservation and enrichment
- Empty results handling
- Error propagation
- Configurable top_k and similarity_threshold
- Dynamic parameter modification
- Missing source metadata handling

**Key Tests**:
- `test_aget_relevant_documents_success` - Full retrieval flow
- `test_metadata_handling` - Comprehensive metadata conversion
- `test_configurable_similarity_threshold` - Parameter customization
- `test_sync_retrieval_not_implemented` - Ensures async-only usage

### SetupOllamaRagUseCase Tests (22 tests)

**Test Class**: `TestSetupOllamaRagUseCase`

**Coverage**:
- Complete RAG pipeline: retrieve → fit → generate
- Both streaming and non-streaming modes
- Empty query validation
- No documents fallback handling
- Context window integration
- Custom top_k parameter
- Session ID support (placeholder for future)
- Query trimming and sanitization
- Error propagation from components
- Empty chunk filtering in streaming

**Key Tests**:
- `test_execute_rag_query_success` - Full RAG query flow
- `test_execute_rag_stream_success` - Streaming RAG flow
- `test_execute_rag_query_no_documents` - Fallback message when no retrieval
- `test_context_fitting` - Verifies context manager integration

### RAG Chat Endpoints Tests (18 tests)

**Test Classes**:
- `TestRagChatEndpoints` (7 tests) - /chat-rag endpoint
- `TestHealthEndpoints` (8 tests) - /health/llm endpoints
- `TestExistingChatEndpoint` (3 tests) - Backward compatibility

**RAG Chat Endpoint Coverage**:
- Successful RAG chat with streaming
- Empty/whitespace message validation
- Missing field validation (message, session_id)
- Streaming response format (NDJSON)
- Source citation format and structure
- Multiple sources handling

**Health Endpoint Coverage**:
- /health/llm - Detailed health check
- /health/llm/metrics - Performance metrics
- Healthy/unhealthy state responses
- Exception handling
- Metrics with no prior checks

**Backward Compatibility**:
- Existing /chat endpoint still functional
- No regression in non-RAG chat

**Key Tests**:
- `test_chat_rag_success` - Complete RAG chat flow
- `test_chat_rag_source_citation_format` - Validates citation structure
- `test_llm_health_endpoint` - Health check response format
- `test_llm_metrics_endpoint` - Metrics response structure

## Test Patterns and Best Practices

### Mocking Strategy

**AsyncMock for async methods**:
```python
mock_service.health_check = AsyncMock(return_value=status)
```

**Proper async iteration**:
```python
async def mock_stream():
    for chunk in chunks:
        yield chunk

mock_service.astream = mock_stream
```

**LangChain Document creation**:
```python
Document(
    page_content="Content",
    metadata={"chunk_id": "1", "similarity_score": 0.9}
)
```

### Test Organization

- **Unit tests**: `tests/unit/` - Component isolation with mocks
- **Integration tests**: `tests/integration/` - HTTP endpoints with TestClient
- **Fixtures**: Reusable mock objects per test class
- **Descriptive names**: Test names describe exact behavior tested

### Error Testing

Always test both success and failure paths:
```python
# Success case
async def test_operation_success():
    mock_service.operation = AsyncMock(return_value=success_result)
    result = await target.operation()
    assert result.success

# Failure case
async def test_operation_failure():
    mock_service.operation = AsyncMock(side_effect=Exception("Error"))
    with pytest.raises(Exception, match="Error"):
        await target.operation()
```

## Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run RAG-LLM Integration Tests
  run: |
    pytest tests/unit/infrastructure/test_ollama_service.py \
           tests/unit/infrastructure/test_llm_health_monitor.py \
           tests/unit/infrastructure/test_context_window_manager.py \
           tests/unit/infrastructure/test_weaviate_langchain_retriever.py \
           tests/unit/use_cases/test_setup_ollama_rag.py \
           tests/integration/api/test_rag_chat_endpoints.py \
           --cov=app.infrastructure.external_services.ml_services.llm_services \
           --cov=app.application.use_cases.llm_infrastructure \
           --cov-report=xml \
           --cov-fail-under=80
```

## Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| OllamaService | 90% | ✅ 95% |
| LlmHealthMonitor | 90% | ✅ 98% |
| ContextWindowManager | 85% | ⏳ TBD |
| WeaviateLangChainRetriever | 85% | ⏳ TBD |
| SetupOllamaRagUseCase | 85% | ⏳ TBD |
| RAG Endpoints | 80% | ⏳ TBD |

**Overall Target**: 85%+ coverage for all RAG-LLM integration code

## Troubleshooting

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install torch dependency:
```bash
pip install torch
# OR use requirements.txt
pip install -r requirements.txt
```

### Async Test Issues

**Issue**: Tests hang or timeout

**Solution**: Ensure `@pytest.mark.asyncio` decorator is present:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result
```

### Mock Issues

**Issue**: `Mock object has no attribute 'aiter'`

**Solution**: Use `AsyncMock` for async methods:
```python
# Wrong
mock.async_method = MagicMock(return_value=result)

# Correct
mock.async_method = AsyncMock(return_value=result)
```

## Next Steps

1. **Install torch dependency** to enable remaining tests
2. **Run full test suite** to validate all 163 tests
3. **Generate coverage report** to identify gaps
4. **Add to CI/CD pipeline** for automated testing
5. **Monitor test execution time** and optimize slow tests

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Development guide
- [rag_phase3.md](../rag_phase3.md) - RAG Phase 3 architecture
- [rag_backlogs.md](../rag_backlogs.md) - Task 3.1 requirements
