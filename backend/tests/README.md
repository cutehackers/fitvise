# Fitvise Backend Test Suite

Professional testing infrastructure for the Fitvise backend API with comprehensive coverage and best practices.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Global pytest configuration and fixtures
├── pytest.ini                 # Pytest configuration file
├── README.md                   # This documentation
├── fixtures/                   # Test data and samples
│   ├── __init__.py
│   └── sample_data.py         # Sample requests, responses, scenarios
├── utils/                      # Testing utilities
│   ├── __init__.py
│   └── test_helpers.py        # Helper functions, generators, assertions
├── unit/                       # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── api/                   # API layer unit tests
│   ├── core/                  # Core functionality tests
│   │   └── test_config.py     # Configuration testing
│   ├── services/              # Service layer tests
│   │   └── test_llm_service.py # LLM service unit tests
│   ├── models/                # Data model tests
│   └── schemas/               # Schema validation tests
├── integration/               # Integration tests (component interactions)
│   ├── __init__.py
│   ├── api/                   # API endpoint integration tests
│   │   └── test_workout_endpoints.py # Workout API tests
│   ├── database/              # Database integration tests
│   └── external_services/     # External service integration tests
└── e2e/                      # End-to-end tests (complete workflows)
    ├── __init__.py
    └── test_workout_workflows.py # Complete user journey tests
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Ensure backend dependencies are installed
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m e2e            # End-to-end tests only

# Run tests excluding external dependencies
pytest -m "not external"

# Run fast tests only (exclude slow tests)
pytest -m "not slow"

# Run specific test file
pytest tests/unit/core/test_config.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_workout"
```

### Test Environment Setup

The test suite automatically configures a test environment with:

- Test database (SQLite)
- Mock LLM service (no external calls)
- Test configuration overrides
- Isolated environment variables

## 📊 Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation
- **Speed**: Fast (< 100ms per test)
- **Dependencies**: Mocked
- **Coverage**: Individual functions, classes, modules

**Examples**:
- Configuration validation
- Service method logic  
- Data model behavior
- Schema validation

### Integration Tests (`tests/integration/`) 

**Purpose**: Test component interactions
- **Speed**: Medium (100ms - 1s per test)
- **Dependencies**: Some real, some mocked
- **Coverage**: API endpoints, database operations, service integration

**Examples**:
- API request/response cycles
- Database CRUD operations
- Service layer integration
- External API mocking

### End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows
- **Speed**: Slow (1s+ per test)
- **Dependencies**: Real or realistic
- **Coverage**: Complete user journeys

**Examples**:
- Full workout generation flow
- Error handling workflows
- Performance under load
- Multi-user scenarios

## 🔧 Testing Utilities

### Test Helpers (`tests/utils/test_helpers.py`)

```python
from tests.utils.test_helpers import generate_data, create_mocks, assert_helpers

# Generate test data
workout_data = generate_data.workout_prompt_data()
response_data = generate_data.workout_response_data()

# Create mocks
mock_service = create_mocks.mock_llm_service()
mock_response = create_mocks.mock_http_response(status_code=200)

# Assertions
assert_helpers.assert_valid_workout_response(response_data)
assert_helpers.assert_valid_health_response(health_data)
```

### Sample Data (`tests/fixtures/sample_data.py`)

Pre-defined test data for consistent testing:

```python
from tests.fixtures.sample_data import (
    SAMPLE_WORKOUT_PROMPTS,
    SAMPLE_WORKOUT_RESPONSES,
    SAMPLE_API_SCENARIOS,
    E2E_SCENARIOS
)
```

### Global Fixtures (`tests/conftest.py`)

Available fixtures:
- `test_client`: FastAPI test client
- `async_test_client`: Async test client  
- `test_settings`: Test configuration
- `mock_llm_service`: Mocked LLM service
- `sample_workout_*`: Sample data fixtures

## 📋 Test Markers

Tests are automatically marked based on location and content:

```python
# Automatic markers
pytest -m unit          # All tests in tests/unit/
pytest -m integration   # All tests in tests/integration/
pytest -m e2e           # All tests in tests/e2e/

# Manual markers
pytest -m slow          # Tests marked as slow
pytest -m external      # Tests requiring external services
pytest -m skip_ci       # Tests to skip in CI

# Combine markers
pytest -m "unit and not external"
pytest -m "integration or e2e"
```

## 🎯 Coverage Requirements

- **Overall Coverage**: ≥80%
- **Unit Tests**: ≥90% of core business logic
- **Critical Paths**: 100% coverage
- **New Code**: 100% coverage required

```bash
# Generate coverage report
pytest --cov --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🔄 Continuous Integration

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### CI Pipeline

The test suite is designed for CI environments:

```yaml
# Example GitHub Actions
- name: Run Tests
  run: |
    pytest -m "not external" --cov --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
```

## 🐛 Debugging Tests

### Useful pytest Options

```bash
# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show local variables in traceback
pytest -l

# Show output (print statements)
pytest -s

# Run last failed tests
pytest --lf

# Run tests that failed, then all
pytest --ff
```

### Test Debugging

```python
# Add debug information to tests
def test_example():
    result = some_function()
    print(f"Debug: result = {result}")  # Use pytest -s to see output
    assert result == expected
```

## 📈 Performance Testing

### Load Testing

```bash
# Run performance tests
pytest -m slow tests/e2e/test_workout_workflows.py::TestWorkoutWorkflows::test_performance_under_load
```

### Benchmarking

```python
# Example benchmark test
def test_performance_benchmark(benchmark):
    result = benchmark(expensive_function, arg1, arg2)
    assert result == expected
```

## 🛠 Maintenance

### Adding New Tests

1. **Choose appropriate category** (unit/integration/e2e)
2. **Follow naming conventions** (`test_*.py`, `test_*()`)
3. **Use existing fixtures and utilities**
4. **Add appropriate markers**
5. **Update documentation**

### Test Data Management

- Keep sample data in `fixtures/sample_data.py`
- Use data generators for dynamic test data
- Clean up test data automatically
- Avoid hardcoded values in tests

### Mock Management

- Use `tests/utils/test_helpers.py` for common mocks
- Mock at the appropriate level (service vs HTTP)
- Verify mock interactions where important
- Keep mocks simple and focused

## 📚 Best Practices

### Test Organization

- **One concept per test**: Each test should verify one specific behavior
- **Descriptive names**: Test names should describe what they verify
- **AAA Pattern**: Arrange, Act, Assert structure
- **Independent tests**: Tests should not depend on each other

### Test Data

- **Realistic data**: Use realistic test data that matches production
- **Edge cases**: Test boundary conditions and edge cases
- **Error conditions**: Test error scenarios and edge cases
- **Data isolation**: Each test should use its own data

### Assertions

- **Specific assertions**: Use specific assertions rather than generic ones
- **Error messages**: Provide helpful error messages for failures
- **Multiple assertions**: Group related assertions logically
- **Custom assertions**: Use helper functions for complex assertions

### Performance

- **Fast feedback**: Keep unit tests fast (< 100ms)
- **Parallel execution**: Design tests to run in parallel
- **Resource cleanup**: Clean up resources after tests
- **CI optimization**: Optimize for CI environment constraints

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH and module structure
2. **Fixture not found**: Verify fixture is in `conftest.py` or properly imported
3. **Mock not working**: Check mock path and ensure correct patching
4. **Async tests failing**: Use `pytest-asyncio` and `@pytest.mark.asyncio`
5. **Database errors**: Ensure test database is properly isolated

### Getting Help

- Check test logs with `pytest -v -s`
- Use `pytest --pdb` for interactive debugging
- Review `conftest.py` for available fixtures
- Check `pytest.ini` for configuration
- Consult pytest documentation for advanced features

## 📝 Contributing

When contributing tests:

1. Follow the established structure and patterns
2. Add appropriate documentation
3. Ensure tests pass locally
4. Update this README if needed
5. Consider test maintenance and readability