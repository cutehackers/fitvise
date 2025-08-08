# Fitvise Backend

> **AI-Powered Fitness API** - Intelligent workout planning and health guidance through advanced language models

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.11.7-red.svg)](https://pydantic-docs.helpmanual.io/)

## 🌟 Overview

Fitvise Backend is a sophisticated REST API that leverages Large Language Models (LLMs) to provide personalized fitness coaching, workout planning, and health guidance. Built with modern Python technologies, it offers a scalable, type-safe, and high-performance solution for fitness applications.

### Key Features

🤖 **AI-Powered Fitness Coaching** - Generate personalized workout plans, nutrition advice, and exercise recommendations
🏗️ **Production-Ready Architecture** - Modular FastAPI structure with async operations and proper error handling  
📊 **Health Monitoring** - Comprehensive health checks and service availability monitoring
🔒 **Type Safety** - Full Pydantic validation for requests and responses
⚡ **High Performance** - Async HTTP client with connection pooling and timeout management
📚 **Auto-Generated Documentation** - Interactive Swagger UI and ReDoc documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LLM service (e.g., [Ollama](https://ollama.ai/))

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

1. **Configure environment variables**:
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
├── api/v1/              # API version 1
│   ├── endpoints/       # Endpoint implementations
│   │   └── workout.py   # Workout-related endpoints
│   └── router.py        # API router aggregation
├── core/
│   └── config.py        # Configuration management
├── schemas/
│   └── workout.py       # Pydantic models
├── services/
│   └── llm_service.py   # LLM integration service
└── main.py              # FastAPI application
```

### Core Components

**LLM Service** (`app/services/llm_service.py`)
- Async HTTP client for LLM API communication
- Request/response processing with error handling
- Token usage tracking and performance metrics
- Health monitoring and timeout management

**Configuration** (`app/core/config.py`)
- Pydantic-based settings management
- Environment variable loading with validation
- Type-safe configuration access
- CORS and security settings

**API Endpoints** (`app/api/v1/endpoints/workout.py`)
- RESTful endpoint implementations
- Request validation and response formatting
- Error handling with appropriate HTTP status codes
- Health monitoring and service status reporting

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_NAME` | Application name | - | ✅ |
| `APP_VERSION` | API version | - | ✅ |
| `LLM_BASE_URL` | LLM service URL | - | ✅ |
| `LLM_MODEL` | Model identifier | - | ✅ |
| `LLM_TEMPERATURE` | Response creativity (0.0-2.0) | `0.7` | ✅ |
| `LLM_MAX_TOKENS` | Max response length | `1000` | ✅ |
| `LLM_TIMEOUT` | Request timeout (seconds) | `30` | ✅ |
| `ENVIRONMENT` | Deployment environment | `local` | ✅ |
| `DEBUG` | Debug mode | `false` | ✅ |
| `API_HOST` | Server host | `0.0.0.0` | ✅ |
| `API_PORT` | Server port | `8000` | ✅ |

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

### Run Examples

Test the API with the provided example client:

```bash
python api_example.py
```

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

## 🤝 Contributing

### Development Guidelines

- **Type Hints**: All functions must have type annotations
- **Pydantic Models**: Use for all request/response structures
- **Async/Await**: Prefer async operations for I/O
- **Error Handling**: Comprehensive exception handling
- **Documentation**: Update API.md for endpoint changes

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for all public functions
- Maintain consistent error handling patterns

## 📚 Additional Resources

- **[API Documentation](API.md)** - Detailed endpoint documentation
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - Framework documentation
- **[Pydantic Documentation](https://pydantic-docs.helpmanual.io/)** - Validation library
- **Interactive API Docs** - Available at `/docs` when running

## 📄 License

This project is part of the Fitvise fitness application suite.

---

**Built with ❤️ for intelligent fitness coaching**