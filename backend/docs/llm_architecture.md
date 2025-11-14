# LLM Service Architecture

This document describes the new LLM service architecture that provides a clean, extensible, and maintainable foundation for AI-powered features.

## Overview

The new architecture follows clean architecture principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                      │
├─────────────────────────────────────────────────────────────┤
│                  Application Layer                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Chat Orchestrator│    │      RAG Use Cases             │ │
│  │                 │    │                                 │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Domain Layer                            │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ LLM Provider     │    │     Chat Orchestrator          │ │
│  │ Interface       │    │     Interface                   │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                        │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  OllamaProvider │    │  LangChainOrchestrator         │ │
│  │                 │    │                                 │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Domain Layer (`app/domain/llm/`)

#### Interfaces (`app/domain/llm/interfaces/`)

- **`LLMProvider`**: Abstract interface for LLM providers
  - `generate()` - Non-streaming generation
  - `generate_stream()` - Streaming generation
  - `health_check()` - Health monitoring
  - `get_model_info()` - Model information

- **`ChatOrchestrator`**: Abstract interface for chat orchestration
  - `chat()` - Process chat messages with streaming
  - `get_session_history()` - Retrieve conversation history
  - `clear_session()` - Clear chat sessions
  - `health_check()` - Health monitoring

#### Entities (`app/domain/llm/entities/`)

- **`Message`**: Represents a chat message with role and content
- **`ModelInfo`**: Information about LLM models and capabilities
- **`ChatSession`**: Chat session with message history and metadata

#### Exceptions (`app/domain/llm/exceptions.py`)

- **`LLMProviderError`**: LLM provider-specific errors
- **`ChatOrchestratorError`**: Chat orchestration errors
- **`SessionNotFoundError`**: Session management errors
- **`MessageValidationError`**: Message validation errors

### 2. Infrastructure Layer (`app/infrastructure/llm/`)

#### Providers (`app/infrastructure/llm/providers/`)

- **`OllamaProvider`**: Concrete implementation for Ollama LLM
  - Direct ChatOllama integration
  - Streaming and non-streaming support
  - Health monitoring and error handling
  - Token limit management

#### Services (`app/infrastructure/llm/services/`)

- **`LangChainOrchestrator`**: Chat orchestration with LangChain
  - Session management using `InMemoryChatMessageHistory`
  - Configurable `turns_window` for conversation history limits
  - Message history with context window management
  - Streaming response handling
  - LangChain integration for advanced features

#### Dependencies (`app/infrastructure/llm/dependencies.py`)

- Clean dependency injection using FastAPI's DI system
- Type aliases for clean dependency declarations
- LRU caching for performance
- Settings-based configuration

### 3. API Layer (`app/api/v1/fitvise/chat.py`)

Updated endpoints using the new architecture:

- **`/health`**: Combined health check for provider and orchestrator
- **`/chat`**: Streaming chat with session management
- **`/chat-rag`**: RAG-enhanced chat with source citations
- **`/health/llm`**: Detailed LLM health metrics

## Benefits of the New Architecture

### 1. **Clean Separation of Concerns**
- Domain interfaces separate business logic from implementation
- Infrastructure layer handles external service integration
- API layer focuses on HTTP concerns only

### 2. **Extensibility**
- Easy to add new LLM providers (OpenAI, Anthropic, etc.)
- Pluggable chat orchestrators
- Configurable session management strategies

### 3. **Testability**
- Interface-based design enables easy mocking
- Dependency injection allows for test doubles
- Clean boundaries between layers

### 4. **Maintainability**
- Single responsibility principle applied throughout
- Clear dependency flow (infrastructure → domain → application)
- Comprehensive error handling and logging

### 5. **Performance**
- Efficient session management with cleanup
- Streaming support for real-time responses
- LRU caching for frequently used dependencies

## Migration from Old Architecture

### What Was Removed
- `app/application/llm_service.py` - Monolithic LLM service
- `app/infrastructure/external_services/ml_services/llm_services/ollama_service.py` - Wrapper pattern

### What Was Added
- Clean domain interfaces
- Provider-based architecture
- Dependency injection system
- Comprehensive test suite

### Breaking Changes
- Chat endpoints now use `ChatOrchestrator` instead of `LlmService`
- RAG use case requires `LLMProvider` instead of `OllamaService`
- Dependency injection signatures have changed

## Usage Examples

### Using the LLM Provider Directly

```python
from app.infrastructure.llm.dependencies import get_llm_provider
from app.domain.llm.entities.message import Message
from app.domain.entities.message_role import MessageRole

# Get provider instance
provider = get_llm_provider()

# Generate response
messages = [Message(content="Hello!", role=MessageRole.USER)]
response = await provider.generate(messages)

# Stream response
async for chunk in provider.generate_stream(messages):
    print(chunk)
```

### Using the Chat Orchestrator

```python
from app.infrastructure.llm.dependencies import get_chat_orchestrator
from app.schemas.chat import ChatRequest, ChatMessage

# Get orchestrator instance (configurable turns window)
orchestrator = get_chat_orchestrator()

# Process chat message with automatic session management
request = ChatRequest(
    message=ChatMessage(role="user", content="Hello!"),
    session_id="my-session"
)

async for response in orchestrator.chat(request):
    print(response.message.content)

# Get session count
session_count = await orchestrator.get_active_session_count()
print(f"Active sessions: {session_count}")

# Get turns window configuration
turns = orchestrator.get_turns_window()
print(f"Conversation turns kept: {turns}")
```

### Adding a New LLM Provider

```python
class CustomLLMProvider(LLMProvider):
    async def generate(self, messages, **kwargs):
        # Implementation
        pass

    async def generate_stream(self, messages, **kwargs):
        # Implementation
        pass

    async def health_check(self):
        # Implementation
        pass

    def get_model_info(self):
        # Implementation
        pass

    @property
    def provider_name(self):
        return "custom"
```

## Configuration

The architecture uses the existing settings system:

```python
# LLM Configuration
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
LLM_TEMPERATURE=0.7

# Chat Configuration
CHAT_TURNS_WINDOW=10              # Number of conversation turns to keep
CHAT_MAX_SESSION_AGE_HOURS=24     # Session expiration time

# RAG Configuration
RAG_RETRIEVAL_TOP_K=5
RAG_RETRIEVAL_SIMILARITY_THRESHOLD=0.7
```

## Testing

The architecture includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Mock Support**: Easy mocking of external dependencies
- **Async Testing**: Full async/await support

Run tests with:
```bash
pytest tests/unit/infrastructure/test_new_llm_architecture.py -v
```

## Future Enhancements

### Planned Features
1. **Additional LLM Providers**: OpenAI, Anthropic, Google Gemini
2. **Advanced Session Management**: Redis-based persistence
3. **Prompt Management**: Dynamic prompt templates
4. **Rate Limiting**: Built-in rate limiting per session/API key
5. **Metrics Collection**: Prometheus metrics integration
6. **Model Routing**: Automatic model selection based on query complexity

### Extension Points
- **Custom Chat Orchestrators**: Implement different chat strategies
- **Session Storage Backends**: Memory, Redis, Database
- **Prompt Management Systems**: Template-based or AI-driven
- **Health Monitoring**: Custom health check strategies

## Conclusion

This new architecture provides a solid foundation for AI-powered features while maintaining clean code principles and enabling future growth. The modular design ensures that each component can be developed, tested, and maintained independently while working together seamlessly.