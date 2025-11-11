# Fitvise - AI-Powered Fitness Assistant

> **Intelligent Fitness Coaching** through conversational AI, personalized workout guidance, and advanced RAG-powered document retrieval

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Flutter](https://img.shields.io/badge/Flutter-3.8+-02569B.svg?style=flat&logo=Flutter)](https://flutter.dev)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Testing](https://img.shields.io/badge/pytest-comprehensive-green.svg)](https://pytest.org/)

## 🌟 Overview

Fitvise is a comprehensive AI-powered fitness platform that provides personalized workout plans, nutrition guidance, and health coaching through intelligent conversational interfaces. Built with Domain-Driven Design principles, the platform features a production-grade RAG (Retrieval-Augmented Generation) pipeline for intelligent document processing and retrieval. It leverages Large Language Models (LLMs) to deliver tailored fitness advice across multiple client applications.

### Key Features

🤖 **AI-Powered Coaching** - Personalized workout plans, nutrition advice, and exercise recommendations
📱 **Multi-Platform Support** - Native mobile apps (Flutter) and web interfaces
🏗️ **Domain-Driven Design** - Modular FastAPI architecture with clear separation of concerns and DDD principles
📚 **RAG Pipeline** - Multi-phase document ingestion, embedding generation, and intelligent retrieval system
🎯 **Conversational Interface** - Natural language interaction with quick replies and voice input
⚡ **Real-Time Chat** - Typing indicators, message editing, and responsive user experience
🔒 **Type-Safe Architecture** - Full validation with Pydantic (backend) and type-safe models (frontend)
🧪 **Comprehensive Testing** - Unit, integration, and E2E tests with 30+ tests for RAG pipeline phases

## 🏗️ Architecture

Fitvise follows a sophisticated microservices architecture with RAG pipeline and Domain-Driven Design principles:

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
├──────────────────┬─────────────────────────┬─────────────────────────┤
│  Flutter App     │   React Web App         │   Mobile/Web            │
│  (iOS/Android)   │   (Browser)             │   Platforms             │
└────────┬─────────┴────────────┬────────────┴─────────────────────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (DDD ARCHITECTURE)                │
├──────────────────────────────────────────────────────────────────────┤
│  API Layer (Endpoints) │ Schemas │ Routing                           │
├──────────────────────────────────────────────────────────────────────┤
│  Application Layer     │ Use Cases │ DTOs │ Orchestration            │
├──────────────────────────────────────────────────────────────────────┤
│  Domain Layer          │ Entities │ Value Objects │ Repositories     │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer  │ DB │ Services │ External Integrations       │
└──────────────────────────────────────────────────────────────────────┘
         │                      │                      │
         │                      │                      │
         ▼                      ▼                      ▼
  ┌─────────────┐       ┌──────────────┐      ┌──────────────┐
  │ LLM Service │       │ RAG Pipeline │      │   Vector DB  │
  │  (Ollama)   │       │              │      │  (Weaviate)  │
  └─────────────┘       └──────────────┘      └──────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
         Phase 1: Infra  Phase 2: Ingest  Phase 3: Embed
         Initialize      Document         Generate
         Services        Processing       Embeddings
```

### Architecture Overview

**Client Layer**: Native mobile (Flutter) and web (React) applications
**FastAPI Backend**: Layered architecture following Domain-Driven Design principles
- **API Layer**: RESTful endpoints with Pydantic validation
- **Application Layer**: Business logic, use cases, and DTOs
- **Domain Layer**: Core entities, value objects, and repository interfaces
- **Infrastructure Layer**: Database, external services, and implementations

**RAG Pipeline**: Three-phase intelligent document processing system
- **Phase 1 (Infrastructure)**: Initialize vector databases and validate services
- **Phase 2 (Ingestion)**: Document discovery, processing, and chunking
- **Phase 3 (Embedding)**: Embedding generation and vector storage

**External Services**: LLM integration and vector database for intelligent retrieval

### Project Structure

```
fitvise/
├── backend/              # FastAPI REST API server
│   ├── app/             # Application code
│   │   ├── api/v1/      # API version 1 endpoints
│   │   ├── core/        # Configuration and settings
│   │   ├── schemas/     # Pydantic models
│   │   ├── services/    # Business logic and LLM integration
│   │   └── main.py      # FastAPI application entry point
│   ├── requirements.txt # Python dependencies
│   └── README.md        # Backend documentation
├── frontend/            # Flutter mobile application
│   ├── lib/            # Dart/Flutter source code
│   │   ├── screens/    # Application screens
│   │   ├── providers/  # State management
│   │   ├── services/   # API integration
│   │   └── widgets/    # UI components
│   ├── pubspec.yaml    # Flutter dependencies
│   └── README.md       # Flutter app documentation
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** for backend development
- **Flutter 3.8+** for mobile app development
- **Ollama** or compatible LLM service

### Backend Setup

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd fitvise/backend
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip sync requirements.txt
```

2. **Configure environment variables**:
```bash
# Create .env file
cat > .env << EOF
APP_NAME=Fitvise Backend API
APP_VERSION=1.0.0
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
API_HOST=0.0.0.0
API_PORT=8000
EOF
```

3. **Start the backend server**:
```bash
python run.py
# API available at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Flutter Mobile App Setup

1. **Navigate to Flutter directory**:
```bash
cd fitvise/frontend
flutter pub get
```

2. **Configure API endpoint** (optional):
```bash
# For custom API endpoint
flutter run --dart-define=API_BASE_URL=http://your-api-url:8000
```

3. **Run the mobile app**:
```bash
flutter run  # Debug mode
flutter run --release  # Production mode
```

### LLM Service Setup

Install and configure Ollama for AI functionality:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull AI model
ollama pull llama3.2

# Start Ollama service
ollama serve  # Runs on localhost:11434
```

## 📱 Applications

### Frontend (Flutter: Android, iOS, Web)

**Cross-platform mobile application** with native performance:

**Key Features**:
- 💪 AI Fitness Assistant with personalized guidance
- 🌙 Theme switching with system preference detection
- 🗣️ Voice input for hands-free interaction
- 📎 File upload for workout photos and documents
- ✏️ Message editing capabilities
- 📋 Quick action buttons for common fitness queries

- **Platforms**: iOS, Android, macOS, Windows, Linux, Web
- **Architecture**: Clean Architecture with Provider state management
- **Features**: Conversational AI chat, voice input with speech-to-text, health platform integration
- **API Integration**: Type-safe Retrofit/Dio HTTP client with Freezed models
- **State Management**: Provider pattern with SharedPreferences persistence
- **Permissions**: Microphone access, health data integration, file system access

**Technical Stack**:
- **HTTP Client**: Dio + Retrofit for type-safe API calls
- **State Management**: Provider with `ChangeNotifier` pattern
- **Models**: Freezed + JSON annotation for immutable data classes
- **UI**: Material Design with custom theming
- **Code Generation**: build_runner for model generation

**Key Components**:
- `ChatScreen` - Main conversational interface with typing indicators
- `ChatProvider` - Message state management and AI interactions
- `ThemeProvider` - Light/dark theme switching with system preference
- `AgentApi` - Type-safe API client with comprehensive error handling
- `PromptRequest`/`PromptResponse` - Freezed models for API communication

### Backend (FastAPI)

**Production-ready REST API** with sophisticated architecture:

- **Framework**: FastAPI with async operations and comprehensive error handling
- **Architecture**: Domain-Driven Design with clear layer separation (domain, application, infrastructure, API)
- **Validation**: Pydantic models for type safety and comprehensive data validation
- **RAG Pipeline**: Multi-phase document processing with intelligent embedding and retrieval
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **Monitoring**: Health checks, service availability monitoring, and performance metrics
- **Integration**: LLM service with connection pooling, embedding service, and vector database

**Core Components**:
- **RAG Pipeline** - Three-phase orchestration (infrastructure setup, document ingestion, embedding generation)
- **Domain Layer** - Entities, value objects, repositories, and domain-specific exceptions
- **Application Layer** - Use cases, business logic orchestration, and DTOs
- **Infrastructure Layer** - Repository implementations, database management, external service integrations
- **LLMService** - AI model integration with error handling and health monitoring
- **WorkoutEndpoints** - Fitness-specific API endpoints with comprehensive validation

## 🔌 API Integration

### Core Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/v1/workout/health` | Service health monitoring |
| `POST` | `/api/v1/workout/prompt` | AI fitness prompts |
| `GET` | `/api/v1/workout/models` | Model information |

### Example Usage

**Generate workout plan**:
```bash
curl -X POST 'http://localhost:8000/api/v1/workout/prompt' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Create a 30-minute HIIT workout for beginners",
    "context": "No equipment needed, limited space",
    "temperature": 0.7
  }'
```

**Response**:
```json
{
  "response": "Here's a beginner-friendly 30-minute HIIT workout...",
  "model": "llama3.2",
  "tokens_used": 347,
  "duration_ms": 1250.5,
  "success": true
}
```

## 🧪 Development

### Backend Development

```bash
cd backend
# Development server with auto-reload
python run.py

# Run example client
python api_example.py

# Test configuration
python test_settings.py
```

### Frontend Development

**Flutter**:
```bash
cd frontend
flutter analyze          # Static analysis
flutter test             # Run tests
flutter build apk        # Build Android APK
flutter build ios        # Build iOS app
```

**React**:
```bash
cd frontend-react
npm start               # Development server (http://localhost:3000)
npm test                # Run tests
npm run build          # Production build
npm run eject          # Eject from Create React App (one-way operation)
```

### Code Generation

**Flutter** (for Freezed models and API clients):
```bash
cd frontend
flutter packages pub run build_runner build       # Generate models
flutter packages pub run build_runner watch      # Watch for changes
```

### Dependencies

**Flutter Dependencies**:
- `provider: ^6.1.1` - State management
- `dio: ^5.4.0` + `retrofit: ^4.1.0` - HTTP client and API generation
- `freezed: ^3.2.0` + `json_annotation: ^4.8.1` - Immutable models
- `speech_to_text: ^6.6.0` - Voice input support
- `shared_preferences: ^2.2.2` - Local storage for themes

## 🔧 Configuration

### Environment Variables

**Backend Configuration**:
- `LLM_BASE_URL`: LLM service endpoint (default: http://localhost:11434)
- `LLM_MODEL`: AI model identifier (default: llama3.2)
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)

**Frontend Configuration**:

*Flutter*:
- **Local Development**: `http://localhost:8000` (debug mode default)
- **Custom API**: `flutter run --dart-define=API_BASE_URL=http://your-api-url:8000`  
- **Environment**: Auto-detection based on build mode in `lib/config/app_config.dart`

*React*:
- **Development**: Configure API endpoints in components
- **Build-time**: Environment variables via `.env` files
- **Backend Integration**: Designed to work with `../backend` directory

### Multi-Environment Support

**Backend**:
- `local`: Development with debug logging and CORS enabled
- `production`: Optimized with security features and disabled docs

**Flutter**:
- **Debug builds**: Connect to `localhost:8000` automatically
- **Release builds**: Use production API endpoints
- **Platform support**: iOS, Android, Web, Desktop (macOS, Windows, Linux)

**React**:
- **Development**: `npm start` with hot reload at `localhost:3000`
- **Production**: `npm run build` for optimized static files

## 📊 Features

### AI-Powered Fitness Coaching

- **Personalized Workouts**: Custom workout plans based on user preferences
- **Nutrition Guidance**: Dietary advice and meal planning
- **Progress Tracking**: Health metrics and goal monitoring  
- **Educational Content**: Fitness knowledge and best practices

### User Experience

- **Conversational Interface**: Natural language interaction with AI fitness coach
- **Quick Replies**: Pre-defined fitness prompts and workout suggestions
- **Voice Input**: Speech-to-text for hands-free interaction (Flutter & React)
- **Message Editing**: Edit and refine messages in real-time
- **File Upload**: Share workout photos, plans, and documents (React)
- **Theme Support**: Light/dark mode with system preference detection
- **Responsive Design**: Optimized for mobile, tablet, and desktop
- **Typing Indicators**: Real-time chat feedback and status updates

### Technical Features

- **Type Safety**: Comprehensive validation across all layers
- **Error Handling**: Graceful error recovery and user feedback
- **Performance**: Optimized for speed and resource efficiency
- **Health Monitoring**: Service availability and performance metrics
- **Cross-Platform**: Consistent experience across mobile and web

## 🔐 Security & Privacy

### Input Validation

- Query length limits (1000 characters)
- Context length limits (2000 characters)
- Parameter validation (temperature: 0.0-2.0)
- Type validation with Pydantic/TypeScript

### Data Privacy

- No persistent storage of user conversations
- Temporary processing only
- Configurable data retention policies
- CORS configuration for production security

## 📈 Performance & Monitoring

### Health Monitoring

- **Service Health**: Comprehensive health check endpoints
- **LLM Availability**: Real-time monitoring of AI service
- **Performance Metrics**: Response times and token usage
- **Error Tracking**: Structured logging and error analysis

### Optimization

- **Async Operations**: Non-blocking I/O throughout
- **Connection Pooling**: Efficient HTTP client management
- **Caching**: Intelligent caching strategies
- **Resource Limits**: Configurable limits and timeouts

## 🤝 Contributing

### Development Guidelines

- **Type Safety**: Use type annotations in all languages
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Keep README files updated
- **Code Style**: Follow language-specific conventions
- **Error Handling**: Implement comprehensive error handling

### Code Style

- **Python**: PEP 8 with type hints and async/await
- **Dart/Flutter**: Official Dart style guide with lints
- **JavaScript/React**: ESLint with React best practices

## 🚀 Project Status

**Current Development Status**: Active development with advanced features and production-grade architecture

✅ **Backend**: Production-ready FastAPI server with Domain-Driven Design architecture, LLM integration, and RAG pipeline
✅ **React Web App**: Full-featured web interface with modern UI, voice input, and file upload capabilities
✅ **Flutter Mobile**: Cross-platform mobile app with native performance and comprehensive feature support

**Key Achievements**:
- **AI-Powered Fitness Coaching**: Complete backend with personalized workout and nutrition guidance
- **RAG Pipeline**: Production-grade document processing with 3-phase orchestration (infrastructure, ingestion, embedding)
- **Domain-Driven Design**: Modular architecture with clear separation of concerns (domain, application, infrastructure layers)
- **Comprehensive Testing**: 30+ tests for RAG pipeline phases, unit, integration, and E2E test coverage
- **Responsive UI**: Web interface with voice input, file upload, and modern design patterns
- **Cross-Platform Mobile**: Native performance on iOS, Android with speech-to-text integration
- **Type-Safe Integration**: Type-safe API integration across all platforms with Pydantic, Freezed, and native type systems
- **Production Monitoring**: Comprehensive health monitoring, performance metrics, and structured logging

## 📚 Documentation

- **[Backend API Documentation](backend/API.md)** - Comprehensive endpoint documentation with examples
- **[Backend README](backend/README.md)** - Backend setup, architecture, and development guide
- **[Backend Development Guide](backend/CLAUDE.md)** - Comprehensive RAG pipeline, DDD architecture, and critical development patterns
- **[React README](frontend-react/README.md)** - Web app features, setup, and customization
- **[Flutter Documentation](frontend/CLAUDE.md)** - Mobile app architecture and development guide
- **Interactive API Docs** - Available at `/docs` when backend is running

## 📄 License

This project is part of the Fitvise fitness application suite.

---

**Built with ❤️ for intelligent fitness coaching and healthy living**