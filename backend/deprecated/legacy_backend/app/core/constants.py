"""Application constants and configuration values.

This module centralizes all magic numbers, default values, and constants
to improve maintainability and eliminate scattered magic values.
"""

from typing import Dict

# Model Token Limits
MAX_TOKENS_TABLE: Dict[str, int] = {
    "llama3.2:3b": 128000,
    "llama3.1:8b": 128000,
    "llama3.1:70b": 128000,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "mistral:7b": 8192,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
}

DEFAULT_MAX_TOKEN_LENGTH = 8192

# Performance Thresholds
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3

# Chat and LLM Configuration
MAX_MESSAGE_LENGTH_FOR_TRACING = 500
TRIM_MESSAGES_THRESHOLD = 20
DEFAULT_TURNS_WINDOW = 10
DEFAULT_MAX_SESSION_AGE_HOURS = 24

# File and Storage Limits
MAX_FILE_SIZE_MB = 10
DEFAULT_CHUNK_SIZE_FOR_DISPLAY = 500

# Health Check Thresholds
HEALTH_CHECK_INTERVAL_SECONDS = 60
MIN_SUCCESS_RATE_PERCENTAGE = 95.0
MAX_RESPONSE_TIME_MS = 5000.0

# Vector Database Configuration
WEAVIATE_DEFAULT_PORT = 8080
WEAVIATE_DEFAULT_HOST = "localhost"
WEAVIATE_DEFAULT_BATCH_SIZE = 100
WEAVIATE_DEFAULT_TIMEOUT = 30.0

# Sentence Transformer Configuration
SENTENCE_TRANSFORMER_DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
SENTENCE_TRANSFORMER_DEFAULT_DIMENSION = 768
SENTENCE_TRANSFORMER_DEFAULT_BATCH_SIZE = 32
SENTENCE_TRANSFORMER_DEFAULT_CACHE_SIZE_MB = 256

# Search Configuration
SEARCH_MAX_TOP_K = 1000
SEARCH_DEFAULT_TOP_K = 10
SEARCH_MIN_SIMILARITY = 0.0
SEARCH_BATCH_SIZE = 100
SEARCH_TIMEOUT_MS = 30000
SEARCH_CACHE_TTL_SECONDS = 3600

# RAG Configuration
RAG_DEFAULT_RETRIEVAL_TOP_K = 5
RAG_DEFAULT_SIMILARITY_THRESHOLD = 0.7
RAG_DEFAULT_PROCESSING_TIMEOUT = 300
RAG_DEFAULT_MAX_MEMORY_MB = 1024

# Framework Observability
FRAMEWORK_TRACING_DEFAULT_BATCH_SIZE = 10
FRAMEWORK_TRACING_FLUSH_INTERVAL_SECONDS = 5
FRAMEWORK_TRACING_TIMEOUT_SECONDS = 30
FRAMEWORK_TRACING_MAX_PAYLOAD_SIZE_KB = 1024

# Error Messages
class ErrorMessages:
    """Centralized error messages for consistent user experience."""

    EMPTY_QUERY = "Query cannot be empty"
    EMPTY_MESSAGE_CONTENT = "Message content cannot be empty or whitespace only"
    MESSAGES_REQUIRED = "Messages cannot be empty"
    SESSION_ID_REQUIRED = "Session ID is required"

    TIMEOUT = "LLM service is currently unavailable due to timeout"
    SERVICE_ERROR = "LLM service is currently experiencing issues"
    GENERATION_FAILED = "Failed to generate workout plan"
    UNEXPECTED_ERROR = "An unexpected error occurred while generating workout plan"

    INVALID_REQUEST = "Invalid request format"
    VALIDATION_ERROR = "Request validation failed"
    SERVICE_UNAVAILABLE = "Service temporarily unavailable"

# Service Names
class ServiceNames:
    """Standardized service names for health checks and logging."""

    WORKOUT_API = "workout-api"
    RAG_CHAT = "rag-chat-service"
    LLM_SERVICE = "llm-service"
    DOCUMENT_PROCESSOR = "document-processor"
    EMBEDDING_SERVICE = "embedding-service"