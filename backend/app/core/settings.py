import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get absolute path to .env file
_env_file = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Settings configuration class for application environment, domain, LLM, API, database, vector store, security, file upload, knowledge base, RAG-LLM integration, and logging.

    Attributes:
        model_config (SettingsConfigDict): Configuration for environment variables and settings parsing.
        app_name (str): Application name.
        app_version (str): Application version.
        app_description (str): Description of the application.
        environment (Literal["local", "staging", "production"]): Deployment environment.
        debug (bool): Debug mode flag.
        domain (str): Main domain for the application.
        api_host (str): API host address.
        api_port (int): API port number.
        llm_base_url (str): Base URL for LLM service.
        llm_model (str): LLM model name.
        llm_timeout (int): Timeout for LLM requests (seconds).
        llm_temperature (float): LLM temperature setting.
        llm_max_tokens (int): Maximum tokens for LLM responses.
        api_v1_prefix (str): API v1 route prefix.
        cors_origins (str): Allowed CORS origins (comma-separated or "*").
        cors_allow_credentials (bool): Allow CORS credentials.
        cors_allow_methods (str): Allowed CORS methods (comma-separated or "*").
        cors_allow_headers (str): Allowed CORS headers (comma-separated or "*").
        database_url (str): Database connection URL.
        database_echo (bool): Enable SQL query logging.
        vector_store_type (Literal["chromadb", "faiss", "weaviate"]): Vector store type.
        vector_store_path (str): Path to vector store.
        embedding_model (str): Embedding model name.
        vector_dimension (int): Embedding vector dimension.
        secret_key (str): Secret key for security.
        access_token_expire_minutes (int): Access token expiration time (minutes).
        algorithm (str): Security algorithm.
        max_file_size (int): Maximum file upload size (bytes).
        allowed_file_types (str): Allowed file types for upload (comma-separated).
        upload_directory (str): Directory for file uploads.
        knowledge_base_path (str): Path to knowledge base.
        auto_index_on_startup (bool): Auto index knowledge base on startup.
        index_update_interval (int): Interval for updating index (seconds).
        log_level (str): Logging level.
        log_file (str): Log file path.
        log_rotation (str): Log rotation policy.
        log_retention (str): Log retention policy.
        rag_retrieval_top_k (int): Number of chunks to retrieve for RAG.
        rag_retrieval_similarity_threshold (float): Minimum similarity for retrieval.
        llm_context_window (int): Maximum context tokens for LLM.
        llm_reserve_tokens (int): Tokens reserved for response generation.
        context_truncation_strategy (Literal["recent", "relevant", "summarize"]): Context truncation strategy.
        llm_max_concurrent (int): Maximum concurrent LLM requests.
        health_check_interval (int): Health check interval (seconds).
        health_min_success_rate (float): Minimum success rate percentage.
        health_max_response_time_ms (float): Maximum response time threshold.

    Properties:
        cors_origins_list (List[str]): List of allowed CORS origins.
        cors_allow_methods_list (List[str]): List of allowed CORS methods.
        cors_allow_headers_list (List[str]): List of allowed CORS headers.
        allowed_file_types_list (List[str]): List of allowed file types for upload.
    """

    model_config = SettingsConfigDict(
        env_file=str(_env_file),  # Absolute path to .env file
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow case-insensitive env var matching
        extra="ignore",
        env_prefix="",  # No prefix for cleaner env vars
        env_nested_delimiter="__",  # Use __ for nested config if needed
    )

    # App Information
    app_name: str
    app_version: str
    app_description: str

    # Environment Configuration
    environment: Literal["local", "staging", "production"]
    debug: bool

    # Domain Configuration
    domain: str
    api_host: str
    api_port: int

    # LLM Configuration
    llm_base_url: str
    llm_model: str
    llm_timeout: int  # seconds
    llm_temperature: float
    llm_max_tokens: int

    # API Configuration
    api_v1_prefix: str
    cors_origins: str  # In production, specify exact origins as comma-separated
    cors_allow_credentials: bool
    cors_allow_methods: str  # Comma-separated methods
    cors_allow_headers: str  # Comma-separated headers

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def cors_allow_methods_list(self) -> List[str]:
        """Convert comma-separated methods to list"""
        if self.cors_allow_methods == "*":
            return ["*"]
        return [method.strip() for method in self.cors_allow_methods.split(",") if method.strip()]

    @property
    def cors_allow_headers_list(self) -> List[str]:
        """Convert comma-separated headers to list"""
        if self.cors_allow_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]

    # Database Configuration
    database_url: str
    database_echo: bool  # Set to True for SQL query logging

    # Vector Store Configuration
    vector_store_type: Literal["chromadb", "faiss", "weaviate"]
    vector_store_path: str
    embedding_model: str
    vector_dimension: int  # MiniLM embedding dimension

    # Weaviate Vector Database Configuration (Task 2.2.1)
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_scheme: Literal["http", "https"] = "http"
    weaviate_auth_type: Literal["NONE", "API_KEY", "OIDC"] = "NONE"
    weaviate_api_key: Optional[str] = None
    weaviate_timeout: float = 30.0
    weaviate_batch_size: int = 100

    @property
    def weaviate_url(self) -> str:
        """Construct Weaviate URL from components."""
        return f"{self.weaviate_scheme}://{self.weaviate_host}:{self.weaviate_port}"

    # Sentence-Transformers Embedding Configuration (Task 2.2.1)
    sentence_transformer_model: str = "Alibaba-NLP/gte-multilingual-base"
    sentence_transformer_dimension: int = 768
    sentence_transformer_device: Literal["AUTO", "CPU", "CUDA"] = "AUTO"
    sentence_transformer_batch_size: int = 32
    sentence_transformer_show_progress: bool = True
    sentence_transformer_normalize: bool = True
    sentence_transformer_cache_strategy: Literal["MEMORY", "DISK", "HYBRID", "NONE"] = "MEMORY"
    sentence_transformer_cache_size_mb: int = 256

    # Search Configuration
    search_max_top_k: int = 1000  # Maximum results per search
    search_default_top_k: int = 10  # Default results per search
    search_min_similarity: float = 0.0  # Minimum similarity threshold
    search_batch_size: int = 100  # Batch size for operations
    search_timeout_ms: int = 30000  # Search timeout in milliseconds
    search_cache_ttl_seconds: int = 3600  # Cache TTL for search results

    # Security Configuration
    secret_key: str
    access_token_expire_minutes: int
    algorithm: str

    # File Upload Configuration
    max_file_size: int  # 10MB
    allowed_file_types: str  # Comma-separated file extensions
    upload_directory: str

    @property
    def allowed_file_types_list(self) -> List[str]:
        """Convert comma-separated file types to list"""
        return [ext.strip() for ext in self.allowed_file_types.split(",") if ext.strip()]

    # Knowledge Base Configuration
    knowledge_base_path: str
    auto_index_on_startup: bool
    index_update_interval: int  # seconds (1 hour)

    # Logging Configuration
    log_level: str
    log_file: str
    log_rotation: str
    log_retention: str

    # RAG System Configuration
    rag_enabled: bool = True
    rag_data_scan_paths: str = ""  # Comma-separated paths to scan for documents
    rag_max_scan_depth: int = 5
    rag_min_file_count: int = 5
    rag_model_save_path: str = "models"
    rag_export_path: str = "exports"
    
    # RAG Data Source Configuration
    rag_database_connections: str = ""  # JSON string of database configs
    rag_api_endpoints: str = ""  # Comma-separated API endpoints
    rag_include_common_apis: bool = True
    rag_api_timeout: int = 10
    rag_api_validation_enabled: bool = True
    
    # RAG ML Configuration  
    rag_ml_model_type: Literal["logistic_regression", "naive_bayes", "random_forest"] = "logistic_regression"
    rag_ml_max_features: int = 10000
    rag_ml_ngram_range_min: int = 1
    rag_ml_ngram_range_max: int = 2
    rag_ml_min_confidence: float = 0.6
    rag_ml_target_accuracy: float = 0.85
    rag_ml_synthetic_data_size: int = 100
    rag_ml_auto_retrain: bool = False
    rag_ml_retrain_interval_hours: int = 24
    
    # RAG Processing Configuration
    rag_batch_size: int = 100
    rag_processing_timeout: int = 300  # seconds
    rag_max_memory_mb: int = 1024
    rag_enable_quality_validation: bool = True
    rag_quality_threshold: float = 0.5

    # RAG-LLM Integration Configuration (Task 3.1.1)
    rag_retrieval_top_k: int = 5  # Number of chunks to retrieve
    rag_retrieval_similarity_threshold: float = 0.7  # Minimum similarity for retrieval

    # Context Window Configuration (Task 3.1.3)
    llm_context_window: int = 4000  # Maximum context tokens (llama3.2:3b)
    llm_reserve_tokens: int = 500  # Tokens reserved for response generation
    context_truncation_strategy: Literal["recent", "relevant", "summarize"] = "relevant"

    # LLM Performance Configuration
    llm_max_concurrent: int = 10  # Maximum concurrent LLM requests

    # Health Monitoring Configuration
    health_check_interval: int = 60  # Health check interval (seconds)
    health_min_success_rate: float = 95.0  # Minimum success rate percentage
    health_max_response_time_ms: float = 5000.0  # Maximum response time threshold

    # LangFuse Configuration
    # - LANGFUSE_SECRET_KEY
    # - LANGFUSE_PUBLIC_KEY
    # - LANGFUSE_HOST
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: Optional[str] = None

    @property
    def langfuse_configured(self) -> bool:
        """Check if LangFuse environment variables are properly configured."""
        return bool(self.langfuse_secret_key and self.langfuse_public_key)

    # Framework Observability Configuration
    # Core framework tracing settings
    llamaindex_tracing_enabled: bool = True
    langchain_tracing_enabled: bool = True
    framework_tracing_sample_rate: float = 1.0
    framework_tracing_privacy_masking: bool = True
    framework_tracing_max_payload_size: int = 1024  # KB

    # LlamaIndex specific tracing settings
    llamaindex_trace_document_loading: bool = True
    llamaindex_trace_vector_operations: bool = True
    llamaindex_trace_embedding_generation: bool = True
    llamaindex_trace_retrieval_operations: bool = True
    llamaindex_max_chunk_size_for_tracing: int = 1000

    # LangChain specific tracing settings
    langchain_trace_message_processing: bool = True
    langchain_trace_chain_execution: bool = True
    langchain_trace_tool_usage: bool = True
    langchain_trace_prompt_optimization: bool = True
    langchain_max_message_length_for_tracing: int = 500

    # Performance optimization settings
    framework_tracing_async_batching: bool = True
    framework_tracing_batch_size: int = 10
    framework_tracing_flush_interval_seconds: int = 5
    framework_tracing_timeout_seconds: int = 30

    @property
    def rag_data_scan_paths_list(self) -> List[str]:
        """Convert comma-separated scan paths to list"""
        if not self.rag_data_scan_paths.strip():
            return []
        return [path.strip() for path in self.rag_data_scan_paths.split(",") if path.strip()]
    
    @property
    def rag_api_endpoints_list(self) -> List[str]:
        """Convert comma-separated API endpoints to list"""
        if not self.rag_api_endpoints.strip():
            return []
        return [endpoint.strip() for endpoint in self.rag_api_endpoints.split(",") if endpoint.strip()]
    
    @property
    def rag_database_connections_list(self) -> List[dict]:
        """Parse JSON database connections"""
        import json
        if not self.rag_database_connections.strip():
            return []
        try:
            return json.loads(self.rag_database_connections)
        except json.JSONDecodeError:
            return []
    
    @property
    def rag_ml_ngram_range(self) -> tuple:
        """Get ngram range as tuple"""
        return (self.rag_ml_ngram_range_min, self.rag_ml_ngram_range_max)

    @field_validator('weaviate_port')
    @classmethod
    def validate_weaviate_port(cls, v):
        """Validate Weaviate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError('Weaviate port must be between 1 and 65535')
        return v

    @field_validator('weaviate_timeout')
    @classmethod
    def validate_weaviate_timeout(cls, v):
        """Validate Weaviate timeout is positive."""
        if v <= 0:
            raise ValueError('Weaviate timeout must be positive')
        return v

    @field_validator('weaviate_batch_size')
    @classmethod
    def validate_weaviate_batch_size(cls, v):
        """Validate Weaviate batch size is positive."""
        if v <= 0:
            raise ValueError('Weaviate batch size must be positive')
        return v

    @field_validator('sentence_transformer_dimension')
    @classmethod
    def validate_sentence_transformer_dimension(cls, v):
        """Validate sentence transformer dimension is positive."""
        if v <= 0:
            raise ValueError('Sentence transformer dimension must be positive')
        return v

    @field_validator('sentence_transformer_batch_size')
    @classmethod
    def validate_sentence_transformer_batch_size(cls, v):
        """Validate sentence transformer batch size is positive."""
        if v <= 0:
            raise ValueError('Sentence transformer batch size must be positive')
        return v

    @field_validator('sentence_transformer_cache_size_mb')
    @classmethod
    def validate_sentence_transformer_cache_size(cls, v):
        """Validate sentence transformer cache size is positive."""
        if v <= 0:
            raise ValueError('Sentence transformer cache size must be positive')
        return v

    
    @field_validator('langfuse_secret_key', 'langfuse_public_key')
    @classmethod
    def validate_langfuse_keys(cls, v, info):
        """Validate LangFuse API keys format and consistency."""
        # Get the field name
        field_name = info.field_name

        # If the field is None, that's okay (optional)
        if v is None:
            return v

        # Check if key is a valid format (should be at least 10 characters)
        if len(v) < 10:
            raise ValueError(f'{field_name} must be at least 10 characters long')

        # Check if key contains only valid characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        if not all(c in allowed_chars for c in v):
            raise ValueError(f'{field_name} contains invalid characters')

        return v

    
    
    # Framework Observability Validators

    @field_validator('framework_tracing_sample_rate')
    @classmethod
    def validate_framework_tracing_sample_rate(cls, v):
        """Validate framework tracing sample rate is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Framework tracing sample rate must be between 0.0 and 1.0')
        return v

    @field_validator('framework_tracing_max_payload_size')
    @classmethod
    def validate_framework_tracing_max_payload_size(cls, v):
        """Validate framework tracing max payload size is positive."""
        if v <= 0:
            raise ValueError('Framework tracing max payload size must be positive')
        if v > 10240:  # 10MB limit
            raise ValueError('Framework tracing max payload size cannot exceed 10240 KB (10MB)')
        return v

    @field_validator('framework_tracing_batch_size')
    @classmethod
    def validate_framework_tracing_batch_size(cls, v):
        """Validate framework tracing batch size is positive."""
        if v <= 0:
            raise ValueError('Framework tracing batch size must be positive')
        if v > 1000:
            raise ValueError('Framework tracing batch size cannot exceed 1000')
        return v

    @field_validator('framework_tracing_flush_interval_seconds')
    @classmethod
    def validate_framework_tracing_flush_interval(cls, v):
        """Validate framework tracing flush interval is positive."""
        if v <= 0:
            raise ValueError('Framework tracing flush interval must be positive')
        if v > 300:  # 5 minutes max
            raise ValueError('Framework tracing flush interval cannot exceed 300 seconds')
        return v

    @field_validator('framework_tracing_timeout_seconds')
    @classmethod
    def validate_framework_tracing_timeout(cls, v):
        """Validate framework tracing timeout is positive."""
        if v <= 0:
            raise ValueError('Framework tracing timeout must be positive')
        if v > 600:  # 10 minutes max
            raise ValueError('Framework tracing timeout cannot exceed 600 seconds')
        return v

    @field_validator('llamaindex_max_chunk_size_for_tracing')
    @classmethod
    def validate_llamaindex_max_chunk_size(cls, v):
        """Validate LlamaIndex max chunk size for tracing is positive."""
        if v <= 0:
            raise ValueError('LlamaIndex max chunk size for tracing must be positive')
        if v > 10000:
            raise ValueError('LlamaIndex max chunk size for tracing cannot exceed 10000')
        return v

    @field_validator('langchain_max_message_length_for_tracing')
    @classmethod
    def validate_langchain_max_message_length(cls, v):
        """Validate LangChain max message length for tracing is positive."""
        if v <= 0:
            raise ValueError('LangChain max message length for tracing must be positive')
        if v > 10000:
            raise ValueError('LangChain max message length for tracing cannot exceed 10000')
        return v

    def validate_framework_observability_configuration(self) -> None:
        """Validate complete framework observability configuration for consistency."""
        # If any framework tracing is enabled, basic analytics should be enabled
        framework_tracing_enabled = (
            self.llamaindex_tracing_enabled or
            self.langchain_tracing_enabled
        )

        if framework_tracing_enabled and not self.langfuse_configured:
            logger.warning(
                "Framework tracing is enabled but LangFuse analytics is not properly configured. "
                "Framework tracing will be disabled until analytics is configured."
            )

        # Validate sampling rate consistency
        if framework_tracing_enabled and self.framework_tracing_sample_rate <= 0:
            logger.warning(
                "Framework tracing is enabled but sample rate is 0. No traces will be collected."
            )

    @property
    def framework_observability_configured(self) -> bool:
        """Check if framework observability is properly configured."""
        return (
            self.langfuse_configured and
            (self.llamaindex_tracing_enabled or self.langchain_tracing_enabled) and
            self.framework_tracing_sample_rate > 0
        )

    def model_post_init(self, __context) -> None:
        """Validate configuration after model initialization."""
        try:
            # Note: LangFuse configuration now uses standard environment variables
            # No additional validation needed for LangFuse settings
            self.validate_framework_observability_configuration()
        except ValueError as e:
            raise ValueError(f"Configuration error: {e}")

    def __hash__(self) -> int:
        """Make Settings hashable by hashing its JSON representation.

        This is needed because FastAPI's dependency injection system may need to use
        the Settings instance as a dictionary key internally.
        """
        return hash(self.model_dump_json())


# Create a global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
