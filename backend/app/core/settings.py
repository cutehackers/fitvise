from pathlib import Path
from typing import List, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get absolute path to .env file
_env_file = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Settings configuration class for application environment, domain, LLM, API, database, vector store, security, file upload, knowledge base, and logging.

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
        vector_store_type (Literal["chromadb", "faiss"]): Vector store type.
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
    weaviate_url: str = ""
    weaviate_api_key: str = ""

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


# Create a global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
