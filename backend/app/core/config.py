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
    vector_store_type: Literal["chromadb", "faiss"]
    vector_store_path: str
    embedding_model: str
    vector_dimension: int  # MiniLM embedding dimension

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


# Create a global settings instance
settings = Settings()
