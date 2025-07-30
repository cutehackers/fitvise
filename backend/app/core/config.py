from typing import List, Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get absolute path to .env file
_env_file = Path(__file__).parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_file),  # Absolute path to .env file
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow case-insensitive env var matching
        extra="ignore",
        env_prefix="",  # No prefix for cleaner env vars
        env_nested_delimiter="__"  # Use __ for nested config if needed
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