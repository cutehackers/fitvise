"""Vector store configuration module.

Handles configuration for different vector stores (Weaviate, ChromaDB, FAISS)
and embedding model settings.
"""

from typing import Literal, Optional
from pydantic import Field, field_validator

from app.core.config.base import BaseConfig
from app.core.constants import (
    WEAVIATE_DEFAULT_PORT,
    WEAVIATE_DEFAULT_HOST,
    WEAVIATE_DEFAULT_BATCH_SIZE,
    WEAVIATE_DEFAULT_TIMEOUT,
    SENTENCE_TRANSFORMER_DEFAULT_MODEL,
    SENTENCE_TRANSFORMER_DEFAULT_DIMENSION,
    SENTENCE_TRANSFORMER_DEFAULT_BATCH_SIZE,
    SENTENCE_TRANSFORMER_DEFAULT_CACHE_SIZE_MB,
    SEARCH_DEFAULT_TOP_K,
    SEARCH_MAX_TOP_K,
    SEARCH_MIN_SIMILARITY,
    SEARCH_BATCH_SIZE
)


class WeaviateConfig(BaseConfig):
    """Configuration for Weaviate vector database."""

    host: str = Field(
        default=WEAVIATE_DEFAULT_HOST,
        description="Weaviate host address"
    )
    port: int = Field(
        default=WEAVIATE_DEFAULT_PORT,
        ge=1,
        le=65535,
        description="Weaviate port"
    )
    scheme: Literal["http", "https"] = Field(
        default="http",
        description="Weaviate connection scheme"
    )
    auth_type: Literal["NONE", "API_KEY", "OIDC"] = Field(
        default="NONE",
        description="Weaviate authentication type"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Weaviate API key (if auth_type=API_KEY)"
    )
    timeout: float = Field(
        default=WEAVIATE_DEFAULT_TIMEOUT,
        gt=0,
        description="Weaviate connection timeout"
    )
    batch_size: int = Field(
        default=WEAVIATE_DEFAULT_BATCH_SIZE,
        gt=0,
        le=1000,
        description="Weaviate batch operation size"
    )

    @property
    def url(self) -> str:
        """Construct Weaviate URL from components."""
        return f"{self.scheme}://{self.host}:{self.port}"

    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError('Batch size must be positive')
        return v

    def validate_configuration(self) -> None:
        """Validate Weaviate configuration."""
        if self.auth_type == "API_KEY" and not self.api_key:
            raise ValueError("API key is required when auth_type is API_KEY")

    def get_config_summary(self) -> dict:
        """Get configuration summary."""
        return {
            "config_type": "WeaviateConfig",
            "url": self.url,
            "auth_type": self.auth_type,
            "timeout": self.timeout,
            "batch_size": self.batch_size
        }


class EmbeddingConfig(BaseConfig):
    """Configuration for embedding models and generation."""

    # Model Configuration
    model_name: str = Field(
        default=SENTENCE_TRANSFORMER_DEFAULT_MODEL,
        description="Embedding model name"
    )
    dimension: int = Field(
        default=SENTENCE_TRANSFORMER_DEFAULT_DIMENSION,
        gt=0,
        description="Embedding vector dimension"
    )
    device: Literal["AUTO", "CPU", "CUDA"] = Field(
        default="AUTO",
        description="Device for embedding computation"
    )
    batch_size: int = Field(
        default=SENTENCE_TRANSFORMER_DEFAULT_BATCH_SIZE,
        gt=0,
        le=128,
        description="Batch size for embedding generation"
    )

    # Caching Configuration
    show_progress: bool = Field(
        default=True,
        description="Show progress during embedding generation"
    )
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings"
    )
    cache_strategy: Literal["MEMORY", "DISK", "HYBRID", "NONE"] = Field(
        default="MEMORY",
        description="Embedding caching strategy"
    )
    cache_size_mb: int = Field(
        default=SENTENCE_TRANSFORMER_DEFAULT_CACHE_SIZE_MB,
        gt=0,
        le=4096,
        description="Cache size in MB"
    )

    @field_validator('dimension')
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Validate dimension is positive."""
        if v <= 0:
            raise ValueError('Embedding dimension must be positive')
        if v > 8192:  # Reasonable upper limit
            raise ValueError('Embedding dimension cannot exceed 8192')
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v <= 0:
            raise ValueError('Batch size must be positive')
        return v

    @field_validator('cache_size_mb')
    @classmethod
    def validate_cache_size(cls, v: int) -> int:
        """Validate cache size."""
        if v <= 0:
            raise ValueError('Cache size must be positive')
        return v

    def validate_configuration(self) -> None:
        """Validate embedding configuration."""
        # Device-specific validation
        if self.device == "CUDA" and self.batch_size > 64:
            # Suggest smaller batch size for CUDA memory management
            raise ValueError(
                "Batch size should not exceed 64 when using CUDA to avoid memory issues"
            )

        # Cache strategy validation
        if self.cache_strategy == "NONE" and self.cache_size_mb > 0:
            # Cache size is meaningless with NONE strategy
            raise ValueError(
                "Cache size must be 0 when cache strategy is NONE"
            )

    def get_config_summary(self) -> dict:
        """Get configuration summary."""
        return {
            "config_type": "EmbeddingConfig",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "cache_strategy": self.cache_strategy,
            "cache_size_mb": self.cache_size_mb,
            "normalize": self.normalize
        }


class SearchConfig(BaseConfig):
    """Configuration for search and retrieval operations."""

    # Search Parameters
    max_top_k: int = Field(
        default=SEARCH_MAX_TOP_K,
        ge=1,
        le=10000,
        description="Maximum results per search"
    )
    default_top_k: int = Field(
        default=SEARCH_DEFAULT_TOP_K,
        ge=1,
        le=self.max_top_k or SEARCH_MAX_TOP_K,
        description="Default results per search"
    )
    min_similarity: float = Field(
        default=SEARCH_MIN_SIMILARITY,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    batch_size: int = Field(
        default=SEARCH_BATCH_SIZE,
        gt=0,
        le=1000,
        description="Batch size for search operations"
    )

    # Performance Configuration
    timeout_ms: int = Field(
        default=30000,
        gt=100,
        le=300000,  # 5 minutes max
        description="Search timeout in milliseconds"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        le=86400,  # 24 hours max
        description="Cache TTL for search results"
    )

    def validate_configuration(self) -> None:
        """Validate search configuration."""
        if self.default_top_k > self.max_top_k:
            raise ValueError(
                f"default_top_k ({self.default_top_k}) cannot exceed max_top_k ({self.max_top_k})"
            )

    def get_config_summary(self) -> dict:
        """Get configuration summary."""
        return {
            "config_type": "SearchConfig",
            "max_top_k": self.max_top_k,
            "default_top_k": self.default_top_k,
            "min_similarity": self.min_similarity,
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
            "cache_ttl_seconds": self.cache_ttl_seconds
        }


class VectorStoreConfig(BaseConfig):
    """Main vector store configuration using composition."""

    # Store Selection
    vector_store_type: Literal["chromadb", "faiss", "weaviate"] = Field(
        default="weaviate",
        description="Vector store type"
    )
    vector_store_path: str = Field(
        default="./data/vector_store",
        description="Path to vector store (for local stores)"
    )

    # Component Configurations
    weaviate: WeaviateConfig = Field(
        default_factory=WeaviateConfig,
        description="Weaviate-specific configuration"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding model configuration"
    )
    search: SearchConfig = Field(
        default_factory=SearchConfig,
        description="Search configuration"
    )

    def validate_configuration(self) -> None:
        """Validate complete vector store configuration."""
        # Validate each component
        self.weaviate.validate_configuration()
        self.embedding.validate_configuration()
        self.search.validate_configuration()

        # Type-specific validation
        if self.vector_store_type == "weaviate":
            self.weaviate.validate_configuration()
        elif self.vector_store_type in ["chromadb", "faiss"]:
            if not self.vector_store_path:
                raise ValueError(f"vector_store_path is required for {self.vector_store_type}")

    def get_config_summary(self) -> dict:
        """Get complete configuration summary."""
        return {
            "config_type": "VectorStoreConfig",
            "vector_store_type": self.vector_store_type,
            "vector_store_path": self.vector_store_path,
            "weaviate": self.weaviate.get_config_summary(),
            "embedding": self.embedding.get_config_summary(),
            "search": self.search.get_config_summary()
        }