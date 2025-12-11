"""Configuration Providers.

This module provides configuration and settings providers for the DI container.
It handles loading and validation of application settings from environment variables.
"""

from pathlib import Path
from typing import Optional

from dependency_injector import containers, providers

from app.core.settings import Settings
from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.config.vector_stores.weaviate_config import WeaviateConfig


class ConfigProviders(containers.DeclarativeContainer):
    """Configuration providers for the FitVise application.

    Provides centralized access to all configuration objects including
    main application settings, model configurations, and external service
    configurations.
    """

    # Main application settings
    settings = providers.Singleton(
        Settings,
    )

    # Embedding model configuration
    embedding_config = providers.Singleton(
        EmbeddingModelConfig.default,
    )

    # Real-time embedding configuration (for API endpoints)
    realtime_embedding_config = providers.Singleton(
        EmbeddingModelConfig.for_realtime,
    )

    # Weaviate vector store configuration
    weaviate_config = providers.Singleton(
        WeaviateConfig,
    )

    # Local development Weaviate configuration
    local_weaviate_config = providers.Singleton(
        WeaviateConfig.for_local_development,
    )

    # Environment-specific configurations
    is_production = providers.Factory(lambda settings: settings.environment == "production", settings=settings)
    is_development = providers.Factory(lambda settings: settings.environment == "local", settings=settings)
    debug_enabled = providers.Factory(lambda settings: settings.debug, settings=settings)
    cors_origins = providers.Factory(lambda settings: settings.cors_origins_list, settings=settings)
    database_url = providers.Factory(lambda settings: settings.database_url, settings=settings)
    log_level = providers.Factory(lambda settings: settings.log_level.lower(), settings=settings)

    upload_path = providers.Factory(lambda settings: Path(settings.upload_directory), settings=settings)
    knowledge_base_path = providers.Factory(lambda settings: Path(settings.knowledge_base_path), settings=settings)
    vector_store_path = providers.Factory(lambda settings: Path(settings.vector_store_path), settings=settings)

    langfuse_enabled = providers.Factory(lambda settings: settings.langfuse_configured, settings=settings)
    rag_enabled = providers.Factory(lambda settings: settings.rag_enabled, settings=settings)
    framework_observability_enabled = providers.Factory(
        lambda settings: settings.framework_observability_configured,
        settings=settings,
    )
