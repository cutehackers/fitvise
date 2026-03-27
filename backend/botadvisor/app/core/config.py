"""Canonical runtime settings for the BotAdvisor package."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Small, explicit settings surface for the canonical BotAdvisor runtime."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development", alias="BOTADVISOR_ENV")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="text", alias="LOG_FORMAT")
    log_file: str | None = Field(default=None, alias="LOG_FILE")

    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_base_url: str = Field(default="http://localhost:11434", alias="LLM_BASE_URL")
    llm_model: str = Field(default="llama3.2:3b", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")
    storage_local_path: str = Field(default="./data/artifacts", alias="STORAGE_LOCAL_PATH")
    storage_minio_endpoint: str | None = Field(default=None, alias="STORAGE_MINIO_ENDPOINT")
    storage_minio_access_key: str | None = Field(default=None, alias="STORAGE_MINIO_ACCESS_KEY")
    storage_minio_secret_key: str | None = Field(default=None, alias="STORAGE_MINIO_SECRET_KEY")
    storage_minio_bucket: str = Field(default="botadvisor-artifacts", alias="STORAGE_MINIO_BUCKET")
    storage_minio_secure: bool = Field(default=False, alias="STORAGE_MINIO_SECURE")

    weaviate_url: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    weaviate_grpc_port: int = Field(default=50051, alias="WEAVIATE_GRPC_PORT")
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings for the current process."""
    return Settings()
