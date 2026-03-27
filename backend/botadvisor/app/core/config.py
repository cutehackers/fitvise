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

    storage_local_path: str = Field(default="./data/artifacts", alias="STORAGE_LOCAL_PATH")

    weaviate_url: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    weaviate_grpc_port: int = Field(default=50051, alias="WEAVIATE_GRPC_PORT")
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings for the current process."""
    return Settings()
