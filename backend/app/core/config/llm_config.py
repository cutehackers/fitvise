"""LLM configuration module.

Handles all LLM-related settings including model configuration,
performance settings, and provider-specific options.
"""

from typing import Literal, Optional
from pydantic import Field, field_validator

from app.core.config.base import BaseConfig
from app.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRY_ATTEMPTS,
    MAX_TOKENS_TABLE,
    DEFAULT_MAX_TOKEN_LENGTH,
    TRIM_MESSAGES_THRESHOLD
)


class LLMConfig(BaseConfig):
    """Configuration for LLM service and model settings."""

    # Basic LLM Configuration
    llm_base_url: str = Field(..., description="Base URL for LLM service")
    llm_model: str = Field(..., description="LLM model name")
    llm_timeout: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        description="Timeout for LLM requests (seconds)"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature setting"
    )
    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens for LLM responses"
    )

    # Performance Configuration
    llm_max_concurrent: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Maximum concurrent LLM requests"
    )

    # Context Window Configuration
    llm_context_window: int = Field(
        default=DEFAULT_MAX_TOKEN_LENGTH,
        gt=0,
        description="Maximum context tokens for LLM"
    )
    llm_reserve_tokens: int = Field(
        default=500,
        gt=0,
        description="Tokens reserved for response generation"
    )
    context_truncation_strategy: Literal["recent", "relevant", "summarize"] = Field(
        default="relevant",
        description="Strategy for context truncation"
    )

    # Message Processing Configuration
    trim_messages_threshold: int = Field(
        default=TRIM_MESSAGES_THRESHOLD,
        gt=0,
        description="Minimum messages before applying trimming"
    )

    # Health Monitoring Configuration
    health_check_interval: int = Field(
        default=60,
        gt=0,
        description="Health check interval (seconds)"
    )
    health_min_success_rate: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Minimum success rate percentage"
    )
    health_max_response_time_ms: float = Field(
        default=5000.0,
        gt=0.0,
        description="Maximum response time threshold"
    )

    @field_validator('llm_timeout')
    @classmethod
    def validate_llm_timeout(cls, v: int) -> int:
        """Validate LLM timeout is reasonable."""
        if v <= 0:
            raise ValueError('LLM timeout must be positive')
        if v > 600:  # 10 minutes max
            raise ValueError('LLM timeout cannot exceed 600 seconds')
        return v

    @field_validator('llm_max_concurrent')
    @classmethod
    def validate_max_concurrent(cls, v: int) -> int:
        """Validate max concurrent requests."""
        if v <= 0:
            raise ValueError('Max concurrent requests must be positive')
        if v > 100:
            raise ValueError('Max concurrent requests cannot exceed 100')
        return v

    @field_validator('llm_context_window')
    @classmethod
    def validate_context_window(cls, v: int) -> int:
        """Validate context window size."""
        if v <= 0:
            raise ValueError('Context window must be positive')
        if v > 200000:  # Reasonable upper limit
            raise ValueError('Context window cannot exceed 200,000 tokens')
        return v

    @property
    def effective_context_window(self) -> int:
        """Get effective context window for the configured model."""
        model_tokens = MAX_TOKENS_TABLE.get(
            self.llm_model.lower(),
            DEFAULT_MAX_TOKEN_LENGTH
        )
        return min(self.llm_context_window, model_tokens)

    @property
    def max_response_tokens(self) -> int:
        """Get maximum tokens available for response generation."""
        return self.effective_context_window - self.llm_reserve_tokens

    def validate_configuration(self) -> None:
        """Validate LLM configuration consistency."""
        # Ensure reserved tokens don't exceed context window
        if self.llm_reserve_tokens >= self.effective_context_window:
            raise ValueError(
                f"Reserved tokens ({self.llm_reserve_tokens}) cannot equal or exceed "
                f"effective context window ({self.effective_context_window})"
            )

        # Validate temperature range
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM temperature must be between 0.0 and 2.0")

    def get_config_summary(self) -> dict:
        """Get configuration summary (excluding sensitive data)."""
        return {
            "config_type": "LLMConfig",
            "model": self.llm_model,
            "base_url": self.llm_base_url.split("://")[0],  # Only show protocol
            "timeout_seconds": self.llm_timeout,
            "temperature": self.llm_temperature,
            "max_concurrent": self.llm_max_concurrent,
            "context_window": self.effective_context_window,
            "max_response_tokens": self.max_response_tokens,
            "truncation_strategy": self.context_truncation_strategy
        }


class ChatConfig(BaseConfig):
    """Configuration for chat-specific functionality."""

    # Session Management
    turns_window: int = Field(
        default=10,
        gt=0,
        le=50,
        description="Number of conversation turns to keep in memory"
    )
    max_session_age_hours: int = Field(
        default=24,
        gt=0,
        le=168,  # 1 week max
        description="Maximum age before sessions expire"
    )

    # Response Configuration
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        gt=0,
        le=128,
        description="Batch size for processing"
    )
    max_retries: int = Field(
        default=MAX_RETRY_ATTEMPTS,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

    def validate_configuration(self) -> None:
        """Validate chat configuration consistency."""
        if self.turns_window <= 0:
            raise ValueError("Turns window must be positive")

        if self.max_session_age_hours <= 0:
            raise ValueError("Max session age must be positive")

    def get_config_summary(self) -> dict:
        """Get configuration summary."""
        return {
            "config_type": "ChatConfig",
            "turns_window": self.turns_window,
            "max_session_age_hours": self.max_session_age_hours,
            "streaming_enabled": self.enable_streaming,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries
        }