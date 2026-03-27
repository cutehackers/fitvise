"""Base configuration classes.

Provides foundation for modular configuration using composition pattern
to follow Single Responsibility Principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel


class BaseConfig(BaseModel, ABC):
    """Base configuration class with common functionality."""

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"

    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging.

        Returns:
            Dictionary with key configuration values (exclude sensitive data)
        """
        return {
            "config_type": self.__class__.__name__,
            "validation_status": "valid"
        }


class MixinConfig(BaseModel):
    """Mixin for reusable configuration patterns."""

    def merge_with(self, other: "MixinConfig") -> "MixinConfig":
        """Merge with another configuration instance."""
        # Create a new instance with merged data
        merged_data = {**self.model_dump(), **other.model_dump()}
        return self.__class__(**merged_data)

    def override_from_env(self, env_prefix: str = "") -> None:
        """Override values from environment variables.

        Args:
            env_prefix: Prefix for environment variables
        """
        import os

        for field_name, field_info in self.model_fields.items():
            env_key = f"{env_prefix}{field_name.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert string to appropriate type
                setattr(self, field_name, self._convert_env_value(env_value, field_info))

    def _convert_env_value(self, value: str, field_info: Any) -> Any:
        """Convert environment variable value to appropriate type."""
        # Simple conversion logic - can be extended
        if field_info.annotation == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif field_info.annotation == int:
            return int(value)
        elif field_info.annotation == float:
            return float(value)
        elif field_info.annotation == list:
            return [item.strip() for item in value.split(",")]
        else:
            return value