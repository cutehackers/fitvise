"""Model information entity."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelSpec:
    """Information about an LLM model."""

    name: str
    provider: str
    max_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    temperature_range: tuple[float, float] = (0.0, 2.0)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate model info after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")

        if not self.provider or not self.provider.strip():
            raise ValueError("Provider name cannot be empty")

        if self.temperature_range[0] < 0 or self.temperature_range[1] < 0:
            raise ValueError("Temperature values must be non-negative")

        if self.temperature_range[0] > self.temperature_range[1]:
            raise ValueError("Min temperature cannot be greater than max temperature")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
            "temperature_range": self.temperature_range,
            "metadata": self.metadata or {},
        }