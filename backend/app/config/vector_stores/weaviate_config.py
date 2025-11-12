"""Weaviate vector database configuration for Epic 2.3.

This module defines configuration classes for Weaviate vector database,
supporting connection management, schema settings, timeout policies,
and retry strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class WeaviateAuthType(str, Enum):
    """Weaviate authentication types."""

    NONE = "none"  # No authentication
    API_KEY = "api_key"  # API key authentication
    OIDC = "oidc"  # OpenID Connect authentication


class ConsistencyLevel(str, Enum):
    """Weaviate consistency levels."""

    ONE = "ONE"  # Single replica must acknowledge
    QUORUM = "QUORUM"  # Majority of replicas must acknowledge
    ALL = "ALL"  # All replicas must acknowledge


@dataclass
class WeaviateConfig:
    """Configuration for Weaviate vector database (Epic 2.3).

    Supports comprehensive configuration for connection management,
    authentication, timeouts, retries, and schema settings.

    Attributes:
        host: Weaviate server host
        port: Weaviate server port
        grpc_port: gRPC port for Weaviate v4+
        scheme: Connection scheme (http/https)
        auth_type: Authentication type
        api_key: API key for authentication
        timeout: Request timeout in seconds
        connection_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        startup_period: Startup wait period in seconds
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        consistency_level: Write consistency level
        additional_headers: Additional HTTP headers
        grpc_secure: Use secure gRPC connection

    Examples:
        >>> config = WeaviateConfig.default()
        >>> config.host
        'localhost'
        >>> config.port
        8080

        >>> prod_config = WeaviateConfig.for_production(
        ...     host="weaviate.example.com",
        ...     api_key="secret-key"
        ... )
        >>> prod_config.scheme
        'https'

        >>> local_config = WeaviateConfig.for_local_development()
        >>> local_config.auth_type
        <WeaviateAuthType.NONE: 'none'>
    """

    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    scheme: str = "http"
    auth_type: WeaviateAuthType = WeaviateAuthType.NONE
    api_key: Optional[str] = None
    timeout: float = 30.0
    connection_timeout: float = 10.0
    read_timeout: float = 60.0
    startup_period: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM
    additional_headers: Dict[str, str] = field(default_factory=dict)
    grpc_secure: bool = False

    @classmethod
    def default(cls) -> WeaviateConfig:
        """Create default Weaviate configuration.

        Returns:
            Default configuration for local development.
        """
        return cls()

    @classmethod
    def for_local_development(cls) -> WeaviateConfig:
        """Create configuration for local development.

        Returns:
            Configuration optimized for local Docker Compose setup.
        """
        return cls(
            host="localhost",
            port=8080,
            grpc_port=50051,
            scheme="http",
            auth_type=WeaviateAuthType.NONE,
            timeout=30.0,
            max_retries=3,
            consistency_level=ConsistencyLevel.ONE,
        )

    @classmethod
    def for_production(
        cls,
        host: str,
        api_key: str,
        port: int = 443,
        grpc_port: int = 50051,
    ) -> WeaviateConfig:
        """Create configuration for production deployment.

        Args:
            host: Production Weaviate server hostname
            api_key: API key for authentication
            port: HTTPS port (default 443)
            grpc_port: gRPC port (default 50051)

        Returns:
            Configuration optimized for production with authentication.
        """
        return cls(
            host=host,
            port=port,
            grpc_port=grpc_port,
            scheme="https",
            auth_type=WeaviateAuthType.API_KEY,
            api_key=api_key,
            timeout=60.0,
            connection_timeout=15.0,
            read_timeout=120.0,
            max_retries=5,
            retry_delay=2.0,
            consistency_level=ConsistencyLevel.QUORUM,
            grpc_secure=True,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> WeaviateConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            WeaviateConfig instance
        """
        # Handle URL parameter - parse it into host, port, scheme
        if "url" in config_dict:
            from urllib.parse import urlparse
            parsed_url = urlparse(config_dict["url"])
            if parsed_url.hostname:
                config_dict["host"] = parsed_url.hostname
            if parsed_url.port:
                config_dict["port"] = parsed_url.port
            if parsed_url.scheme:
                config_dict["scheme"] = parsed_url.scheme
            # Remove 'url' from dict as it's not a dataclass field
            del config_dict["url"]

        # Handle timeout_config tuple
        if "timeout_config" in config_dict:
            timeout_tuple = config_dict["timeout_config"]
            if isinstance(timeout_tuple, (tuple, list)) and len(timeout_tuple) >= 2:
                config_dict["connection_timeout"] = float(timeout_tuple[0])
                config_dict["timeout"] = float(timeout_tuple[1])
                del config_dict["timeout_config"]

        # Convert string enums
        if "auth_type" in config_dict and isinstance(config_dict["auth_type"], str):
            config_dict["auth_type"] = WeaviateAuthType(config_dict["auth_type"])
        if "consistency_level" in config_dict and isinstance(
            config_dict["consistency_level"], str
        ):
            config_dict["consistency_level"] = ConsistencyLevel(
                config_dict["consistency_level"]
            )

        return cls(**config_dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "host": self.host,
            "port": self.port,
            "grpc_port": self.grpc_port,
            "scheme": self.scheme,
            "auth_type": self.auth_type.value,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "connection_timeout": self.connection_timeout,
            "read_timeout": self.read_timeout,
            "startup_period": self.startup_period,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "consistency_level": self.consistency_level.value,
            "additional_headers": self.additional_headers,
            "grpc_secure": self.grpc_secure,
        }

    def get_url(self) -> str:
        """Get Weaviate connection URL.

        Returns:
            Full Weaviate connection URL (e.g., http://localhost:8080)
        """
        return f"{self.scheme}://{self.host}:{self.port}"

    def get_grpc_host(self) -> str:
        """Get gRPC connection host.

        Returns:
            gRPC host string (e.g., localhost:50051)
        """
        return f"{self.host}:{self.grpc_port}"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.host:
            raise ValueError("host cannot be empty")
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be 1-65535, got {self.port}")
        if self.grpc_port < 1 or self.grpc_port > 65535:
            raise ValueError(f"grpc_port must be 1-65535, got {self.grpc_port}")
        if self.scheme not in ("http", "https"):
            raise ValueError(f"scheme must be http or https, got {self.scheme}")
        if self.auth_type == WeaviateAuthType.API_KEY and not self.api_key:
            raise ValueError("api_key required when auth_type is API_KEY")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be >0, got {self.timeout}")
        if self.connection_timeout <= 0:
            raise ValueError(
                f"connection_timeout must be >0, got {self.connection_timeout}"
            )
        if self.read_timeout <= 0:
            raise ValueError(f"read_timeout must be >0, got {self.read_timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be ≥0, got {self.max_retries}")
        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be ≥0, got {self.retry_delay}")


def get_weaviate_config(
    config_dict: Optional[Dict[str, Any]] = None,
) -> WeaviateConfig:
    """Get Weaviate configuration.

    Args:
        config_dict: Optional configuration dictionary override

    Returns:
        WeaviateConfig instance
    """
    if config_dict:
        return WeaviateConfig.from_dict(config_dict)
    return WeaviateConfig.for_local_development()
