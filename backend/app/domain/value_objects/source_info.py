"""Source information value object for RAG system."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class SourceType(str, Enum):
    """Data source types - filesystem, DB, API, scraper, cloud, email, SharePoint, etc."""
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    WEB_API = "web_api"
    WEB_SCRAPER = "web_scraper"
    CLOUD_STORAGE = "cloud_storage"
    EMAIL = "email"
    SHAREPOINT = "sharepoint"
    CONFLUENCE = "confluence"
    SLACK = "slack"
    GITHUB = "github"
    NOTION = "notion"
    GOOGLE_DRIVE = "google_drive"


class AuthType(str, Enum):
    """Auth methods - none, API key, bearer token, basic auth, OAuth2, JWT, cert."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CERTIFICATE = "certificate"


class AccessFrequency(str, Enum):
    """How frequently to access the source."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


@dataclass(frozen=True)
class ConnectionConfig:
    """Connection configuration for external sources."""
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    base_url: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    ssl_verify: bool = True
    
    # Connection pooling
    pool_size: int = 5
    max_overflow: int = 10
    
    # Custom parameters
    custom_params: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.custom_params is None:
            object.__setattr__(self, 'custom_params', {})


@dataclass(frozen=True)
class AuthConfig:
    """Authentication configuration."""
    auth_type: AuthType = AuthType.NONE
    username: Optional[str] = None
    password: Optional[str] = None  # Should be encrypted/hashed
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_url: Optional[str] = None
    scopes: List[str] = None
    
    # Certificate authentication
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.scopes is None:
            object.__setattr__(self, 'scopes', [])


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: Optional[float] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    burst_size: Optional[int] = None
    
    # Retry configuration
    backoff_factor: float = 1.0
    max_backoff_seconds: float = 60.0


@dataclass(frozen=True)
class SourceInfo:
    """Data source config - connection, auth, rate limits, health & access patterns.
    
    Examples:
        >>> conn_config = ConnectionConfig(base_url="https://api.slack.com")
        >>> auth_config = AuthConfig(auth_type=AuthType.BEARER_TOKEN)
        >>> rate_config = RateLimitConfig(requests_per_minute=100)
        >>> source = SourceInfo("Slack API", SourceType.WEB_API, "Team messages",
        ...                     conn_config, auth_config, rate_config)
        >>> source.is_healthy
        True
    """
    
    # Basic identification
    name: str
    source_type: SourceType
    description: str
    
    # Connection details
    connection_config: ConnectionConfig
    auth_config: AuthConfig
    rate_limit_config: Optional[RateLimitConfig] = None
    
    # Access patterns
    access_frequency: AccessFrequency = AccessFrequency.DAILY
    last_accessed: Optional[datetime] = None
    next_scheduled_access: Optional[datetime] = None
    
    # Content filtering
    file_patterns: List[str] = None  # e.g., ["*.pdf", "*.docx"]
    exclude_patterns: List[str] = None  # e.g., ["*temp*", "*.tmp"]
    max_file_size_mb: Optional[int] = None
    
    # Health monitoring
    is_active: bool = True
    health_check_url: Optional[str] = None
    last_health_check: Optional[datetime] = None
    health_status: Optional[str] = None
    
    # Metadata
    tags: List[str] = None
    priority: int = 0  # 0 = lowest, 10 = highest
    estimated_document_count: Optional[int] = None
    
    # Quality metrics
    success_rate: Optional[float] = None  # 0.0 to 1.0
    average_response_time: Optional[float] = None  # in seconds
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.file_patterns is None:
            object.__setattr__(self, 'file_patterns', [])
        if self.exclude_patterns is None:
            object.__setattr__(self, 'exclude_patterns', [])
        if self.tags is None:
            object.__setattr__(self, 'tags', [])
    
    @property
    def is_healthy(self) -> bool:
        """Check if source is healthy based on recent checks."""
        if not self.is_active:
            return False
        if self.health_status:
            return self.health_status.lower() == "healthy"
        return True
    
    @property
    def needs_health_check(self) -> bool:
        """Check if source needs a health check."""
        if not self.health_check_url:
            return False
        if not self.last_health_check:
            return True
        
        # Check if last health check was more than an hour ago
        time_since_check = datetime.utcnow() - self.last_health_check
        return time_since_check.total_seconds() > 3600
    
    @property
    def has_auth(self) -> bool:
        """Check if source requires authentication."""
        return self.auth_config.auth_type != AuthType.NONE
    
    def with_health_status(self, status: str, error: Optional[str] = None) -> 'SourceInfo':
        """Create new source info with updated health status."""
        new_error_count = self.error_count + 1 if error else self.error_count
        
        return SourceInfo(
            name=self.name,
            source_type=self.source_type,
            description=self.description,
            connection_config=self.connection_config,
            auth_config=self.auth_config,
            rate_limit_config=self.rate_limit_config,
            access_frequency=self.access_frequency,
            last_accessed=self.last_accessed,
            next_scheduled_access=self.next_scheduled_access,
            file_patterns=self.file_patterns.copy(),
            exclude_patterns=self.exclude_patterns.copy(),
            max_file_size_mb=self.max_file_size_mb,
            is_active=self.is_active,
            health_check_url=self.health_check_url,
            last_health_check=datetime.utcnow(),
            health_status=status,
            tags=self.tags.copy(),
            priority=self.priority,
            estimated_document_count=self.estimated_document_count,
            success_rate=self.success_rate,
            average_response_time=self.average_response_time,
            error_count=new_error_count,
            last_error=error,
            last_error_time=datetime.utcnow() if error else self.last_error_time
        )