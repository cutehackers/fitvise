"""Data source domain entity for RAG system."""
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from app.domain.value_objects.source_info import SourceInfo, SourceType
from app.domain.value_objects.quality_metrics import DataQualityMetrics


class DataSource:
    """Data source entity - tracks files/APIs/DBs with health & processing metrics for RAG pipeline.
    
    Examples:
        >>> from app.domain.value_objects.source_info import SourceInfo, SourceType, ConnectionConfig, AuthConfig
        >>> source_info = SourceInfo("GitHub API", SourceType.WEB_API, "GitHub repos", 
        ...                          ConnectionConfig(base_url="https://api.github.com"), 
        ...                          AuthConfig())
        >>> ds = DataSource("github-api", SourceType.WEB_API, "GitHub API source", source_info)
        >>> ds.is_healthy()  # Check if source is operational
        True
        >>> ds.needs_scan()  # Check if data needs refreshing
        False
    """
    
    def __init__(
        self,
        name: str,
        source_type: SourceType,
        description: str,
        source_info: SourceInfo,
        id: Optional[UUID] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        is_active: bool = True
    ):
        """Initialize a data source entity."""
        self._id = id or uuid4()
        self._name = name
        self._source_type = source_type
        self._description = description
        self._source_info = source_info
        self._created_at = created_at or datetime.utcnow()
        self._updated_at = updated_at or datetime.utcnow()
        self._is_active = is_active
        
        # Processing history
        self._last_scan_time: Optional[datetime] = None
        self._last_scan_document_count: int = 0
        self._total_documents_processed: int = 0
        self._processing_errors: List[str] = []
        
        # Quality tracking
        self._quality_history: List[DataQualityMetrics] = []
        self._current_quality_score: Optional[float] = None
        
        # Validation
        self._validate()
    
    def _validate(self) -> None:
        """Validate the data source entity."""
        if not self._name.strip():
            raise ValueError("Data source name cannot be empty")
        
        if not self._description.strip():
            raise ValueError("Data source description cannot be empty")
        
        if len(self._name) > 255:
            raise ValueError("Data source name cannot exceed 255 characters")
        
        if len(self._description) > 1000:
            raise ValueError("Data source description cannot exceed 1000 characters")
    
    # Properties (read-only)
    @property
    def id(self) -> UUID:
        """Get the unique identifier."""
        return self._id
    
    @property
    def name(self) -> str:
        """Get the data source name."""
        return self._name
    
    @property
    def source_type(self) -> SourceType:
        """Get the source type."""
        return self._source_type
    
    @property
    def description(self) -> str:
        """Get the description."""
        return self._description
    
    @property
    def source_info(self) -> SourceInfo:
        """Get the source information."""
        return self._source_info
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def is_active(self) -> bool:
        """Check if data source is active."""
        return self._is_active
    
    @property
    def last_scan_time(self) -> Optional[datetime]:
        """Get last scan timestamp."""
        return self._last_scan_time
    
    @property
    def last_scan_document_count(self) -> int:
        """Get document count from last scan."""
        return self._last_scan_document_count
    
    @property
    def total_documents_processed(self) -> int:
        """Get total number of documents processed."""
        return self._total_documents_processed
    
    @property
    def processing_errors(self) -> List[str]:
        """Get list of processing errors."""
        return self._processing_errors.copy()
    
    @property
    def quality_history(self) -> List[DataQualityMetrics]:
        """Get quality metrics history."""
        return self._quality_history.copy()
    
    @property
    def current_quality_score(self) -> Optional[float]:
        """Get current quality score."""
        return self._current_quality_score
    
    # Business methods
    def update_name(self, name: str) -> None:
        """Update the data source name."""
        if not name.strip():
            raise ValueError("Data source name cannot be empty")
        if len(name) > 255:
            raise ValueError("Data source name cannot exceed 255 characters")
        
        self._name = name
        self._updated_at = datetime.utcnow()
    
    def update_description(self, description: str) -> None:
        """Update the data source description."""
        if not description.strip():
            raise ValueError("Data source description cannot be empty")
        if len(description) > 1000:
            raise ValueError("Data source description cannot exceed 1000 characters")
        
        self._description = description
        self._updated_at = datetime.utcnow()
    
    def update_source_info(self, source_info: SourceInfo) -> None:
        """Update the source information."""
        self._source_info = source_info
        self._updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activate the data source."""
        if not self._is_active:
            self._is_active = True
            self._updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the data source."""
        if self._is_active:
            self._is_active = False
            self._updated_at = datetime.utcnow()
    
    def record_scan_result(self, document_count: int, errors: List[str] = None) -> None:
        """Record the result of a data source scan."""
        self._last_scan_time = datetime.utcnow()
        self._last_scan_document_count = document_count
        self._total_documents_processed += document_count
        
        if errors:
            self._processing_errors.extend(errors)
            # Keep only the last 100 errors
            if len(self._processing_errors) > 100:
                self._processing_errors = self._processing_errors[-100:]
        
        self._updated_at = datetime.utcnow()
    
    def add_quality_metrics(self, metrics: DataQualityMetrics) -> None:
        """Add quality metrics to history."""
        self._quality_history.append(metrics)
        self._current_quality_score = metrics.overall_quality_score
        
        # Keep only the last 50 quality measurements
        if len(self._quality_history) > 50:
            self._quality_history = self._quality_history[-50:]
        
        self._updated_at = datetime.utcnow()
    
    def clear_processing_errors(self) -> None:
        """Clear all processing errors."""
        self._processing_errors.clear()
        self._updated_at = datetime.utcnow()
    
    # Status methods
    def is_healthy(self) -> bool:
        """Check if the data source is healthy."""
        if not self._is_active:
            return False
        
        # Check source info health
        if not self._source_info.is_healthy:
            return False
        
        # Check recent quality score
        if self._current_quality_score is not None and self._current_quality_score < 0.3:
            return False
        
        # Check for too many recent errors
        if len(self._processing_errors) > 20:
            return False
        
        return True
    
    def needs_scan(self) -> bool:
        """Check if the data source needs to be scanned."""
        if not self._is_active:
            return False
        
        if self._last_scan_time is None:
            return True
        
        # Check based on access frequency
        from app.domain.value_objects.source_info import AccessFrequency
        
        now = datetime.utcnow()
        time_since_scan = now - self._last_scan_time
        
        frequency_map = {
            AccessFrequency.REAL_TIME: 300,     # 5 minutes
            AccessFrequency.HOURLY: 3600,       # 1 hour
            AccessFrequency.DAILY: 86400,       # 24 hours
            AccessFrequency.WEEKLY: 604800,     # 7 days
            AccessFrequency.MONTHLY: 2592000,   # 30 days
            AccessFrequency.ON_DEMAND: float('inf')  # Never auto-scan
        }
        
        required_interval = frequency_map.get(self._source_info.access_frequency, 86400)
        return time_since_scan.total_seconds() >= required_interval
    
    def get_health_status(self) -> dict:
        """Get comprehensive health status."""
        return {
            "is_healthy": self.is_healthy(),
            "is_active": self._is_active,
            "source_healthy": self._source_info.is_healthy,
            "needs_scan": self.needs_scan(),
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "total_documents": self._total_documents_processed,
            "error_count": len(self._processing_errors),
            "quality_score": self._current_quality_score,
            "quality_measurements": len(self._quality_history)
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"DataSource(id={self._id}, name='{self._name}', type={self._source_type.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"DataSource(id={self._id}, name='{self._name}', "
                f"type={self._source_type.value}, active={self._is_active})")