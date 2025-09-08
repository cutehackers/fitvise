"""Data source repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from app.domain.entities.data_source import DataSource
from app.domain.value_objects.source_info import SourceType


class DataSourceRepository(ABC):
    """Abstract repository for data source persistence."""
    
    @abstractmethod
    async def save(self, data_source: DataSource) -> DataSource:
        """Save a data source."""
        pass
    
    @abstractmethod
    async def find_by_id(self, data_source_id: UUID) -> Optional[DataSource]:
        """Find a data source by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DataSource]:
        """Find a data source by name."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[DataSource]:
        """Find all data sources."""
        pass
    
    @abstractmethod
    async def find_by_type(self, source_type: SourceType) -> List[DataSource]:
        """Find data sources by type."""
        pass
    
    @abstractmethod
    async def find_active(self) -> List[DataSource]:
        """Find all active data sources."""
        pass
    
    @abstractmethod
    async def find_inactive(self) -> List[DataSource]:
        """Find all inactive data sources."""
        pass
    
    @abstractmethod
    async def find_needing_scan(self) -> List[DataSource]:
        """Find data sources that need scanning."""
        pass
    
    @abstractmethod
    async def find_unhealthy(self) -> List[DataSource]:
        """Find unhealthy data sources."""
        pass
    
    @abstractmethod
    async def delete(self, data_source_id: UUID) -> bool:
        """Delete a data source."""
        pass
    
    @abstractmethod
    async def exists_by_name(self, name: str) -> bool:
        """Check if a data source with the given name exists."""
        pass
    
    @abstractmethod
    async def count_all(self) -> int:
        """Count total number of data sources."""
        pass
    
    @abstractmethod
    async def count_by_type(self, source_type: SourceType) -> int:
        """Count data sources by type."""
        pass
    
    @abstractmethod
    async def count_active(self) -> int:
        """Count active data sources."""
        pass