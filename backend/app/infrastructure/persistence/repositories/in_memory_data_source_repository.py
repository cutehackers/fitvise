"""In-memory implementation of data source repository for development/testing."""
from typing import List, Optional, Dict
from uuid import UUID

from app.domain.entities.data_source import DataSource
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.value_objects.source_info import SourceType


class InMemoryDataSourceRepository(DataSourceRepository):
    """In-memory implementation of data source repository."""
    
    def __init__(self):
        """Initialize the repository."""
        self._data_sources: Dict[UUID, DataSource] = {}
        self._name_index: Dict[str, UUID] = {}
    
    async def save(self, data_source: DataSource) -> DataSource:
        """Save a data source."""
        # Update name index if name changed
        if data_source.id in self._data_sources:
            old_source = self._data_sources[data_source.id]
            if old_source.name != data_source.name:
                # Remove old name from index
                if old_source.name in self._name_index:
                    del self._name_index[old_source.name]
        
        # Store data source and update name index
        self._data_sources[data_source.id] = data_source
        self._name_index[data_source.name] = data_source.id
        
        return data_source
    
    async def find_by_id(self, data_source_id: UUID) -> Optional[DataSource]:
        """Find a data source by ID."""
        return self._data_sources.get(data_source_id)
    
    async def find_by_name(self, name: str) -> Optional[DataSource]:
        """Find a data source by name."""
        source_id = self._name_index.get(name)
        if source_id:
            return self._data_sources.get(source_id)
        return None
    
    async def find_all(self) -> List[DataSource]:
        """Find all data sources."""
        return list(self._data_sources.values())
    
    async def find_by_type(self, source_type: SourceType) -> List[DataSource]:
        """Find data sources by type."""
        return [
            ds for ds in self._data_sources.values() 
            if ds.source_type == source_type
        ]
    
    async def find_active(self) -> List[DataSource]:
        """Find all active data sources."""
        return [
            ds for ds in self._data_sources.values() 
            if ds.is_active
        ]
    
    async def find_inactive(self) -> List[DataSource]:
        """Find all inactive data sources."""
        return [
            ds for ds in self._data_sources.values() 
            if not ds.is_active
        ]
    
    async def find_needing_scan(self) -> List[DataSource]:
        """Find data sources that need scanning."""
        return [
            ds for ds in self._data_sources.values() 
            if ds.needs_scan()
        ]
    
    async def find_unhealthy(self) -> List[DataSource]:
        """Find unhealthy data sources."""
        return [
            ds for ds in self._data_sources.values() 
            if not ds.is_healthy()
        ]
    
    async def delete(self, data_source_id: UUID) -> bool:
        """Delete a data source."""
        if data_source_id in self._data_sources:
            data_source = self._data_sources[data_source_id]
            # Remove from name index
            if data_source.name in self._name_index:
                del self._name_index[data_source.name]
            # Remove from main storage
            del self._data_sources[data_source_id]
            return True
        return False
    
    async def exists_by_name(self, name: str) -> bool:
        """Check if a data source with the given name exists."""
        return name in self._name_index
    
    async def count_all(self) -> int:
        """Count total number of data sources."""
        return len(self._data_sources)
    
    async def count_by_type(self, source_type: SourceType) -> int:
        """Count data sources by type."""
        return len([
            ds for ds in self._data_sources.values() 
            if ds.source_type == source_type
        ])
    
    async def count_active(self) -> int:
        """Count active data sources."""
        return len([
            ds for ds in self._data_sources.values() 
            if ds.is_active
        ])