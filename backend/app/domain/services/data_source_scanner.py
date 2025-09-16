"""Data source scanning and discovery service."""
import os
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from urllib.parse import urlparse

from app.domain.entities.data_source import DataSource
from app.domain.value_objects.source_info import (
    SourceInfo, 
    SourceType, 
    ConnectionConfig, 
    AuthConfig, 
    AuthType,
    AccessFrequency
)
from app.domain.value_objects.document_metadata import DocumentFormat


class DataSourceScanner:
    """Discovers & catalogs data sources from filesystems, DBs & APIs with export capabilities.
    
    Examples:
        >>> scanner = DataSourceScanner()
        >>> sources = await scanner.discover_file_system_sources(["/docs", "/reports"])
        >>> len(sources) >= 5  # Found document collections
        True
        >>> data_source = await scanner.create_data_source_from_discovery(sources[0])
        >>> await scanner.export_inventory_to_csv([data_source], "inventory.csv")
    """
    
    def __init__(self):
        """Initialize the scanner."""
        # File extensions to formats mapping
        self.file_format_mapping = {
            '.pdf': DocumentFormat.PDF,
            '.docx': DocumentFormat.DOCX,
            '.doc': DocumentFormat.DOCX,
            '.html': DocumentFormat.HTML,
            '.htm': DocumentFormat.HTML,
            '.txt': DocumentFormat.TXT,
            '.md': DocumentFormat.MD,
            '.csv': DocumentFormat.CSV,
            '.json': DocumentFormat.JSON,
            '.xml': DocumentFormat.XML
        }
        
        # Common document directories
        self.common_doc_dirs = [
            'documents', 'docs', 'files', 'data', 'content',
            'reports', 'manuals', 'guides', 'resources'
        ]
        
        # Patterns to exclude
        self.exclude_patterns = {
            '.git', '.svn', '__pycache__', 'node_modules',
            '.DS_Store', 'Thumbs.db', '.tmp', '.temp'
        }
    
    async def discover_file_system_sources(
        self, 
        root_paths: List[str], 
        max_depth: int = 5,
        min_file_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Discover file system data sources."""
        discovered_sources = []
        
        for root_path in root_paths:
            if not os.path.exists(root_path):
                continue
            
            try:
                sources = await self._scan_directory_tree(
                    root_path, max_depth, min_file_count
                )
                discovered_sources.extend(sources)
            except Exception as e:
                # Log error but continue with other paths
                print(f"Error scanning {root_path}: {e}")
        
        return discovered_sources
    
    async def _scan_directory_tree(
        self, 
        root_path: str, 
        max_depth: int,
        min_file_count: int
    ) -> List[Dict[str, Any]]:
        """Scan a directory tree for document collections."""
        sources = []
        
        for current_root, dirs, files in os.walk(root_path):
            # Calculate current depth
            depth = current_root[len(root_path):].count(os.sep)
            if depth >= max_depth:
                dirs.clear()  # Don't go deeper
                continue
            
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            
            # Analyze current directory
            source_info = await self._analyze_directory(current_root, files)
            
            if source_info and source_info['file_count'] >= min_file_count:
                sources.append(source_info)
        
        return sources
    
    async def _analyze_directory(
        self, 
        directory_path: str, 
        files: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a directory for data source characteristics."""
        # Filter relevant files
        document_files = [
            f for f in files 
            if self._is_document_file(f) and not self._should_exclude(f)
        ]
        
        if not document_files:
            return None
        
        # Analyze file types
        format_counts = {}
        total_size = 0
        
        for file_name in document_files:
            file_path = os.path.join(directory_path, file_name)
            
            try:
                stat_info = os.stat(file_path)
                total_size += stat_info.st_size
                
                # Determine format
                ext = Path(file_name).suffix.lower()
                format_type = self.file_format_mapping.get(ext, DocumentFormat.TXT)
                format_counts[format_type] = format_counts.get(format_type, 0) + 1
                
            except (OSError, IOError):
                continue
        
        # Calculate metrics
        avg_file_size = total_size / len(document_files) if document_files else 0
        
        # Determine source characteristics
        directory_name = os.path.basename(directory_path)
        is_common_doc_dir = any(
            common in directory_name.lower() 
            for common in self.common_doc_dirs
        )
        
        return {
            'name': f"FileSystem: {directory_name}",
            'description': f"Document collection in {directory_path}",
            'path': directory_path,
            'file_count': len(document_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'avg_file_size_kb': round(avg_file_size / 1024, 2),
            'format_distribution': format_counts,
            'is_common_doc_directory': is_common_doc_dir,
            'priority': 8 if is_common_doc_dir else 5
        }
    
    def _is_document_file(self, filename: str) -> bool:
        """Check if file is a supported document type."""
        ext = Path(filename).suffix.lower()
        return ext in self.file_format_mapping
    
    def _should_exclude(self, name: str) -> bool:
        """Check if file/directory should be excluded."""
        return (
            name.startswith('.') or 
            name in self.exclude_patterns or
            any(pattern in name.lower() for pattern in self.exclude_patterns)
        )
    
    async def discover_database_sources(
        self, 
        connection_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Discover database data sources."""
        discovered_sources = []
        
        for config in connection_configs:
            try:
                source_info = await self._analyze_database_connection(config)
                if source_info:
                    discovered_sources.append(source_info)
            except Exception as e:
                print(f"Error analyzing database {config.get('name', 'unknown')}: {e}")
        
        return discovered_sources
    
    async def _analyze_database_connection(
        self, 
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a database connection for content."""
        # This would require actual database connections
        # For now, return basic structure based on config
        
        db_type = config.get('type', 'unknown')
        host = config.get('host', 'localhost')
        database_name = config.get('database')
        
        # Estimate table count and row counts
        # In real implementation, this would query information_schema
        estimated_tables = config.get('estimated_tables', 10)
        estimated_rows = config.get('estimated_rows', 1000)
        
        return {
            'name': f"Database: {database_name or host}",
            'description': f"{db_type.upper()} database on {host}",
            'db_type': db_type,
            'host': host,
            'database': database_name,
            'estimated_tables': estimated_tables,
            'estimated_rows': estimated_rows,
            'connection_config': config,
            'priority': 7
        }
    
    async def discover_web_api_sources(
        self, 
        api_endpoints: List[str]
    ) -> List[Dict[str, Any]]:
        """Discover web API data sources."""
        discovered_sources = []
        
        for endpoint in api_endpoints:
            try:
                source_info = await self._analyze_api_endpoint(endpoint)
                if source_info:
                    discovered_sources.append(source_info)
            except Exception as e:
                print(f"Error analyzing API endpoint {endpoint}: {e}")
        
        return discovered_sources
    
    async def _analyze_api_endpoint(
        self, 
        endpoint: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze an API endpoint for content characteristics."""
        parsed_url = urlparse(endpoint)
        
        # Basic API analysis
        # In real implementation, this would make HEAD/OPTIONS requests
        
        return {
            'name': f"API: {parsed_url.netloc}",
            'description': f"Web API endpoint: {endpoint}",
            'base_url': f"{parsed_url.scheme}://{parsed_url.netloc}",
            'endpoint': endpoint,
            'requires_auth': True,  # Assume auth required for safety
            'estimated_records': 100,  # Default estimate
            'priority': 6
        }
    
    async def create_data_source_from_discovery(
        self, 
        discovery_info: Dict[str, Any]
    ) -> DataSource:
        """Create a DataSource entity from discovery information."""
        
        # Determine source type
        if 'path' in discovery_info:
            source_type = SourceType.FILE_SYSTEM
            connection_config = ConnectionConfig(
                custom_params={
                    'base_path': discovery_info['path'],
                    'file_patterns': list(discovery_info.get('format_distribution', {}).keys())
                }
            )
        elif 'db_type' in discovery_info:
            source_type = SourceType.DATABASE
            connection_config = ConnectionConfig(
                host=discovery_info.get('host'),
                database_name=discovery_info.get('database'),
                custom_params=discovery_info.get('connection_config', {})
            )
        elif 'base_url' in discovery_info:
            source_type = SourceType.WEB_API
            connection_config = ConnectionConfig(
                base_url=discovery_info['base_url'],
                endpoint=discovery_info.get('endpoint')
            )
        else:
            raise ValueError("Unknown discovery info format")
        
        # Create auth config
        auth_type = AuthType.NONE
        if discovery_info.get('requires_auth', False):
            auth_type = AuthType.API_KEY  # Default to API key
        
        auth_config = AuthConfig(auth_type=auth_type)
        
        # Create source info
        source_info = SourceInfo(
            name=discovery_info['name'],
            source_type=source_type,
            description=discovery_info['description'],
            connection_config=connection_config,
            auth_config=auth_config,
            access_frequency=AccessFrequency.DAILY,
            is_active=True,
            priority=discovery_info.get('priority', 5),
            estimated_document_count=discovery_info.get(
                'file_count', 
                discovery_info.get('estimated_records', 100)
            )
        )
        
        # Create data source
        data_source = DataSource(
            name=discovery_info['name'],
            source_type=source_type,
            description=discovery_info['description'],
            source_info=source_info
        )
        
        return data_source
    
    async def export_inventory_to_csv(
        self, 
        data_sources: List[DataSource],
        file_path: str
    ) -> None:
        """Export data source inventory to CSV file."""
        import csv
        import io
        
        headers = [
            'ID', 'Name', 'Type', 'Description', 'Is Active', 'Priority',
            'Estimated Documents', 'Last Scan', 'Quality Score', 'Health Status',
            'Connection Info', 'Created At'
        ]
        
        # Create CSV content in memory first
        csv_content = io.StringIO()
        writer = csv.writer(csv_content)
        writer.writerow(headers)
        
        for ds in data_sources:
            health = ds.get_health_status()
            connection_info = f"{ds.source_info.connection_config.host or 'N/A'}"
            if ds.source_info.connection_config.database_name:
                connection_info += f"/{ds.source_info.connection_config.database_name}"
            elif ds.source_info.connection_config.base_url:
                connection_info = ds.source_info.connection_config.base_url
            
            row = [
                str(ds.id),
                ds.name,
                ds.source_type.value,
                ds.description,
                ds.is_active,
                ds.source_info.priority,
                ds.source_info.estimated_document_count,
                ds.last_scan_time.isoformat() if ds.last_scan_time else 'Never',
                ds.current_quality_score,
                'Healthy' if health['is_healthy'] else 'Unhealthy',
                connection_info,
                ds.created_at.isoformat()
            ]
            writer.writerow(row)
        
        # Write to file asynchronously
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(csv_content.getvalue())
    
    async def export_inventory_to_json(
        self, 
        data_sources: List[DataSource],
        file_path: str
    ) -> None:
        """Export data source inventory to JSON file."""
        import json
        
        inventory_data = {
            'exported_at': datetime.utcnow().isoformat(),
            'total_sources': len(data_sources),
            'sources': []
        }
        
        for ds in data_sources:
            health = ds.get_health_status()
            
            source_data = {
                'id': str(ds.id),
                'name': ds.name,
                'type': ds.source_type.value,
                'description': ds.description,
                'is_active': ds.is_active,
                'priority': ds.source_info.priority,
                'estimated_documents': ds.source_info.estimated_document_count,
                'last_scan': ds.last_scan_time.isoformat() if ds.last_scan_time else None,
                'total_processed': ds.total_documents_processed,
                'quality_score': ds.current_quality_score,
                'health_status': health,
                'connection_config': {
                    'host': ds.source_info.connection_config.host,
                    'database': ds.source_info.connection_config.database_name,
                    'base_url': ds.source_info.connection_config.base_url,
                    'endpoint': ds.source_info.connection_config.endpoint
                },
                'auth_type': ds.source_info.auth_config.auth_type.value,
                'access_frequency': ds.source_info.access_frequency.value,
                'created_at': ds.created_at.isoformat(),
                'updated_at': ds.updated_at.isoformat()
            }
            
            inventory_data['sources'].append(source_data)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as jsonfile:
            await jsonfile.write(json.dumps(inventory_data, indent=2, ensure_ascii=False))
    
    def get_discovery_statistics(
        self, 
        discovered_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about discovered sources."""
        if not discovered_sources:
            return {
                'total_sources': 0,
                'types': {},
                'total_estimated_documents': 0,
                'avg_priority': 0
            }
        
        type_counts = {}
        total_docs = 0
        total_priority = 0
        
        for source in discovered_sources:
            # Determine type from source info
            if 'path' in source:
                source_type = 'file_system'
                doc_count = source.get('file_count', 0)
            elif 'db_type' in source:
                source_type = 'database'
                doc_count = source.get('estimated_rows', 0)
            elif 'base_url' in source:
                source_type = 'web_api'
                doc_count = source.get('estimated_records', 0)
            else:
                source_type = 'unknown'
                doc_count = 0
            
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
            total_docs += doc_count
            total_priority += source.get('priority', 5)
        
        return {
            'total_sources': len(discovered_sources),
            'types': type_counts,
            'total_estimated_documents': total_docs,
            'avg_priority': round(total_priority / len(discovered_sources), 1)
        }