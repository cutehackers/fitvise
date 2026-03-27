"""Data source audit use case implementation (Task 1.1.1)."""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.domain.entities.data_source import DataSource
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.services.data_source_scanner import DataSourceScanner


class AuditDataSourcesRequest:
    """Audit request - scan paths, DB configs, APIs, export options & save settings."""
    
    def __init__(
        self,
        scan_paths: Optional[List[str]] = None,
        database_configs: Optional[List[Dict[str, Any]]] = None,
        api_endpoints: Optional[List[str]] = None,
        max_scan_depth: int = 5,
        min_file_count: int = 5,
        export_csv_path: Optional[str] = None,
        export_json_path: Optional[str] = None,
        save_to_repository: bool = True
    ):
        """Initialize audit request."""
        # Default scan paths if none provided
        self.scan_paths = scan_paths or self._get_default_scan_paths()
        self.database_configs = database_configs or []
        self.api_endpoints = api_endpoints or []
        self.max_scan_depth = max_scan_depth
        self.min_file_count = min_file_count
        self.export_csv_path = export_csv_path
        self.export_json_path = export_json_path
        self.save_to_repository = save_to_repository
    
    def _get_default_scan_paths(self) -> List[str]:
        """Get default paths to scan for documents."""
        default_paths = []
        
        # User home directory document paths
        home = Path.home()
        common_doc_paths = [
            home / "Documents",
            home / "Downloads", 
            home / "Desktop",
            Path("/Users/Shared/Documents"),  # macOS
            Path("/home/shared/documents"),   # Linux
            Path("C:\\Users\\Public\\Documents")  # Windows
        ]
        
        # Add existing paths
        for path in common_doc_paths:
            if path.exists() and path.is_dir():
                default_paths.append(str(path))
        
        # Current working directory
        cwd = Path.cwd()
        potential_dirs = ['docs', 'documents', 'data', 'files', 'content']
        for dir_name in potential_dirs:
            doc_dir = cwd / dir_name
            if doc_dir.exists() and doc_dir.is_dir():
                default_paths.append(str(doc_dir))
        
        return default_paths


class AuditDataSourcesResponse:
    """Response from data source audit operation."""
    
    def __init__(
        self,
        success: bool,
        discovered_sources: List[Dict[str, Any]],
        created_data_sources: List[DataSource],
        statistics: Dict[str, Any],
        export_files: List[str],
        error_message: Optional[str] = None
    ):
        """Initialize audit response."""
        self.success = success
        self.discovered_sources = discovered_sources
        self.created_data_sources = created_data_sources
        self.statistics = statistics
        self.export_files = export_files
        self.error_message = error_message
    
    @property
    def total_discovered(self) -> int:
        """Get total number of discovered sources."""
        return len(self.discovered_sources)
    
    @property
    def total_created(self) -> int:
        """Get total number of created data sources."""
        return len(self.created_data_sources)


class AuditDataSourcesUseCase:
    """Task 1.1.1 - Audit & catalog data sources, export inventory with ≥20 sources.
    
    Scans filesystems/DBs/APIs, creates DataSource entities, exports CSV/JSON inventory.
    
    Examples:
        >>> use_case = AuditDataSourcesUseCase(repository)
        >>> request = AuditDataSourcesRequest(scan_paths=["/documents"], export_csv="inventory.csv")
        >>> response = await use_case.execute(request)
        >>> response.success and response.total_discovered >= 5
        True
        >>> "inventory.csv" in response.export_files
        True
    """
    
    def __init__(
        self,
        data_source_repository: DataSourceRepository,
        scanner: Optional[DataSourceScanner] = None
    ):
        """Initialize the use case."""
        self.repository = data_source_repository
        self.scanner = scanner or DataSourceScanner()
    
    async def execute(self, request: AuditDataSourcesRequest) -> AuditDataSourcesResponse:
        """Execute the data source audit."""
        try:
            discovered_sources = []
            created_data_sources = []
            export_files = []
            
            # 1. Discover file system sources
            if request.scan_paths:
                fs_sources = await self.scanner.discover_file_system_sources(
                    request.scan_paths,
                    request.max_scan_depth,
                    request.min_file_count
                )
                discovered_sources.extend(fs_sources)
            
            # 2. Discover database sources
            if request.database_configs:
                db_sources = await self.scanner.discover_database_sources(
                    request.database_configs
                )
                discovered_sources.extend(db_sources)
            
            # 3. Discover API sources
            if request.api_endpoints:
                api_sources = await self.scanner.discover_web_api_sources(
                    request.api_endpoints
                )
                discovered_sources.extend(api_sources)
            
            # 4. Create DataSource entities if requested
            if request.save_to_repository:
                for source_info in discovered_sources:
                    try:
                        # Check if source already exists by name
                        existing = await self.repository.find_by_name(source_info['name'])
                        if existing:
                            continue
                        
                        # Create new data source
                        data_source = await self.scanner.create_data_source_from_discovery(
                            source_info
                        )
                        
                        # Save to repository
                        saved_source = await self.repository.save(data_source)
                        created_data_sources.append(saved_source)
                        
                    except Exception as e:
                        print(f"Error creating data source {source_info.get('name', 'unknown')}: {e}")
                        continue
            
            # 5. Get all data sources for export (including existing ones)
            all_data_sources = await self.repository.find_all()
            
            # 6. Export to CSV if requested
            if request.export_csv_path:
                await self.scanner.export_inventory_to_csv(
                    all_data_sources, 
                    request.export_csv_path
                )
                export_files.append(request.export_csv_path)
            
            # 7. Export to JSON if requested  
            if request.export_json_path:
                await self.scanner.export_inventory_to_json(
                    all_data_sources,
                    request.export_json_path
                )
                export_files.append(request.export_json_path)
            
            # 8. Generate statistics
            statistics = self._generate_audit_statistics(
                discovered_sources, 
                all_data_sources
            )
            
            return AuditDataSourcesResponse(
                success=True,
                discovered_sources=discovered_sources,
                created_data_sources=created_data_sources,
                statistics=statistics,
                export_files=export_files
            )
            
        except Exception as e:
            return AuditDataSourcesResponse(
                success=False,
                discovered_sources=[],
                created_data_sources=[],
                statistics={},
                export_files=[],
                error_message=str(e)
            )
    
    def _generate_audit_statistics(
        self, 
        discovered: List[Dict[str, Any]], 
        all_sources: List[DataSource]
    ) -> Dict[str, Any]:
        """Generate comprehensive audit statistics."""
        
        # Discovery statistics
        discovery_stats = self.scanner.get_discovery_statistics(discovered)
        
        # Repository statistics
        repo_stats = self._get_repository_statistics(all_sources)
        
        # Combined statistics
        return {
            'audit_timestamp': str(os.times()),
            'discovery': discovery_stats,
            'repository': repo_stats,
            'new_sources_discovered': len(discovered),
            'total_sources_in_inventory': len(all_sources),
            'meets_acceptance_criteria': len(all_sources) >= 20
        }
    
    def _get_repository_statistics(self, sources: List[DataSource]) -> Dict[str, Any]:
        """Get statistics about sources in repository."""
        if not sources:
            return {
                'total': 0,
                'active': 0,
                'inactive': 0,
                'by_type': {},
                'healthy': 0,
                'unhealthy': 0,
                'avg_quality_score': 0,
                'total_documents_processed': 0
            }
        
        type_counts = {}
        active_count = 0
        healthy_count = 0
        quality_scores = []
        total_docs = 0
        
        for source in sources:
            # Count by type
            type_name = source.source_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Count active/healthy
            if source.is_active:
                active_count += 1
            
            if source.is_healthy():
                healthy_count += 1
            
            # Quality scores
            if source.current_quality_score is not None:
                quality_scores.append(source.current_quality_score)
            
            # Document counts
            total_docs += source.total_documents_processed
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'total': len(sources),
            'active': active_count,
            'inactive': len(sources) - active_count,
            'by_type': type_counts,
            'healthy': healthy_count,
            'unhealthy': len(sources) - healthy_count,
            'avg_quality_score': round(avg_quality, 2),
            'total_documents_processed': total_docs,
            'sources_with_quality_data': len(quality_scores)
        }
    
    async def get_audit_summary(self) -> Dict[str, Any]:
        """Get a summary of the current data source inventory."""
        all_sources = await self.repository.find_all()
        return {
            'inventory_summary': self._get_repository_statistics(all_sources),
            'meets_task_criteria': len(all_sources) >= 20,
            'sources_needing_scan': len([s for s in all_sources if s.needs_scan()]),
            'unhealthy_sources': len([s for s in all_sources if not s.is_healthy()]),
            'last_updated': max([s.updated_at for s in all_sources]).isoformat() if all_sources else None
        }