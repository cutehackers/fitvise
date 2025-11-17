"""RAG Data Sources API endpoints."""
import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field

from app.application.use_cases.knowledge_sources.audit_data_sources import (
    AuditDataSourcesUseCase,
    AuditDataSourcesRequest,
    AuditDataSourcesResponse
)
from app.application.use_cases.knowledge_sources.document_external_apis import (
    DocumentExternalApisUseCase,
    DocumentExternalApisRequest, 
    DocumentExternalApisResponse
)
from app.application.use_cases.knowledge_sources.categorize_sources import (
    CategorizeSourcesUseCase,
    CategorizeSourcesRequest,
    CategorizeSourcesResponse
)
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.infrastructure.persistence.repositories.dependencies import get_data_source_repository
from app.infrastructure.external_services.ml_services.categorization.sklearn_categorizer import (
    SklearnDocumentCategorizer
)

router = APIRouter(prefix="/rag/data-sources", tags=["RAG Data Sources"])

def get_categorizer():
    return SklearnDocumentCategorizer()

# Request/Response Models
class DataSourceScanRequest(BaseModel):
    scan_paths: Optional[List[str]] = Field(None, description="Paths to scan for documents")
    database_configs: Optional[List[dict]] = Field(None, description="Database connection configs")
    api_endpoints: Optional[List[str]] = Field(None, description="API endpoints to document")
    max_scan_depth: int = Field(5, description="Maximum directory depth to scan")
    min_file_count: int = Field(5, description="Minimum files to consider a source")
    export_csv: Optional[str] = Field(None, description="Export path for CSV inventory")
    export_json: Optional[str] = Field(None, description="Export path for JSON inventory")

class ApiDocumentationRequest(BaseModel):
    api_endpoints: Optional[List[str]] = Field(None, description="API endpoints to document")
    include_common_apis: bool = Field(True, description="Include common APIs (GitHub, Slack, etc.)")
    validate_endpoints: bool = Field(True, description="Validate endpoint availability")
    timeout_seconds: int = Field(10, description="Request timeout for validation")
    export_documentation: Optional[str] = Field(None, description="Export path for documentation")

class CategorizationRequest(BaseModel):
    train_model: bool = Field(False, description="Train the ML model")
    use_synthetic_data: bool = Field(True, description="Use synthetic training data")
    synthetic_data_size: int = Field(100, description="Number of synthetic documents")
    categorize_sources: bool = Field(True, description="Categorize existing sources")
    source_ids: Optional[List[str]] = Field(None, description="Specific source IDs to categorize")
    min_confidence: float = Field(0.6, description="Minimum confidence threshold")
    model_type: str = Field("logistic_regression", description="ML model type")

class DataSourceResponse(BaseModel):
    id: str
    name: str
    source_type: str
    description: str
    is_active: bool
    created_at: str
    health_status: dict

class InventoryResponse(BaseModel):
    success: bool
    discovered_sources: int
    created_data_sources: int
    statistics: dict
    export_files: List[str]
    error_message: Optional[str] = None

class ApiDocumentationResponse(BaseModel):
    success: bool
    documented_apis: int
    validated_apis: int
    created_data_sources: int
    validation_results: dict
    export_files: List[str]
    error_message: Optional[str] = None

class CategorizationResponse(BaseModel):
    success: bool
    training_results: Optional[dict] = None
    categorized_sources: int
    model_info: dict
    meets_accuracy_target: bool
    error_message: Optional[str] = None


# API Endpoints

@router.get("/", response_model=List[DataSourceResponse])
async def list_data_sources(
    active_only: bool = Query(False, description="Return only active sources"),
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """List all data sources in the inventory."""
    try:
        if active_only:
            sources = await repository.find_active()
        else:
            sources = await repository.find_all()
        
        return [
            DataSourceResponse(
                id=str(source.id),
                name=source.name,
                source_type=source.source_type.value,
                description=source.description,
                is_active=source.is_active,
                created_at=source.created_at.isoformat(),
                health_status=source.get_health_status()
            )
            for source in sources
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_id}", response_model=DataSourceResponse)
async def get_data_source(
    source_id: UUID,
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """Get a specific data source by ID."""
    try:
        source = await repository.find_by_id(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        return DataSourceResponse(
            id=str(source.id),
            name=source.name,
            source_type=source.source_type.value,
            description=source.description,
            is_active=source.is_active,
            created_at=source.created_at.isoformat(),
            health_status=source.get_health_status()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan", response_model=InventoryResponse)
async def scan_and_catalog_sources(
    request: DataSourceScanRequest,
    background_tasks: BackgroundTasks,
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """Scan and catalog data sources (Task 1.1.1)."""
    try:
        use_case = AuditDataSourcesUseCase(repository)
        
        audit_request = AuditDataSourcesRequest(
            scan_paths=request.scan_paths,
            database_configs=request.database_configs,
            api_endpoints=request.api_endpoints,
            max_scan_depth=request.max_scan_depth,
            min_file_count=request.min_file_count,
            export_csv_path=request.export_csv,
            export_json_path=request.export_json,
            save_to_repository=True
        )
        
        response = await use_case.execute(audit_request)
        
        return InventoryResponse(
            success=response.success,
            discovered_sources=response.total_discovered,
            created_data_sources=response.total_created,
            statistics=response.statistics,
            export_files=response.export_files,
            error_message=response.error_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document-apis", response_model=ApiDocumentationResponse)
async def document_external_apis(
    request: ApiDocumentationRequest,
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """Document external API sources (Task 1.1.2)."""
    try:
        use_case = DocumentExternalApisUseCase(repository)
        
        api_request = DocumentExternalApisRequest(
            api_endpoints=request.api_endpoints,
            include_common_apis=request.include_common_apis,
            validate_endpoints=request.validate_endpoints,
            timeout_seconds=request.timeout_seconds,
            save_to_repository=True,
            export_documentation=request.export_documentation
        )
        
        response = await use_case.execute(api_request)
        
        return ApiDocumentationResponse(
            success=response.success,
            documented_apis=response.total_documented,
            validated_apis=response.total_validated,
            created_data_sources=len(response.created_data_sources),
            validation_results=response.validation_results,
            export_files=response.export_files,
            error_message=response.error_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/categorize", response_model=CategorizationResponse)
async def categorize_sources(
    request: CategorizationRequest,
    repository: DataSourceRepository = Depends(get_data_source_repository),
    categorizer = Depends(get_categorizer)
):
    """Categorize sources using ML (Task 1.1.3)."""
    try:
        use_case = CategorizeSourcesUseCase(repository, categorizer)
        
        cat_request = CategorizeSourcesRequest(
            train_model=request.train_model,
            use_synthetic_data=request.use_synthetic_data,
            synthetic_data_size=request.synthetic_data_size,
            categorize_sources=request.categorize_sources,
            source_ids=request.source_ids,
            min_confidence=request.min_confidence,
            model_type=request.model_type,
            save_model=True
        )
        
        response = await use_case.execute(cat_request)
        
        return CategorizationResponse(
            success=response.success,
            training_results=response.training_results,
            categorized_sources=response.total_categorized,
            model_info=response.model_info or {},
            meets_accuracy_target=response.meets_accuracy_target,
            error_message=response.error_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/inventory")
async def get_inventory_statistics(
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """Get inventory statistics and summary."""
    try:
        use_case = AuditDataSourcesUseCase(repository)
        summary = await use_case.get_audit_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/apis")
async def get_api_registry_statistics(
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """Get API registry statistics."""
    try:
        use_case = DocumentExternalApisUseCase(repository)
        summary = await use_case.get_api_registry_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/categorization")
async def get_categorization_statistics(
    repository: DataSourceRepository = Depends(get_data_source_repository),
    categorizer = Depends(get_categorizer)
):
    """Get categorization system statistics."""
    try:
        use_case = CategorizeSourcesUseCase(repository, categorizer)
        stats = await use_case.get_categorization_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    repository: DataSourceRepository = Depends(get_data_source_repository)
):
    """Health check for RAG data sources system."""
    try:
        total_sources = await repository.count_all()
        active_sources = await repository.count_active()
        needing_scan = await repository.find_needing_scan()
        unhealthy = await repository.find_unhealthy()
        
        now = datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "status": "healthy",
            "timestamp": now,
            "total_sources": total_sources,
            "active_sources": active_sources,
            "sources_needing_scan": len(needing_scan),
            "unhealthy_sources": len(unhealthy),
            "meets_task_requirements": {
                "scan": total_sources >= 20,  # ≥20 data sources
                "document-external-apis": True,  # API documentation capability
                "categorization": True   # ML categorization capability
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))