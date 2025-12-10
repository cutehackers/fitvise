"""External API documentation use case implementation (Task 1.1.2)."""
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta, timezone

from app.domain.entities.data_source import DataSource
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.value_objects.source_info import (
    SourceInfo, 
    SourceType, 
    ConnectionConfig,
    AuthConfig,
    AuthType,
    RateLimitConfig,
    AccessFrequency
)


class DocumentExternalApisRequest:
    """Request for external API documentation operation."""
    
    def __init__(
        self,
        api_endpoints: Optional[List[str]] = None,
        api_discovery_urls: Optional[List[str]] = None,
        include_common_apis: bool = True,
        validate_endpoints: bool = True,
        rate_limit_test: bool = False,
        timeout_seconds: int = 10,
        save_to_repository: bool = True,
        export_documentation: Optional[str] = None
    ):
        """Initialize API documentation request."""
        self.api_endpoints = api_endpoints or []
        self.api_discovery_urls = api_discovery_urls or []
        self.include_common_apis = include_common_apis
        self.validate_endpoints = validate_endpoints
        self.rate_limit_test = rate_limit_test
        self.timeout_seconds = timeout_seconds
        self.save_to_repository = save_to_repository
        self.export_documentation = export_documentation


class ApiDocumentation:
    """Documentation for a single API."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        description: str,
        version: Optional[str] = None,
        auth_methods: Optional[List[str]] = None,
        rate_limits: Optional[Dict[str, Any]] = None,
        endpoints: Optional[List[Dict[str, Any]]] = None,
        documentation_url: Optional[str] = None,
        openapi_spec_url: Optional[str] = None,
        requires_key: bool = True,
        health_check_url: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        status: str = "unknown"
    ):
        """Initialize API documentation."""
        self.name = name
        self.base_url = base_url
        self.description = description
        self.version = version
        self.auth_methods = auth_methods or []
        self.rate_limits = rate_limits or {}
        self.endpoints = endpoints or []
        self.documentation_url = documentation_url
        self.openapi_spec_url = openapi_spec_url
        self.requires_key = requires_key
        self.health_check_url = health_check_url
        self.response_time_ms = response_time_ms
        self.status = status


class DocumentExternalApisResponse:
    """Response from external API documentation operation."""
    
    def __init__(
        self,
        success: bool,
        documented_apis: List[ApiDocumentation],
        created_data_sources: List[DataSource],
        validation_results: Dict[str, Any],
        export_files: List[str],
        error_message: Optional[str] = None
    ):
        """Initialize API documentation response."""
        self.success = success
        self.documented_apis = documented_apis
        self.created_data_sources = created_data_sources
        self.validation_results = validation_results
        self.export_files = export_files
        self.error_message = error_message
    
    @property
    def total_documented(self) -> int:
        """Get total number of documented APIs."""
        return len(self.documented_apis)
    
    @property
    def total_validated(self) -> int:
        """Get total number of validated APIs."""
        return len([api for api in self.documented_apis if api.status == "healthy"])


class DocumentExternalApisUseCase:
    """Task 1.1.2 - Document external APIs with access requirements & rate limits.
    
    Examples:
        >>> use_case = DocumentExternalApisUseCase(repository)
        >>> request = DocumentExternalApisRequest(include_common_apis=True, validate_endpoints=True)
        >>> response = await use_case.execute(request)
        >>> response.success and response.total_documented >= 3
        True
        >>> any("github" in api.name.lower() for api in response.documented_apis)
        True
    """
    
    def __init__(
        self,
        data_source_repository: DataSourceRepository
    ):
        """Initialize the use case."""
        self.repository = data_source_repository
        
        # Common APIs that might be useful for RAG systems
        self.common_apis = {
            "github": {
                "name": "GitHub API",
                "base_url": "https://api.github.com",
                "description": "Access GitHub repositories, issues, and documentation",
                "documentation_url": "https://docs.github.com/en/rest",
                "auth_methods": ["token", "oauth"],
                "rate_limits": {"authenticated": "5000/hour", "unauthenticated": "60/hour"},
                "health_check": "/rate_limit"
            },
            "confluence": {
                "name": "Confluence API",
                "base_url": "https://{domain}.atlassian.net/wiki/rest/api",
                "description": "Access Confluence pages and spaces",
                "documentation_url": "https://developer.atlassian.com/cloud/confluence/rest/",
                "auth_methods": ["basic", "oauth2", "api_token"],
                "rate_limits": {"standard": "1000/hour"},
                "health_check": "/space"
            },
            "slack": {
                "name": "Slack API",
                "base_url": "https://slack.com/api",
                "description": "Access Slack messages, channels, and files",
                "documentation_url": "https://api.slack.com/",
                "auth_methods": ["oauth2", "bot_token"],
                "rate_limits": {"tier1": "1/minute", "tier2": "20/minute", "tier3": "50/minute"},
                "health_check": "/auth.test"
            },
            "notion": {
                "name": "Notion API",
                "base_url": "https://api.notion.com/v1",
                "description": "Access Notion pages and databases",
                "documentation_url": "https://developers.notion.com/reference",
                "auth_methods": ["bearer_token"],
                "rate_limits": {"requests": "3/second"},
                "health_check": "/users/me"
            },
            "sharepoint": {
                "name": "SharePoint API",
                "base_url": "https://{tenant}.sharepoint.com/_api",
                "description": "Access SharePoint sites, lists, and documents",
                "documentation_url": "https://docs.microsoft.com/en-us/sharepoint/dev/",
                "auth_methods": ["oauth2", "app_only"],
                "rate_limits": {"standard": "2000/minute"},
                "health_check": "/web"
            }
        }
    
    async def execute(self, request: DocumentExternalApisRequest) -> DocumentExternalApisResponse:
        """Execute the external API documentation process."""
        try:
            documented_apis = []
            created_data_sources = []
            validation_results = {}
            export_files = []
            
            # 1. Include common APIs if requested
            apis_to_document = []
            if request.include_common_apis:
                apis_to_document.extend(self.common_apis.keys())
            
            # 2. Add explicitly provided endpoints
            if request.api_endpoints:
                for endpoint in request.api_endpoints:
                    api_name = self._extract_api_name_from_url(endpoint)
                    apis_to_document.append(api_name)
            
            # 3. Discover APIs from discovery URLs
            if request.api_discovery_urls:
                discovered = await self._discover_apis_from_urls(request.api_discovery_urls)
                apis_to_document.extend(discovered)
            
            # 4. Document each API
            for api_key in apis_to_document:
                try:
                    if api_key in self.common_apis:
                        api_doc = await self._document_common_api(
                            api_key, 
                            self.common_apis[api_key],
                            request
                        )
                    else:
                        api_doc = await self._document_custom_api(api_key, request)
                    
                    if api_doc:
                        documented_apis.append(api_doc)
                        
                        # Validate if requested
                        if request.validate_endpoints:
                            validation_result = await self._validate_api(api_doc, request)
                            validation_results[api_doc.name] = validation_result
                            api_doc.status = validation_result.get("status", "unknown")
                            api_doc.response_time_ms = validation_result.get("response_time_ms")
                
                except Exception as e:
                    print(f"Error documenting API {api_key}: {e}")
                    continue
            
            # 5. Create DataSource entities if requested
            if request.save_to_repository:
                for api_doc in documented_apis:
                    try:
                        # Check if source already exists
                        existing = await self.repository.find_by_name(api_doc.name)
                        if existing:
                            continue
                        
                        # Create data source from API documentation
                        data_source = self._create_data_source_from_api_doc(api_doc)
                        saved_source = await self.repository.save(data_source)
                        created_data_sources.append(saved_source)
                        
                    except Exception as e:
                        print(f"Error creating data source for {api_doc.name}: {e}")
                        continue
            
            # 6. Export documentation if requested
            if request.export_documentation:
                export_path = await self._export_api_documentation(
                    documented_apis, 
                    request.export_documentation
                )
                export_files.append(export_path)
            
            return DocumentExternalApisResponse(
                success=True,
                documented_apis=documented_apis,
                created_data_sources=created_data_sources,
                validation_results=validation_results,
                export_files=export_files
            )
            
        except Exception as e:
            return DocumentExternalApisResponse(
                success=False,
                documented_apis=[],
                created_data_sources=[],
                validation_results={},
                export_files=[],
                error_message=str(e)
            )
    
    async def _document_common_api(
        self, 
        api_key: str, 
        api_info: Dict[str, Any], 
        request: DocumentExternalApisRequest
    ) -> ApiDocumentation:
        """Document a common API from predefined information."""
        
        # Parse rate limits
        rate_limit_config = None
        if api_info.get("rate_limits"):
            rate_limit_config = self._parse_rate_limits(api_info["rate_limits"])
        
        # Create health check URL
        health_check_url = None
        if api_info.get("health_check"):
            health_check_url = urljoin(api_info["base_url"], api_info["health_check"])
        
        return ApiDocumentation(
            name=api_info["name"],
            base_url=api_info["base_url"],
            description=api_info["description"],
            auth_methods=api_info.get("auth_methods", []),
            rate_limits=api_info.get("rate_limits", {}),
            documentation_url=api_info.get("documentation_url"),
            requires_key=len(api_info.get("auth_methods", [])) > 0,
            health_check_url=health_check_url
        )
    
    async def _document_custom_api(
        self, 
        endpoint: str, 
        request: DocumentExternalApisRequest
    ) -> Optional[ApiDocumentation]:
        """Document a custom API endpoint."""
        parsed_url = urlparse(endpoint)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        return ApiDocumentation(
            name=f"Custom API: {parsed_url.netloc}",
            base_url=base_url,
            description=f"Custom API endpoint at {parsed_url.netloc}",
            auth_methods=["api_key"],  # Assume API key required
            requires_key=True,
            health_check_url=endpoint
        )
    
    async def _discover_apis_from_urls(self, discovery_urls: List[str]) -> List[str]:
        """Discover APIs from discovery URLs (OpenAPI directories, etc.)."""
        discovered = []
        
        # This would implement actual API discovery logic
        # For now, return empty list as this requires complex web scraping
        
        return discovered
    
    async def _validate_api(
        self, 
        api_doc: ApiDocumentation, 
        request: DocumentExternalApisRequest
    ) -> Dict[str, Any]:
        """Validate an API endpoint."""
        if not api_doc.health_check_url:
            return {"status": "unknown", "error": "No health check URL"}
        
        try:
            timeout = aiohttp.ClientTimeout(total=request.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = datetime.now(timezone.utc)
                
                async with session.head(api_doc.health_check_url) as response:
                    end_time = datetime.now(timezone.utc)
                    response_time = (end_time - start_time).total_seconds() * 1000
                    
                    if response.status < 400:
                        return {
                            "status": "healthy",
                            "response_time_ms": response_time,
                            "status_code": response.status
                        }
                    elif response.status == 401:
                        return {
                            "status": "requires_auth",
                            "response_time_ms": response_time,
                            "status_code": response.status,
                            "note": "API requires authentication"
                        }
                    else:
                        return {
                            "status": "error",
                            "response_time_ms": response_time,
                            "status_code": response.status,
                            "error": f"HTTP {response.status}"
                        }
        
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": f"Request timed out after {request.timeout_seconds}s"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _extract_api_name_from_url(self, url: str) -> str:
        """Extract API name from URL."""
        parsed = urlparse(url)
        return parsed.netloc.replace(".", "_")
    
    def _parse_rate_limits(self, rate_limit_info: Dict[str, str]) -> RateLimitConfig:
        """Parse rate limit information into config."""
        # Simple parsing - in real implementation would be more sophisticated
        config = RateLimitConfig()
        
        for key, value in rate_limit_info.items():
            if "hour" in value.lower():
                requests = int(value.split("/")[0])
                config = RateLimitConfig(requests_per_hour=requests)
            elif "minute" in value.lower():
                requests = int(value.split("/")[0])
                config = RateLimitConfig(requests_per_minute=requests)
            elif "second" in value.lower():
                requests = float(value.split("/")[0])
                config = RateLimitConfig(requests_per_second=requests)
        
        return config
    
    def _create_data_source_from_api_doc(self, api_doc: ApiDocumentation) -> DataSource:
        """Create a DataSource entity from API documentation."""
        
        # Determine auth type from methods
        auth_type = AuthType.NONE
        if api_doc.requires_key:
            if "oauth" in api_doc.auth_methods or "oauth2" in api_doc.auth_methods:
                auth_type = AuthType.OAUTH2
            elif "bearer" in api_doc.auth_methods or "token" in api_doc.auth_methods:
                auth_type = AuthType.BEARER_TOKEN
            elif "basic" in api_doc.auth_methods:
                auth_type = AuthType.BASIC_AUTH
            else:
                auth_type = AuthType.API_KEY
        
        # Create configurations
        connection_config = ConnectionConfig(
            base_url=api_doc.base_url,
            custom_params={
                "version": api_doc.version,
                "documentation_url": api_doc.documentation_url,
                "openapi_spec_url": api_doc.openapi_spec_url
            }
        )
        
        auth_config = AuthConfig(auth_type=auth_type)
        
        rate_limit_config = None
        if api_doc.rate_limits:
            rate_limit_config = self._parse_rate_limits(api_doc.rate_limits)
        
        # Create source info
        source_info = SourceInfo(
            name=api_doc.name,
            source_type=SourceType.WEB_API,
            description=api_doc.description,
            connection_config=connection_config,
            auth_config=auth_config,
            rate_limit_config=rate_limit_config,
            access_frequency=AccessFrequency.DAILY,
            is_active=True,
            health_check_url=api_doc.health_check_url,
            health_status=api_doc.status,
            priority=7,  # APIs typically high priority
            estimated_document_count=1000  # Estimate
        )
        
        # Create data source
        return DataSource(
            name=api_doc.name,
            source_type=SourceType.WEB_API,
            description=api_doc.description,
            source_info=source_info
        )
    
    async def _export_api_documentation(
        self, 
        apis: List[ApiDocumentation], 
        export_path: str
    ) -> str:
        """Export API documentation to file."""
        import json
        import aiofiles
        
        export_data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_apis": len(apis),
            "apis": []
        }
        
        for api in apis:
            api_data = {
                "name": api.name,
                "base_url": api.base_url,
                "description": api.description,
                "version": api.version,
                "auth_methods": api.auth_methods,
                "rate_limits": api.rate_limits,
                "endpoints": api.endpoints,
                "documentation_url": api.documentation_url,
                "openapi_spec_url": api.openapi_spec_url,
                "requires_key": api.requires_key,
                "health_check_url": api.health_check_url,
                "response_time_ms": api.response_time_ms,
                "status": api.status
            }
            export_data["apis"].append(api_data)
        
        async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(export_data, indent=2, ensure_ascii=False))
        
        return export_path
    
    async def get_api_registry_summary(self) -> Dict[str, Any]:
        """Get summary of documented APIs."""
        # Get all API-type data sources
        api_sources = await self.repository.find_by_type(SourceType.WEB_API)
        
        healthy_count = len([source for source in api_sources if source.is_healthy()])
        
        return {
            "total_documented_apis": len(api_sources),
            "healthy_apis": healthy_count,
            "unhealthy_apis": len(api_sources) - healthy_count,
            "apis_requiring_auth": len([
                source for source in api_sources 
                if source.source_info.has_auth
            ]),
            "avg_response_time": None,  # Would calculate from stored metrics
            "last_validation": None,  # Would track last validation run
            "meets_task_criteria": len(api_sources) > 0  # Has documented APIs
        }