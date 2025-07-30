"""
Workout API endpoints for fitness-related LLM interactions.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.workout import (
    ApiErrorResponse,
    HealthResponse,
    PromptRequest,
    PromptResponse,
)
from app.services.llm_service import LlmService, QueryRequest, llm_service

logger = logging.getLogger(__name__)

# Create router for workout endpoints
router = APIRouter()


# Dependencies
def get_llm_service() -> LlmService:
    """Get LLM service instance dependency."""
    return llm_service


# Constants
HEALTH_CHECK_SERVICE_NAME = "workout-api"
ERROR_MESSAGES = {
    "empty_query": "Query cannot be empty",
    "timeout": "LLM service is currently unavailable due to timeout",
    "service_error": "LLM service is currently experiencing issues",
    "generation_failed": "Failed to generate workout plan",
    "unexpected_error": "An unexpected error occurred while generating workout plan",
}


# Helper functions
def _get_current_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _build_health_response(
    status: str, 
    llm_available: bool, 
    timestamp: str = None
) -> HealthResponse:
    """Create standardized health response."""
    return HealthResponse(
        status=status,
        service=HEALTH_CHECK_SERVICE_NAME,
        version=settings.app_version,
        llm_service_available=llm_available,
        timestamp=timestamp or _get_current_timestamp()
    )


def _build_error_response(
    message: str, 
    error_type: str, 
    code: Optional[str] = None, 
    param: Optional[str] = None
) -> dict:
    """Create standardized error response dictionary."""
    return ApiErrorResponse(
        code=code,
        type=error_type,
        param=param,
        message=message
    ).model_dump()


def _on_llm_error(error_message: str) -> HTTPException:
    """Handle LLM service errors with appropriate HTTP status codes."""
    error_lower = error_message.lower()
    
    if "timeout" in error_lower:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_build_error_response(
                message=ERROR_MESSAGES["timeout"],
                error_type="service_timeout_error",
                code="LLM_TIMEOUT"
            )
        )
    elif "service error" in error_lower:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_build_error_response(
                message=ERROR_MESSAGES["service_error"],
                error_type="service_unavailable_error",
                code="LLM_SERVICE_ERROR"
            )
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_build_error_response(
                message=ERROR_MESSAGES["generation_failed"],
                error_type="internal_server_error",
                code="GENERATION_FAILED"
            )
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the workout API and LLM service",
    tags=["health"]
)
async def health(
    llm_service: LlmService = Depends(get_llm_service)
) -> HealthResponse:
    """
    Perform comprehensive health check of the workout API service.
    
    Checks:
        - LLM service availability
        - Overall service status determination
        
    Returns:
        HealthResponse: Service status and availability information
    """
    try:
        llm_available = await llm_service.health()
        service_status = "healthy" if llm_available else "degraded"
        
        return _build_health_response(service_status, llm_available)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return _build_health_response("unhealthy", False)


@router.post(
    "/prompt",
    response_model=PromptResponse,
    responses={
        400: {"model": ApiErrorResponse, "description": "Invalid request parameters"},
        503: {"model": ApiErrorResponse, "description": "LLM service unavailable"},
        500: {"model": ApiErrorResponse, "description": "Internal server error"},
    },
    summary="Prompt Fitness AI",
    description="Send prompts to AI for fitness advice, workout plans, and health guidance",
    tags=["prompt"]
)
async def prompt(
    request: PromptRequest,
    llm_service: LlmService = Depends(get_llm_service),
) -> PromptResponse:
    """
    Send prompt to fitness AI and get response.
    
    Processes fitness-related prompts and returns AI-generated responses for
    workout plans, exercise recommendations, nutrition advice, and general fitness guidance.
    
    Args:
        request: User prompt and generation preferences
        llm_service: LLM service dependency
        
    Returns:
        WorkoutPromptResponse: AI response with metadata
        
    Raises:
        HTTPException: 400 for invalid requests, 503 for service issues, 500 for errors
    """
    try:
        # Input validation
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_build_error_response(
                    message=ERROR_MESSAGES["empty_query"],
                    error_type="invalid_request_error",
                    code="EMPTY_QUERY",
                    param="query"
                )
            )
        
        # Prepare LLM request
        llm_request = QueryRequest(
            query=request.query,
            context=request.context,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        # Generate workout plan
        logger.info(f"Generating response for query: {request.query[:100]}...")
        llm_response = await llm_service.query(llm_request)
        
        # Handle service errors
        if not llm_response.success:
            raise _on_llm_error(llm_response.error)
        
        # Build successful response
        return PromptResponse(
            response=llm_response.response,
            model=llm_response.model,
            tokens_used=llm_response.tokens_used,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            duration_ms=llm_response.total_duration_ms,
            success=True,
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in prompt processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_build_error_response(
                message=ERROR_MESSAGES["unexpected_error"],
                error_type="internal_server_error",
                code="UNEXPECTED_ERROR"
            )
        )


@router.get(
    "/models",
    response_model=Dict[str, Any],
    summary="Get Available Models",
    description="Get information about available LLM models for fitness AI prompts",
    tags=["models"]
)
async def get_available_models() -> Dict[str, Any]:
    """
    Get information about available LLM models and service configuration.
    
    Returns:
        Dict: Model information and service configuration details
    """
    return {
        "current_model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "default_temperature": settings.llm_temperature,
        "default_max_tokens": settings.llm_max_tokens,
        "timeout_seconds": settings.llm_timeout,
        "service": HEALTH_CHECK_SERVICE_NAME,
        "version": settings.app_version,
    }