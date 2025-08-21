"""
Workout API endpoints for fitness-related LLM interactions.
"""

import logging
from datetime import datetime, timezone
from queue import Empty
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.chat import (
    ApiErrorResponse,
    ChatRequest,
    HealthResponse,
)
from app.application.llm_service import LlmService, llm_service

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
    status: str, llm_available: bool, timestamp: str = None
) -> HealthResponse:
    """Create standardized health response."""
    return HealthResponse(
        status=status,
        service=HEALTH_CHECK_SERVICE_NAME,
        version=settings.app_version,
        llm_service_available=llm_available,
        timestamp=timestamp or _get_current_timestamp(),
    )


def _build_error_response(
    message: str,
    error_type: str,
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> dict:
    """Create standardized error response dictionary."""
    return ApiErrorResponse(
        code=code, type=error_type, param=param, message=message
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
                code="LLM_TIMEOUT",
            ),
        )
    elif "service error" in error_lower:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_build_error_response(
                message=ERROR_MESSAGES["service_error"],
                error_type="service_unavailable_error",
                code="LLM_SERVICE_ERROR",
            ),
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_build_error_response(
                message=ERROR_MESSAGES["generation_failed"],
                error_type="internal_server_error",
                code="GENERATION_FAILED",
            ),
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the workout API and LLM service",
    tags=["health"],
)
async def health(llm_service: LlmService = Depends(get_llm_service)) -> HealthResponse:
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
        logger.error("Health check failed: %s", e)
        return _build_health_response("unhealthy", False)


@router.get(
    "/models",
    response_model=Dict[str, Any],
    summary="Get Available Models",
    description="Get information about available LLM models for fitness AI prompts",
    tags=["models"],
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


@router.post(
    "/chat",
    response_class=StreamingResponse,
    summary="Chat with Fitvise AI (Streaming)",
    description="Send a chat message to the AI and receive a streaming response.",
    tags=["chat"],
)
async def chat(
    request: ChatRequest,
    llm_service: LlmService = Depends(get_llm_service),
) -> StreamingResponse:
    """
    Handle chat requests with streaming responses.

    Args:
        request: Chat request with message history
        llm_service: LLM service dependency

    Returns:
        StreamingResponse: A stream of JSON objects with response chunks
    """
    try:
        if not request.message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_build_error_response(
                    message="Messages cannot be empty",
                    error_type="invalid_request_error",
                    code="EMPTY_MESSAGES",
                    param="messages",
                ),
            )

        async def stream_generator():
            try:
                async for chunk in llm_service.chat(request):
                    yield f"{chunk.model_dump_json()}\n"
            except Exception as e:
                # Handle any exceptions from the LLM service
                error_response = _build_error_response(
                    message=str(e), error_type="stream_error", code="STREAM_ERROR"
                )
                yield f"{error_response}\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Unexpected error in chat endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_build_error_response(
                message="An unexpected error occurred",
                error_type="internal_server_error",
                code="UNEXPECTED_ERROR",
            ),
        )
