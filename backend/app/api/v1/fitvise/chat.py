"""
Workout API endpoints for fitness-related LLM interactions.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.v1.fitvise.dependencies import (
    get_rag_chat_use_case,
    get_llm_health_monitor,
)
from app.infrastructure.llm.dependencies import get_chat_use_case, get_llm_service
from app.core.settings import settings
from app.core.error_handler import (
    ErrorResponseBuilder,
    LLMErrorHandler,
    ValidationErrorHandler,
)
from app.core.constants import ServiceNames
from app.schemas.chat import (
    ChatRequest,
    HealthResponse,
    RagChatResponse,
    SourceCitation,
    ChatMessage,
)
from app.domain.entities.message_role import MessageRole
from app.domain.llm.exceptions import ChatOrchestratorError, MessageValidationError
from app.application.use_cases.chat.rag_chat_use_case import RagChatUseCase

logger = logging.getLogger(__name__)

# Create router for workout endpoints
router = APIRouter()




# Helper functions
def _get_current_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _build_health_response(status: str, llm_available: bool, timestamp: str = None) -> HealthResponse:
    """Create standardized health response."""
    return HealthResponse(
        status=status,
        service=ServiceNames.WORKOUT_API,
        version=settings.app_version,
        llm_service_available=llm_available,
        timestamp=timestamp or _get_current_timestamp(),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the workout API and LLM service",
    tags=["health"],
)
async def health(
    llm_service=Depends(get_llm_service),
    chat_orchestrator=Depends(get_chat_use_case),
) -> HealthResponse:
    """
    Perform comprehensive health check of the workout API service.

    Checks:
        - LLM service availability
        - Chat orchestrator health
        - Overall service status determination

    Returns:
        HealthResponse: Service status and availability information
    """
    try:
        # Check both LLM service and chat orchestrator
        service_healthy = await llm_service.health_check()
        orchestrator_healthy = await chat_orchestrator.health_check()

        llm_available = service_healthy and orchestrator_healthy
        service_status = "healthy" if llm_available else "degraded"

        return _build_health_response(service_status, llm_available)

    except Exception as e:
        logger.error("Health check failed: %s", e)
        return _build_health_response("unhealthy", False)


@router.get(
    "/health/llm",
    response_model=Dict[str, Any],
    summary="LLM Health Check",
    description="Detailed health check for Ollama LLM service with performance metrics",
    tags=["health"],
)
async def llm_health(
    health_monitor=Depends(get_llm_health_monitor),
) -> Dict[str, Any]:
    """
    Perform detailed LLM health check.

    Returns health status, response times, and success rates for the Ollama service.

    Returns:
        Dict: Detailed LLM health metrics including:
            - status: healthy/unhealthy
            - model: Model name
            - response_time_ms: Current response time
            - avg_response_time_ms: Average response time
            - p95_response_time_ms: 95th percentile response time
            - success_rate: Success rate percentage
            - error: Error message if unhealthy
    """
    try:
        health_status = await health_monitor.check_health()
        return health_status

    except Exception as e:
        logger.error("LLM health check failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "model": settings.llm_model,
        }


@router.get(
    "/health/llm/metrics",
    response_model=Dict[str, Any],
    summary="LLM Performance Metrics",
    description="Get LLM performance metrics without performing health check",
    tags=["health"],
)
async def llm_metrics(
    health_monitor=Depends(get_llm_health_monitor),
) -> Dict[str, Any]:
    """
    Get current LLM performance metrics.

    Returns accumulated metrics without performing a new health check.

    Returns:
        Dict: Performance metrics including:
            - avg_response_time_ms: Average response time
            - p95_response_time_ms: 95th percentile response time
            - success_rate: Success rate percentage
            - total_checks: Total health checks performed
            - error_count: Number of errors
            - last_check: Last check timestamp
    """
    try:
        metrics = await health_monitor.get_metrics()
        return metrics

    except Exception as e:
        logger.error("Failed to get LLM metrics: %s", e)
        return {
            "error": str(e),
            "total_checks": 0,
        }


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
    chat_use_case=Depends(get_chat_use_case),
) -> StreamingResponse:
    """
    Handle chat requests with streaming responses.

    Args:
        request: Chat request with message history
        chat_use_case: Chat orchestrator dependency

    Returns:
        StreamingResponse: A stream of JSON objects with response chunks
    """
    try:
        # Log request details for debugging
        logger.info(
            f"Received chat request - session_id: {request.session_id}, "
            f"message_length: {len(request.message.content) if request.message and request.message.content else 'None'}, "
            f"message_role: {request.message.role if request.message else 'None'}"
        )

        if not request.message:
            raise ValidationErrorHandler.messages_required()

        # Add validation for message content (consistent with /chat-rag endpoint)
        if not request.message.content or not request.message.content.strip():
            raise ValidationErrorHandler.empty_message_content()

        async def stream_generator():
            try:
                async for chunk in chat_use_case.chat(request):
                    yield f"{chunk.model_dump_json()}\n"
            except MessageValidationError as e:
                error_response = ErrorResponseBuilder.build_error_response(
                    message=str(e),
                    error_type="invalid_request_error",
                    code="VALIDATION_ERROR",
                    param=e.field if hasattr(e, 'field') else None
                )
                yield f"{error_response}\n"
            except ChatOrchestratorError as e:
                error_response = ErrorResponseBuilder.build_error_response(
                    message=f"Chat processing failed: {str(e)}",
                    error_type="service_error",
                    code="CHAT_PROCESSING_ERROR"
                )
                yield f"{error_response}\n"
            except Exception as e:
                error_response = ErrorResponseBuilder.build_error_response(
                    message=str(e),
                    error_type="stream_error",
                    code="STREAM_ERROR"
                )
                yield f"{error_response}\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Unexpected error in chat endpoint: %s", e)
        raise ValidationErrorHandler.unexpected_error(e)


@router.post(
    "/chat-rag",
    response_class=StreamingResponse,
    summary="Chat with RAG (Retrieval-Augmented Generation)",
    description="Send a chat message and receive AI response enhanced with retrieved context from documents.",
    tags=["chat", "rag"],
)
async def chat_with_rag(
    request: ChatRequest,
    rag_chat_use_case: Annotated[RagChatUseCase, Depends(get_rag_chat_use_case)],
) -> StreamingResponse:
    """
    Handle chat requests with RAG (Retrieval-Augmented Generation).

    Retrieves relevant document chunks from the knowledge base and uses them
    as context for generating responses. Sources are included in the final response.

    Args:
        request: Chat request with message
        rag_orchestrator: RAG orchestrator dependency

    Returns:
        StreamingResponse: Streaming JSON objects with response chunks and sources
    """
    try:
        if not request.message or not request.message.content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponseBuilder.build_error_response(
                    message="Message content cannot be empty",
                    error_type="invalid_request_error",
                    code="EMPTY_MESSAGE_CONTENT",
                    param="message.content",
                ),
            )

        async def stream_rag_generator():
            """Generate streaming RAG response with sources."""
            try:
                # Stream RAG-enhanced responses with context and sources
                async for response in rag_chat_use_case.chat(request):
                    yield f"{response.model_dump_json()}\n"

            except MessageValidationError as e:
                logger.error("RAG validation error: %s", str(e))
                error_response = ErrorResponseBuilder.build_error_response(
                    message=str(e),
                    error_type="invalid_request_error",
                    code="RAG_VALIDATION_ERROR",
                    param=e.field if hasattr(e, 'field') else None
                )
                yield f"{error_response}\n"
            except ChatOrchestratorError as e:
                logger.error("RAG processing error: %s", str(e))
                error_response = ErrorResponseBuilder.build_error_response(
                    message=f"RAG processing failed: {str(e)}",
                    error_type="service_error",
                    code="RAG_PROCESSING_ERROR"
                )
                yield f"{error_response}\n"
            except Exception as e:
                logger.error("RAG streaming error: %s", str(e))
                error_response = ErrorResponseBuilder.build_error_response(
                    message=f"RAG streaming failed: {str(e)}",
                    error_type="rag_stream_error",
                    code="RAG_STREAM_ERROR",
                )
                yield f"{error_response}\n"

        return StreamingResponse(stream_rag_generator(), media_type="application/x-ndjson")

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Unexpected error in RAG chat endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponseBuilder.build_error_response(
                message="An unexpected error occurred",
                error_type="internal_server_error",
                code="UNEXPECTED_ERROR",
            ),
        )
