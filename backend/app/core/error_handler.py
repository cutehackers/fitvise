"""Centralized error handling and response building.

This module provides consistent error handling patterns and response builders
to eliminate duplicate error handling code across the application.
"""

from typing import Dict, Optional, Any
from fastapi import HTTPException, status

from app.schemas.chat import ApiErrorResponse
from app.core.constants import ErrorMessages


class ErrorResponseBuilder:
    """Builder for creating standardized error responses.

    Provides consistent error response format across all API endpoints
    and eliminates duplicate error handling code.
    """

    @staticmethod
    def build_error_response(
        message: str,
        error_type: str,
        code: Optional[str] = None,
        param: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build standardized error response dictionary.

        Args:
            message: Human-readable error message
            error_type: Error type/category
            code: Optional error code for programmatic handling
            param: Optional parameter name that caused the error
            details: Additional error details

        Returns:
            Dictionary in standard error response format
        """
        error_response = ApiErrorResponse(
            code=code,
            type=error_type,
            param=param,
            message=message
        )

        response_dict = error_response.model_dump()
        if details:
            response_dict["details"] = details

        return response_dict

    @staticmethod
    def create_http_exception(
        message: str,
        error_type: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        code: Optional[str] = None,
        param: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """Create HTTPException with standardized error response.

        Args:
            message: Human-readable error message
            error_type: Error type/category
            status_code: HTTP status code
            code: Optional error code
            param: Optional parameter name
            details: Additional error details

        Returns:
            HTTPException with standardized error response
        """
        return HTTPException(
            status_code=status_code,
            detail=ErrorResponseBuilder.build_error_response(
                message=message,
                error_type=error_type,
                code=code,
                param=param,
                details=details
            )
        )


class LLMErrorHandler:
    """Specialized handler for LLM service errors.

    Analyzes error messages and returns appropriate HTTP status codes
    and error responses for LLM-related failures.
    """

    @staticmethod
    def handle_llm_error(error_message: str) -> HTTPException:
        """Handle LLM service errors with appropriate HTTP status codes.

        Args:
            error_message: Error message from LLM service

        Returns:
            HTTPException with appropriate status code and error details
        """
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            return ErrorResponseBuilder.create_http_exception(
                message=ErrorMessages.TIMEOUT,
                error_type="service_timeout_error",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="LLM_TIMEOUT"
            )
        elif "service error" in error_lower:
            return ErrorResponseBuilder.create_http_exception(
                message=ErrorMessages.SERVICE_ERROR,
                error_type="service_unavailable_error",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="LLM_SERVICE_ERROR"
            )
        else:
            return ErrorResponseBuilder.create_http_exception(
                message=ErrorMessages.GENERATION_FAILED,
                error_type="internal_server_error",
                code="GENERATION_FAILED"
            )


class ValidationErrorHandler:
    """Handler for input validation errors.

    Provides consistent error responses for common validation failures.
    """

    @staticmethod
    def empty_message_content(field: str = "content") -> HTTPException:
        """Create error for empty message content."""
        return ErrorResponseBuilder.create_http_exception(
            message=ErrorMessages.EMPTY_MESSAGE_CONTENT,
            error_type="invalid_request_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="EMPTY_MESSAGE_CONTENT",
            param=field
        )

    @staticmethod
    def messages_required() -> HTTPException:
        """Create error for missing messages."""
        return ErrorResponseBuilder.create_http_exception(
            message=ErrorMessages.MESSAGES_REQUIRED,
            error_type="invalid_request_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="EMPTY_MESSAGES",
            param="messages"
        )

    @staticmethod
    def session_id_required() -> HTTPException:
        """Create error for missing session ID."""
        return ErrorResponseBuilder.create_http_exception(
            message=ErrorMessages.SESSION_ID_REQUIRED,
            error_type="invalid_request_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="MISSING_SESSION_ID",
            param="session_id"
        )

    @staticmethod
    def unexpected_error(original_error: Exception) -> HTTPException:
        """Create error for unexpected exceptions."""
        return ErrorResponseBuilder.create_http_exception(
            message=ErrorMessages.UNEXPECTED_ERROR,
            error_type="internal_server_error",
            code="UNEXPECTED_ERROR",
            details={"original_error": str(original_error)}
        )