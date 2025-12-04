"""Test suite for error handler module."""

import pytest
from fastapi import HTTPException, status

from app.core.error_handler import (
    ErrorResponseBuilder,
    LLMErrorHandler,
    ValidationErrorHandler,
)
from app.core.constants import ErrorMessages


class TestErrorResponseBuilder:
    """Test ErrorResponseBuilder functionality."""

    def test_build_basic_error_response(self):
        """Test building basic error response."""
        response = ErrorResponseBuilder.build_error_response(
            message="Test error",
            error_type="test_error"
        )

        assert response["type"] == "test_error"
        assert response["message"] == "Test error"
        assert response["code"] is None
        assert response["param"] is None

    def test_build_error_response_with_optional_fields(self):
        """Test building error response with all optional fields."""
        details = {"additional_info": "test"}

        response = ErrorResponseBuilder.build_error_response(
            message="Detailed error",
            error_type="detailed_error",
            code="TEST_ERROR",
            param="test_param",
            details=details
        )

        assert response["type"] == "detailed_error"
        assert response["message"] == "Detailed error"
        assert response["code"] == "TEST_ERROR"
        assert response["param"] == "test_param"
        assert response["details"] == details

    def test_create_http_exception_basic(self):
        """Test creating basic HTTP exception."""
        exception = ErrorResponseBuilder.create_http_exception(
            message="HTTP error",
            error_type="http_error"
        )

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exception.detail["type"] == "http_error"
        assert exception.detail["message"] == "HTTP error"

    def test_create_http_exception_with_status_code(self):
        """Test creating HTTP exception with custom status code."""
        exception = ErrorResponseBuilder.create_http_exception(
            message="Bad request",
            error_type="bad_request",
            status_code=status.HTTP_400_BAD_REQUEST
        )

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_400_BAD_REQUEST
        assert exception.detail["type"] == "bad_request"
        assert exception.detail["message"] == "Bad request"

    def test_error_response_structure_consistency(self):
        """Test all error responses have consistent structure."""
        responses = [
            ErrorResponseBuilder.build_error_response("test", "test"),
            ErrorResponseBuilder.create_http_exception("test", "test").detail
        ]

        for response in responses:
            assert "type" in response
            assert "message" in response
            assert isinstance(response["type"], str)
            assert isinstance(response["message"], str)


class TestLLMErrorHandler:
    """Test LLMErrorHandler functionality."""

    def test_handle_timeout_error(self):
        """Test handling timeout errors."""
        exception = LLMErrorHandler.handle_llm_error("Request timeout occurred")

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert exception.detail["code"] == "LLM_TIMEOUT"
        assert exception.detail["type"] == "service_timeout_error"
        assert "timeout" in exception.detail["message"].lower()

    def test_handle_service_error(self):
        """Test handling service errors."""
        exception = LLMErrorHandler.handle_llm_error("Service error occurred")

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert exception.detail["code"] == "LLM_SERVICE_ERROR"
        assert exception.detail["type"] == "service_unavailable_error"
        assert "service" in exception.detail["message"].lower()

    def test_handle_generic_error(self):
        """Test handling generic LLM errors."""
        exception = LLMErrorHandler.handle_llm_error("Generic error occurred")

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exception.detail["code"] == "GENERATION_FAILED"
        assert exception.detail["type"] == "internal_server_error"

    def test_case_insensitive_error_detection(self):
        """Test error detection is case insensitive."""
        timeout_variants = [
            "Request TIMEOUT occurred",
            "request timeout happened",
            "TIMEOUT error"
        ]

        for error_msg in timeout_variants:
            exception = LLMErrorHandler.handle_llm_error(error_msg)
            assert exception.detail["code"] == "LLM_TIMEOUT"

        service_variants = [
            "Service ERROR occurred",
            "service error happened",
            "SERVICE ERROR"
        ]

        for error_msg in service_variants:
            exception = LLMErrorHandler.handle_llm_error(error_msg)
            assert exception.detail["code"] == "LLM_SERVICE_ERROR"

    def test_unknown_error_fallback(self):
        """Test fallback for unknown error types."""
        unknown_errors = [
            "Unknown error occurred",
            "Random issue",
            "Something went wrong"
        ]

        for error_msg in unknown_errors:
            exception = LLMErrorHandler.handle_llm_error(error_msg)
            assert exception.detail["code"] == "GENERATION_FAILED"
            assert exception.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestValidationErrorHandler:
    """Test ValidationErrorHandler functionality."""

    def test_empty_message_content_error(self):
        """Test empty message content validation error."""
        exception = ValidationErrorHandler.empty_message_content()

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_400_BAD_REQUEST
        assert exception.detail["code"] == "EMPTY_MESSAGE_CONTENT"
        assert exception.detail["param"] == "content"
        assert exception.detail["type"] == "invalid_request_error"

    def test_empty_message_content_custom_field(self):
        """Test empty message content with custom field."""
        exception = ValidationErrorHandler.empty_message_content("custom_field")

        assert exception.detail["param"] == "custom_field"

    def test_messages_required_error(self):
        """Test messages required validation error."""
        exception = ValidationErrorHandler.messages_required()

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_400_BAD_REQUEST
        assert exception.detail["code"] == "EMPTY_MESSAGES"
        assert exception.detail["param"] == "messages"
        assert exception.detail["type"] == "invalid_request_error"

    def test_session_id_required_error(self):
        """Test session ID required validation error."""
        exception = ValidationErrorHandler.session_id_required()

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_400_BAD_REQUEST
        assert exception.detail["code"] == "MISSING_SESSION_ID"
        assert exception.detail["param"] == "session_id"
        assert exception.detail["type"] == "invalid_request_error"

    def test_unexpected_error_with_details(self):
        """Test unexpected error with original exception details."""
        original_error = ValueError("Something went wrong")
        exception = ValidationErrorHandler.unexpected_error(original_error)

        assert isinstance(exception, HTTPException)
        assert exception.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exception.detail["code"] == "UNEXPECTED_ERROR"
        assert exception.detail["type"] == "internal_server_error"
        assert exception.detail["details"]["original_error"] == "Something went wrong"

    def test_error_message_consistency(self):
        """Test error messages match constants."""
        # Test that error messages match the constants
        empty_content = ValidationErrorHandler.empty_message_content()
        assert empty_content.detail["message"] == ErrorMessages.EMPTY_MESSAGE_CONTENT

        messages_required = ValidationErrorHandler.messages_required()
        assert messages_required.detail["message"] == ErrorMessages.MESSAGES_REQUIRED

        session_required = ValidationErrorHandler.session_id_required()
        assert session_required.detail["message"] == ErrorMessages.SESSION_ID_REQUIRED

        timeout = LLMErrorHandler.handle_llm_error("timeout error")
        assert timeout.detail["message"] == ErrorMessages.TIMEOUT

        service_error = LLMErrorHandler.handle_llm_error("service error")
        assert service_error.detail["message"] == ErrorMessages.SERVICE_ERROR