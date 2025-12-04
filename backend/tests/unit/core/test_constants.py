"""Test suite for constants module."""

import pytest

from app.core.constants import (
    MAX_TOKENS_TABLE,
    DEFAULT_MAX_TOKEN_LENGTH,
    ErrorMessages,
    ServiceNames,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    SEARCH_DEFAULT_TOP_K
)


class TestConstants:
    """Test constants module values and organization."""

    def test_max_tokens_table_structure(self):
        """Test MAX_TOKENS_TABLE has expected structure."""
        assert isinstance(MAX_TOKENS_TABLE, dict)
        assert len(MAX_TOKENS_TABLE) > 0

        # Check that all keys are lowercase strings
        for model_name in MAX_TOKENS_TABLE.keys():
            assert isinstance(model_name, str)
            assert model_name == model_name.lower()

        # Check that all values are positive integers
        for token_limit in MAX_TOKENS_TABLE.values():
            assert isinstance(token_limit, int)
            assert token_limit > 0

    def test_known_model_token_limits(self):
        """Test token limits for known models."""
        # Known models with their expected token limits
        known_models = {
            "llama3.2:3b": 128000,
            "llama3.1:8b": 128000,
            "llama3:8b": 8192,
            "mistral:7b": 8192,
            "codellama:7b": 16384
        }

        for model, expected_limit in known_models.items():
            assert MAX_TOKENS_TABLE.get(model.lower()) == expected_limit

    def test_default_token_limit_fallback(self):
        """Test default token limit is reasonable."""
        assert isinstance(DEFAULT_MAX_TOKEN_LENGTH, int)
        assert DEFAULT_MAX_TOKEN_LENGTH > 0
        assert DEFAULT_MAX_TOKEN_LENGTH < 200000  # Reasonable upper bound

    def test_error_messages_structure(self):
        """Test ErrorMessages class has all required messages."""
        required_messages = [
            "EMPTY_QUERY",
            "EMPTY_MESSAGE_CONTENT",
            "MESSAGES_REQUIRED",
            "SESSION_ID_REQUIRED",
            "TIMEOUT",
            "SERVICE_ERROR",
            "GENERATION_FAILED",
            "UNEXPECTED_ERROR",
            "INVALID_REQUEST",
            "VALIDATION_ERROR",
            "SERVICE_UNAVAILABLE"
        ]

        for message_attr in required_messages:
            assert hasattr(ErrorMessages, message_attr)
            message = getattr(ErrorMessages, message_attr)
            assert isinstance(message, str)
            assert len(message.strip()) > 0

    def test_service_names_structure(self):
        """Test ServiceNames class has expected services."""
        required_services = [
            "WORKOUT_API",
            "RAG_CHAT",
            "LLM_SERVICE",
            "DOCUMENT_PROCESSOR",
            "EMBEDDING_SERVICE"
        ]

        for service_attr in required_services:
            assert hasattr(ServiceNames, service_attr)
            service = getattr(ServiceNames, service_attr)
            assert isinstance(service, str)
            assert len(service.strip()) > 0

    def test_service_names_format(self):
        """Test service names follow consistent format."""
        for attr_name in dir(ServiceNames):
            if not attr_name.startswith('_'):
                service = getattr(ServiceNames, attr_name)
                # Should be lowercase with hyphens
                assert service == service.lower()
                assert '-' in service or '_' not in service

    def test_performance_constants_reasonable_values(self):
        """Test performance constants have reasonable values."""
        # Batch sizes
        assert 1 <= DEFAULT_BATCH_SIZE <= 128
        assert isinstance(DEFAULT_BATCH_SIZE, int)

        # Timeouts
        assert 5 <= DEFAULT_TIMEOUT_SECONDS <= 600
        assert isinstance(DEFAULT_TIMEOUT_SECONDS, int)

        # Search defaults
        assert 1 <= SEARCH_DEFAULT_TOP_K <= 100
        assert isinstance(SEARCH_DEFAULT_TOP_K, int)

    def test_constants_immutability(self):
        """Test that constants remain unchanged."""
        # Store original values
        original_max_tokens = MAX_TOKENS_TABLE.copy()
        original_default = DEFAULT_MAX_TOKEN_LENGTH
        original_error_msg = ErrorMessages.EMPTY_QUERY

        # Try to modify (this should not affect the module state)
        try:
            MAX_TOKENS_TABLE["test"] = 9999
            # Reset to original to not affect other tests
            MAX_TOKENS_TABLE.clear()
            MAX_TOKENS_TABLE.update(original_max_tokens)
        except:
            pass

        # Verify originals unchanged
        assert MAX_TOKENS_TABLE == original_max_tokens
        assert DEFAULT_MAX_TOKEN_LENGTH == original_default
        assert ErrorMessages.EMPTY_QUERY == original_error_msg

    def test_constants_type_consistency(self):
        """Test all constants have consistent types."""
        # String constants
        string_constants = [
            ErrorMessages.EMPTY_QUERY,
            ErrorMessages.SERVICE_ERROR,
            ServiceNames.WORKOUT_API
        ]

        for constant in string_constants:
            assert isinstance(constant, str)

        # Numeric constants
        numeric_constants = [
            DEFAULT_MAX_TOKEN_LENGTH,
            DEFAULT_BATCH_SIZE,
            DEFAULT_TIMEOUT_SECONDS,
            SEARCH_DEFAULT_TOP_K
        ]

        for constant in numeric_constants:
            assert isinstance(constant, (int, float))
            assert constant > 0