"""
Unit tests for configuration management.

Tests the Settings class and configuration loading functionality.
"""

import os
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tests.utils.test_helpers import test_env


class TestSettings:
    """Test the Settings configuration class."""

    def test_settings_initialization_with_defaults(self):
        """Test Settings initialization with default values."""
        # Setup test environment
        test_env_vars = {
            "APP_NAME": "Test App",
            "APP_VERSION": "1.0.0",
            "ENVIRONMENT": "test",
            "SECRET_KEY": "test-secret-key-minimum-32-characters",
        }
        original_env = test_env.setup_test_env_vars(test_env_vars)

        try:
            settings = Settings()

            assert settings.app_name == "Test App"
            assert settings.app_version == "1.0.0"
            assert settings.environment == "test"
            assert settings.debug is True  # Should default to True in test environment

        finally:
            test_env.restore_env_vars(original_env)

    def test_settings_required_fields_validation(self):
        """Test that required fields raise validation errors when missing."""
        # Clear environment variables
        required_vars = ["APP_NAME", "SECRET_KEY"]
        original_values = {}

        for var in required_vars:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        try:
            with pytest.raises(ValidationError):
                Settings()

        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_settings_type_validation(self):
        """Test that settings validate data types correctly."""
        test_env_vars = {
            "APP_NAME": "Test App",
            "SECRET_KEY": "test-secret-key-minimum-32-characters",
            "API_PORT": "invalid_port",  # Should be integer
            "DEBUG": "not_a_boolean",  # Should be boolean
        }
        original_env = test_env.setup_test_env_vars(test_env_vars)

        try:
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            # Check that validation errors mention the problematic fields
            error_str = str(exc_info.value)
            assert "api_port" in error_str or "API_PORT" in error_str

        finally:
            test_env.restore_env_vars(original_env)

    def test_settings_port_range_validation(self):
        """Test that port numbers are validated within correct ranges."""
        test_cases = [
            {"API_PORT": "0", "should_fail": True},  # Too low
            {"API_PORT": "65536", "should_fail": True},  # Too high
            {"API_PORT": "8000", "should_fail": False},  # Valid
            {"API_PORT": "1", "should_fail": False},  # Edge case valid
            {"API_PORT": "65535", "should_fail": False},  # Edge case valid
        ]

        base_env = {
            "APP_NAME": "Test App",
            "SECRET_KEY": "test-secret-key-minimum-32-characters",
        }

        for test_case in test_cases:
            env_vars = {**base_env, **test_case}
            original_env = test_env.setup_test_env_vars(env_vars)

            try:
                if test_case["should_fail"]:
                    with pytest.raises(ValidationError):
                        Settings()
                else:
                    settings = Settings()
                    assert settings.api_port == int(test_case["API_PORT"])

            finally:
                test_env.restore_env_vars(original_env)

    def test_settings_secret_key_length_validation(self):
        """Test that secret key meets minimum length requirements."""
        test_cases = [
            {"SECRET_KEY": "short", "should_fail": True},  # Too short
            {
                "SECRET_KEY": "exactly-32-characters-long!!",
                "should_fail": False,
            },  # Exactly 32
            {
                "SECRET_KEY": "much-longer-than-32-characters-should-work",
                "should_fail": False,
            },  # Longer
        ]

        base_env = {"APP_NAME": "Test App"}

        for test_case in test_cases:
            env_vars = {**base_env, **test_case}
            original_env = test_env.setup_test_env_vars(env_vars)

            try:
                if test_case["should_fail"]:
                    with pytest.raises(ValidationError):
                        Settings()
                else:
                    settings = Settings()
                    assert len(settings.secret_key) >= 32

            finally:
                test_env.restore_env_vars(original_env)

    def test_cors_origins_list_property(self):
        """Test that CORS origins string is properly converted to list."""
        test_env_vars = {
            "APP_NAME": "Test App",
            "SECRET_KEY": "test-secret-key-minimum-32-characters",
            "CORS_ORIGINS": "http://localhost:3000,https://example.com,https://app.fitvise.com",
        }
        original_env = test_env.setup_test_env_vars(test_env_vars)

        try:
            settings = Settings()
            origins_list = settings.cors_origins_list

            assert isinstance(origins_list, list)
            assert len(origins_list) == 3
            assert "http://localhost:3000" in origins_list
            assert "https://example.com" in origins_list
            assert "https://app.fitvise.com" in origins_list

        finally:
            test_env.restore_env_vars(original_env)

    def test_allowed_file_types_list_property(self):
        """Test that allowed file types string is properly converted to list."""
        test_env_vars = {
            "APP_NAME": "Test App",
            "SECRET_KEY": "test-secret-key-minimum-32-characters",
            "ALLOWED_FILE_TYPES": "jpg,png,gif,pdf,txt",
        }
        original_env = test_env.setup_test_env_vars(test_env_vars)

        try:
            settings = Settings()
            file_types_list = settings.allowed_file_types_list

            assert isinstance(file_types_list, list)
            assert len(file_types_list) == 5
            assert "jpg" in file_types_list
            assert "pdf" in file_types_list

        finally:
            test_env.restore_env_vars(original_env)

    def test_environment_specific_debug_setting(self):
        """Test that debug setting varies correctly by environment."""
        environments = [
            {"ENVIRONMENT": "production", "expected_debug": False},
            {"ENVIRONMENT": "staging", "expected_debug": False},
            {"ENVIRONMENT": "local", "expected_debug": True},
            {"ENVIRONMENT": "test", "expected_debug": True},
        ]

        base_env = {
            "APP_NAME": "Test App",
            "SECRET_KEY": "test-secret-key-minimum-32-characters",
        }

        for env_case in environments:
            env_vars = {**base_env, **env_case}
            original_env = test_env.setup_test_env_vars(env_vars)

            try:
                settings = Settings()
                assert settings.debug == env_case["expected_debug"]

            finally:
                test_env.restore_env_vars(original_env)
