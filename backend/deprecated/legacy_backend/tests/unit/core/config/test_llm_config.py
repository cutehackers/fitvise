"""Test suite for LLM configuration module."""

import pytest
from pydantic import ValidationError

from app.core.config.llm_config import LLMConfig, ChatConfig
from app.core.constants import MAX_TOKENS_TABLE, DEFAULT_MAX_TOKEN_LENGTH


class TestLLMConfig:
    """Test LLMConfig functionality."""

    def test_minimal_valid_config(self):
        """Test creating minimal valid LLM configuration."""
        config = LLMConfig(
            llm_base_url="http://localhost:11434",
            llm_model="llama3:8b"
        )

        assert config.llm_base_url == "http://localhost:11434"
        assert config.llm_model == "llama3:8b"
        assert config.llm_timeout == 30  # Default value
        assert config.llm_temperature == 0.7  # Default value
        assert config.llm_max_tokens == 2048  # Default value

    def test_complete_valid_config(self):
        """Test creating complete valid LLM configuration."""
        config = LLMConfig(
            llm_base_url="https://api.example.com",
            llm_model="llama3.1:8b",
            llm_timeout=60,
            llm_temperature=0.5,
            llm_max_tokens=4096,
            llm_max_concurrent=20,
            llm_context_window=100000,
            llm_reserve_tokens=1000,
            context_truncation_strategy="recent",
            trim_messages_threshold=25,
            health_check_interval=120,
            health_min_success_rate=98.0,
            health_max_response_time_ms=3000.0
        )

        assert config.llm_base_url == "https://api.example.com"
        assert config.llm_model == "llama3.1:8b"
        assert config.llm_timeout == 60
        assert config.llm_temperature == 0.5
        assert config.llm_max_tokens == 4096
        assert config.llm_max_concurrent == 20
        assert config.llm_context_window == 100000
        assert config.llm_reserve_tokens == 1000
        assert config.context_truncation_strategy == "recent"
        assert config.trim_messages_threshold == 25

    def test_invalid_timeout(self):
        """Test validation of invalid timeout values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_timeout=-1)

        assert "LLM timeout must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_timeout=700)

        assert "LLM timeout cannot exceed 600 seconds" in str(exc_info.value)

    def test_invalid_max_concurrent(self):
        """Test validation of invalid max concurrent values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_max_concurrent=0)

        assert "Max concurrent requests must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_max_concurrent=150)

        assert "Max concurrent requests cannot exceed 100" in str(exc_info.value)

    def test_invalid_temperature(self):
        """Test validation of invalid temperature values."""
        with pytest.raises(ValidationError):
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_temperature=-0.1)

        with pytest.raises(ValidationError):
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_temperature=2.1)

    def test_invalid_context_window(self):
        """Test validation of invalid context window values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_context_window=0)

        assert "Context window must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(llm_base_url="http://localhost", llm_model="test", llm_context_window=300000)

        assert "Context window cannot exceed 200,000 tokens" in str(exc_info.value)

    def test_effective_context_window_known_model(self):
        """Test effective context window for known models."""
        config = LLMConfig(
            llm_base_url="http://localhost",
            llm_model="llama3.2:3b",
            llm_context_window=150000  # Larger than model limit
        )

        # Should use the model's actual limit
        assert config.effective_context_window == 128000

    def test_effective_context_window_unknown_model(self):
        """Test effective context window for unknown models."""
        config = LLMConfig(
            llm_base_url="http://localhost",
            llm_model="unknown:model",
            llm_context_window=10000
        )

        # Should use the configured limit since model is unknown
        assert config.effective_context_window == 10000

    def test_max_response_tokens(self):
        """Test calculation of max response tokens."""
        config = LLMConfig(
            llm_base_url="http://localhost",
            llm_model="llama3:8b",
            llm_context_window=8192,
            llm_reserve_tokens=512
        )

        # Should be context window minus reserved tokens
        assert config.max_response_tokens == 8192 - 512

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        config = LLMConfig(
            llm_base_url="http://localhost",
            llm_model="llama3:8b",
            llm_context_window=8192,
            llm_reserve_tokens=500
        )

        # Should not raise any exception
        config.validate_configuration()

    def test_validate_configuration_reserved_too_large(self):
        """Test validation failure when reserved tokens exceed context window."""
        config = LLMConfig(
            llm_base_url="http://localhost",
            llm_model="llama3:8b",
            llm_context_window=8192,
            llm_reserve_tokens=8192  # Equal to context window
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate_configuration()

        assert "Reserved tokens cannot equal or exceed" in str(exc_info.value)

    def test_get_config_summary(self):
        """Test configuration summary generation."""
        config = LLMConfig(
            llm_base_url="https://api.example.com/v1",
            llm_model="llama3.2:3b",
            llm_timeout=60,
            llm_temperature=0.5
        )

        summary = config.get_config_summary()

        assert summary["config_type"] == "LLMConfig"
        assert summary["model"] == "llama3.2:3b"
        assert summary["base_url"] == "https"  # Only protocol shown
        assert summary["timeout_seconds"] == 60
        assert summary["temperature"] == 0.5
        assert "effective_context_window" in summary
        assert "max_response_tokens" in summary

    def test_model_name_case_insensitive_lookup(self):
        """Test that model lookup is case insensitive."""
        config1 = LLMConfig(llm_base_url="http://localhost", llm_model="LLAMA3:8B")
        config2 = LLMConfig(llm_base_url="http://localhost", llm_model="llama3:8b")

        assert config1.effective_context_window == config2.effective_context_window


class TestChatConfig:
    """Test ChatConfig functionality."""

    def test_minimal_valid_config(self):
        """Test creating minimal valid chat configuration."""
        config = ChatConfig()

        assert config.turns_window == 10  # Default value
        assert config.max_session_age_hours == 24  # Default value
        assert config.enable_streaming is True  # Default value
        assert config.batch_size == 32  # Default value
        assert config.max_retries == 3  # Default value

    def test_complete_valid_config(self):
        """Test creating complete valid chat configuration."""
        config = ChatConfig(
            turns_window=20,
            max_session_age_hours=48,
            enable_streaming=False,
            batch_size=64,
            max_retries=5
        )

        assert config.turns_window == 20
        assert config.max_session_age_hours == 48
        assert config.enable_streaming is False
        assert config.batch_size == 64
        assert config.max_retries == 5

    def test_invalid_turns_window(self):
        """Test validation of invalid turns window values."""
        with pytest.raises(ValidationError):
            ChatConfig(turns_window=0)

        with pytest.raises(ValidationError):
            ChatConfig(turns_window=-5)

        with pytest.raises(ValidationError):
            ChatConfig(turns_window=60)  # Exceeds max of 50

    def test_invalid_max_session_age(self):
        """Test validation of invalid max session age values."""
        with pytest.raises(ValidationError):
            ChatConfig(max_session_age_hours=0)

        with pytest.raises(ValidationError):
            ChatConfig(max_session_age_hours=-1)

        with pytest.raises(ValidationError):
            ChatConfig(max_session_age_hours=200)  # Exceeds max of 168

    def test_invalid_batch_size(self):
        """Test validation of invalid batch size values."""
        with pytest.raises(ValidationError):
            ChatConfig(batch_size=0)

        with pytest.raises(ValidationError):
            ChatConfig(batch_size=-1)

        with pytest.raises(ValidationError):
            ChatConfig(batch_size=150)  # Exceeds max of 128

    def test_invalid_max_retries(self):
        """Test validation of invalid max retries values."""
        with pytest.raises(ValidationError):
            ChatConfig(max_retries=-1)

        with pytest.raises(ValidationError):
            ChatConfig(max_retries=15)  # Exceeds max of 10

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        config = ChatConfig(
            turns_window=15,
            max_session_age_hours=36
        )

        # Should not raise any exception
        config.validate_configuration()

    def test_get_config_summary(self):
        """Test configuration summary generation."""
        config = ChatConfig(
            turns_window=25,
            max_session_age_hours=48,
            enable_streaming=False,
            batch_size=64,
            max_retries=5
        )

        summary = config.get_config_summary()

        assert summary["config_type"] == "ChatConfig"
        assert summary["turns_window"] == 25
        assert summary["max_session_age_hours"] == 48
        assert summary["streaming_enabled"] is False
        assert summary["batch_size"] == 64
        assert summary["max_retries"] == 5

    def test_config_validation_edge_cases(self):
        """Test configuration validation at edge cases."""
        # Test minimum valid values
        config_min = ChatConfig(
            turns_window=1,
            max_session_age_hours=1,
            batch_size=1,
            max_retries=0
        )
        config_min.validate_configuration()

        # Test maximum valid values
        config_max = ChatConfig(
            turns_window=50,
            max_session_age_hours=168,
            batch_size=128,
            max_retries=10
        )
        config_max.validate_configuration()