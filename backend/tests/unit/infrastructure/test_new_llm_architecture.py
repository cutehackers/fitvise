"""Tests for the new LLM architecture."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from app.domain.llm.entities.message import Message
from app.domain.llm.entities.model_info import ModelInfo
from app.domain.llm.entities.session import ChatSession
from app.domain.llm.exceptions import LLMProviderError, ChatOrchestratorError
from app.domain.entities.message_role import MessageRole
from app.infrastructure.llm.providers.ollama_provider import OllamaProvider
from app.infrastructure.llm.services.langchain_orchestrator import LangChainOrchestrator
from app.infrastructure.llm.dependencies import get_llm_provider, get_chat_orchestrator
from app.core.settings import Settings


class TestOllamaProvider:
    """Test OllamaProvider implementation."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.llm_base_url = "http://localhost:11434"
        settings.llm_model = "llama3.2:3b"
        settings.llm_temperature = 0.7
        return settings

    @pytest.fixture
    def mock_chat_ollama(self) -> AsyncMock:
        """Create mock ChatOllama."""
        mock_ollama = AsyncMock()
        mock_ollama.model = "llama3.2:3b"
        return mock_ollama

    @pytest.fixture
    def ollama_provider(self, mock_settings: Settings) -> OllamaProvider:
        """Create OllamaProvider instance."""
        with patch('app.infrastructure.llm.providers.ollama_provider.ChatOllama'):
            return OllamaProvider(mock_settings)

    def test_initialization(self, ollama_provider: OllamaProvider, mock_settings: Settings):
        """Test provider initialization."""
        assert ollama_provider.provider_name == "ollama"
        model_info = ollama_provider.get_model_info()
        assert model_info.name == mock_settings.llm_model
        assert model_info.provider == "ollama"
        assert model_info.supports_streaming is True

    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_provider: OllamaProvider):
        """Test successful generation."""
        # Mock ChatOllama response
        with patch.object(ollama_provider._llm, 'ainvoke') as mock_invoke:
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_invoke.return_value = mock_response

            messages = [Message(content="Hello", role=MessageRole.USER)]
            result = await ollama_provider.generate(messages)

            assert result == "Test response"
            mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_failure(self, ollama_provider: OllamaProvider):
        """Test generation failure."""
        with patch.object(ollama_provider._llm, 'ainvoke', side_effect=Exception("API Error")):
            messages = [Message(content="Hello", role=MessageRole.USER)]

            with pytest.raises(LLMProviderError) as exc_info:
                await ollama_provider.generate(messages)

            assert "Ollama generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, ollama_provider: OllamaProvider):
        """Test successful streaming generation."""
        with patch.object(ollama_provider._llm, 'astream') as mock_stream:
            mock_chunk1 = MagicMock()
            mock_chunk1.content = "Hello "
            mock_chunk2 = MagicMock()
            mock_chunk2.content = "world!"

            mock_stream.return_value = [mock_chunk1, mock_chunk2]

            messages = [Message(content="Hello", role=MessageRole.USER)]
            chunks = []
            async for chunk in ollama_provider.generate_stream(messages):
                chunks.append(chunk)

            assert chunks == ["Hello ", "world!"]
            mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_provider: OllamaProvider):
        """Test successful health check."""
        with patch.object(ollama_provider._llm, 'ainvoke') as mock_invoke:
            mock_invoke.return_value = MagicMock(content="pong")

            result = await ollama_provider.health_check()
            assert result is True
            mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_provider: OllamaProvider):
        """Test health check failure."""
        with patch.object(ollama_provider._llm, 'ainvoke', side_effect=Exception("Connection error")):
            result = await ollama_provider.health_check()
            assert result is False


class TestLangChainOrchestrator:
    """Test LangChainOrchestrator implementation."""

    @pytest.fixture
    def mock_llm_provider(self) -> AsyncMock:
        """Create mock LLM provider."""
        provider = AsyncMock(spec=OllamaProvider)
        provider.get_model_info.return_value = ModelInfo(
            name="llama3.2:3b",
            provider="ollama",
            max_tokens=128000,
        )
        provider.health_check.return_value = True
        return provider

    @pytest.fixture
    def chat_orchestrator(self, mock_llm_provider: AsyncMock) -> LangChainOrchestrator:
        """Create chat orchestrator instance."""
        return LangChainOrchestrator(
            llm_provider=mock_llm_provider,
            max_session_age_hours=24,
            max_history_length=50,
        )

    @pytest.mark.asyncio
    async def test_chat_success(self, chat_orchestrator: LangChainOrchestrator):
        """Test successful message processing."""
        from app.schemas.chat import ChatRequest, ChatMessage

        # Mock the chain response
        with patch.object(chat_orchestrator, '_chain') as mock_chain:
            mock_response = MagicMock()
            mock_response.content = "Hello! How can I help you?"
            mock_chain.astream.return_value = [mock_response]

            request = ChatRequest(
                message=ChatMessage(role="user", content="Hello"),
                session_id="test-session",
            )

            responses = []
            async for response in chat_orchestrator.chat(request):
                responses.append(response)

            assert len(responses) >= 1
            assert any(r.done for r in responses)
            # Verify session was created
            session_count = await chat_orchestrator.get_active_session_count()
            assert session_count == 1

    @pytest.mark.asyncio
    async def test_get_session_history(self, chat_orchestrator: LangChainOrchestrator):
        """Test getting session history."""
        from app.schemas.chat import ChatRequest, ChatMessage

        # First, add a message to create a session
        request = ChatRequest(
            message=ChatMessage(role="user", content="Hello"),
            session_id="test-session",
        )

        # Mock the chain to avoid actual LLM calls
        with patch.object(chat_orchestrator, '_chain'):
            async for _ in chat_orchestrator.chat(request):
                break  # Just create the session

        # Now get history
        history = await chat_orchestrator.get_session_history("test-session")
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_clear_session(self, chat_orchestrator: LangChainOrchestrator):
        """Test clearing a session."""
        from app.schemas.chat import ChatRequest, ChatMessage

        # Create a session first
        request = ChatRequest(
            message=ChatMessage(role="user", content="Hello"),
            session_id="test-session",
        )

        with patch.object(chat_orchestrator, '_chain'):
            async for _ in chat_orchestrator.chat(request):
                break

        # Verify session exists
        session_count_before = await chat_orchestrator.get_active_session_count()
        assert session_count_before == 1

        # Clear session
        result = await chat_orchestrator.clear_session("test-session")
        assert result is True

        # Try to clear non-existent session
        result = await chat_orchestrator.clear_session("non-existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check(self, chat_orchestrator: LangChainOrchestrator):
        """Test health check."""
        result = await chat_orchestrator.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_provider_failure(self, chat_orchestrator: LangChainOrchestrator):
        """Test health check when provider fails."""
        chat_orchestrator._llm_provider.health_check.return_value = False
        result = await chat_orchestrator.health_check()
        assert result is False


class TestDependencyInjection:
    """Test dependency injection functions."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.llm_base_url = "http://localhost:11434"
        settings.llm_model = "llama3.2:3b"
        settings.llm_temperature = 0.7
        settings.chat_max_session_age_hours = 24
        settings.chat_max_history_length = 50
        return settings

    def test_get_llm_provider(self, mock_settings: Settings):
        """Test LLM provider dependency injection."""
        with patch('app.infrastructure.llm.dependencies.settings', mock_settings):
            with patch('app.infrastructure.llm.providers.ollama_provider.ChatOllama'):
                provider = get_llm_provider()
                assert isinstance(provider, OllamaProvider)
                assert provider.provider_name == "ollama"

    def test_get_chat_orchestrator(self, mock_settings: Settings):
        """Test chat orchestrator dependency injection."""
        with patch('app.infrastructure.llm.dependencies.settings', mock_settings):
            with patch('app.infrastructure.llm.providers.ollama_provider.ChatOllama'):
                orchestrator = get_chat_orchestrator()
                assert isinstance(orchestrator, LangChainOrchestrator)


class TestIntegration:
    """Integration tests for the new architecture."""

    @pytest.mark.asyncio
    async def test_full_chat_flow(self):
        """Test complete chat flow from provider to orchestrator."""
        # This test would require actual Ollama instance or more sophisticated mocking
        # For now, we'll test the integration points

        mock_settings = MagicMock(spec=Settings)
        mock_settings.llm_base_url = "http://localhost:11434"
        mock_settings.llm_model = "llama3.2:3b"
        mock_settings.llm_temperature = 0.7

        with patch('app.infrastructure.llm.providers.ollama_provider.ChatOllama'):
            provider = OllamaProvider(mock_settings)

            # Test that we can create an orchestrator with the provider
            orchestrator = LangChainOrchestrator(
                llm_provider=provider,
                max_session_age_hours=24,
                max_history_length=50,
            )

            # Test health checks
            with patch.object(provider._llm, 'ainvoke', return_value=MagicMock(content="pong")):
                provider_healthy = await provider.health_check()
                orchestrator_healthy = await orchestrator.health_check()

                # With mocked responses, both should be healthy
                assert provider_healthy is True
                assert orchestrator_healthy is True

            # Test model info
            model_info = provider.get_model_info()
            assert model_info.provider == "ollama"
            assert model_info.supports_streaming is True