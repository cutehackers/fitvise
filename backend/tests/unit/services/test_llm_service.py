"""
Unit tests for LLM service functionality.

Tests the LlmService class methods in isolation using mocks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.application.llm_service import LlmService
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse


class TestLlmService:
    """Test the LlmService class."""

    @pytest.fixture
    def llm_service(self):
        """Create an LlmService instance for testing."""
        return LlmService()

    # Validation tests
    @pytest.mark.asyncio
    async def test_chat_validation_missing_message(self, llm_service):
        """Test chat validation when message is missing."""
        request = ChatRequest(message=None, session_id="test_session_123")

        with pytest.raises(ValueError, match="Message is required in the request"):
            async for _ in llm_service.chat(request):
                pass

    @pytest.mark.asyncio
    async def test_chat_validation_empty_message_content(self, llm_service):
        """Test chat validation when message content is empty."""
        request = ChatRequest(message=ChatMessage(role="user", content=""), session_id="test_session_123")

        with pytest.raises(ValueError, match="Message content cannot be empty"):
            async for _ in llm_service.chat(request):
                pass

    @pytest.mark.asyncio
    async def test_chat_validation_whitespace_only_content(self, llm_service):
        """Test chat validation when message content is only whitespace."""
        request = ChatRequest(
            message=ChatMessage(role="user", content="   \n\t  "),
            session_id="test_session_123",
        )

        with pytest.raises(ValueError, match="Message content cannot be empty"):
            async for _ in llm_service.chat(request):
                pass

    @pytest.mark.asyncio
    async def test_chat_validation_missing_session_id(self, llm_service):
        """Test chat validation when session_id is missing."""
        request = ChatRequest(message=ChatMessage(role="user", content="Hello"), session_id="")

        with pytest.raises(ValueError, match="Session ID is required for chat history management"):
            async for _ in llm_service.chat(request):
                pass

    # Session management tests
    def test_get_session_history_creates_new_session(self, llm_service):
        """Test that get_session_history creates a new session if it doesn't exist."""
        session_id = "new_session_123"

        # Ensure session doesn't exist initially
        assert session_id not in llm_service.session_store

        # Get session history - should create new session
        history = llm_service.get_session_history(session_id)

        # Verify session was created
        assert session_id in llm_service.session_store
        assert history is not None

    def test_get_session_history_reuses_existing_session(self, llm_service):
        """Test that get_session_history reuses existing session."""
        session_id = "existing_session_123"

        # Create session first
        first_history = llm_service.get_session_history(session_id)

        # Get session history again
        second_history = llm_service.get_session_history(session_id)

        # Should be the same instance
        assert first_history is second_history

    def test_get_session_history_invalid_session_id(self, llm_service):
        """Test that get_session_history raises error for invalid session ID."""
        with pytest.raises(ValueError, match="Session ID cannot be empty or None"):
            llm_service.get_session_history("")

        with pytest.raises(ValueError, match="Session ID cannot be empty or None"):
            llm_service.get_session_history("   ")

    def test_clear_session(self, llm_service):
        """Test clearing a specific session."""
        session_id = "session_to_clear"

        # Create session
        llm_service.get_session_history(session_id)
        assert session_id in llm_service.session_store

        # Clear session
        result = llm_service.clear_session(session_id)

        # Verify session was cleared
        assert result is True
        assert session_id not in llm_service.session_store

    def test_clear_nonexistent_session(self, llm_service):
        """Test clearing a session that doesn't exist."""
        result = llm_service.clear_session("nonexistent_session")
        assert result is False

    def test_clear_all_sessions(self, llm_service):
        """Test clearing all sessions."""
        # Create multiple sessions
        llm_service.get_session_history("session1")
        llm_service.get_session_history("session2")
        llm_service.get_session_history("session3")

        assert llm_service.get_session_count() == 3

        # Clear all sessions
        cleared_count = llm_service.clear_all_sessions()

        # Verify all sessions were cleared
        assert cleared_count == 3
        assert llm_service.get_session_count() == 0

    def test_get_session_count(self, llm_service):
        """Test getting session count."""
        assert llm_service.get_session_count() == 0

        # Add sessions
        llm_service.get_session_history("session1")
        assert llm_service.get_session_count() == 1

        llm_service.get_session_history("session2")
        assert llm_service.get_session_count() == 2

    # Chat functionality tests
    @pytest.mark.asyncio
    async def test_chat_success_mock(self, llm_service):
        """Test successful chat with mocked LLM."""
        request = ChatRequest(
            message=ChatMessage(role="user", content="Hello, how are you?"),
            session_id="test_session_123",
        )

        # Mock the LLM chain to return a simple message
        mock_ai_message = AIMessage(content="I'm doing well, thank you!")

        with patch.object(llm_service, "chain") as mock_chain:
            # Mock the chain's astream method
            async def mock_astream(*args, **kwargs):
                yield mock_ai_message

            mock_chain_with_history = MagicMock()
            mock_chain_with_history.astream = mock_astream

            with patch("app.application.llm_service.RunnableWithMessageHistory") as mock_runnable:
                mock_runnable.return_value = mock_chain_with_history

                responses = []
                async for response in llm_service.chat(request):
                    responses.append(response)

                # Should get at least 2 responses: content + done
                assert len(responses) >= 1

                # Check content response
                content_response = responses[0]
                assert isinstance(content_response, ChatResponse)
                assert content_response.message.content == "I'm doing well, thank you!"
                assert not content_response.done

                # Check final response
                final_response = responses[-1]
                assert final_response.done

    @pytest.mark.asyncio
    async def test_chat_streaming_error_handling(self, llm_service):
        """Test chat error handling during streaming."""
        request = ChatRequest(
            message=ChatMessage(role="user", content="Hello"),
            session_id="test_session_123",
        )

        with patch("app.application.llm_service.RunnableWithMessageHistory") as mock_runnable:
            # Mock the chain to raise an exception during streaming
            mock_chain = MagicMock()
            mock_chain.astream.side_effect = Exception("Streaming failed")
            mock_runnable.return_value = mock_chain

            responses = []
            async for response in llm_service.chat(request):
                responses.append(response)

            # Should get error response
            assert len(responses) == 1
            error_response = responses[0]
            assert error_response.done
            assert not error_response.success
            assert "Chat streaming failed" in error_response.error

    # Health check tests
    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service):
        """Test successful health check."""
        with patch.object(llm_service.llm, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = "Health check response"

            result = await llm_service.health()

            assert result is True
            mock_ainvoke.assert_called_once_with("Health check")

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service):
        """Test health check with service failure."""
        with patch.object(llm_service.llm, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.side_effect = Exception("LLM service unavailable")

            result = await llm_service.health()

            assert result is False

    # Test the _parse_chat_stream_chunk method
    def test_parse_chat_stream_chunk(self, llm_service):
        """Test parsing a chat stream chunk."""
        mock_message = AIMessage(content="Test response content")

        result = llm_service._parse_chat_stream_chunk(mock_message)

        assert isinstance(result, ChatResponse)
        assert result.message.content == "Test response content"
        assert result.message.role == "assistant"
        assert not result.done
        assert result.model == llm_service.llm.model
