"""Tests for LangChain LLM adapter."""

import pytest
from unittest.mock import AsyncMock, Mock, create_autospec

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from app.domain.entities.message_role import MessageRole
from app.domain.llm.entities.message import Message
from app.domain.llm.entities.model_info import ModelInfo
from app.domain.llm.interfaces.llm_provider import LLMProvider
from app.infrastructure.llm.adapters import LangChainLLMAdapter


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider with proper spec."""
    provider = create_autospec(LLMProvider, instance=True)
    provider.get_model_info = Mock(return_value=ModelInfo(
        name="test-model",
        provider="test-provider",
        max_tokens=4096,
        supports_streaming=True,
    ))
    provider.generate = AsyncMock()
    provider.generate_stream = AsyncMock()
    return provider


@pytest.fixture
def adapter(mock_llm_provider):
    """Create a LangChain LLM adapter with mocked provider."""
    return LangChainLLMAdapter(llm_provider=mock_llm_provider)


class TestLangChainLLMAdapter:
    """Test suite for LangChain LLM adapter."""

    def test_adapter_initialization(self, mock_llm_provider):
        """Test that adapter initializes correctly."""
        adapter = LangChainLLMAdapter(llm_provider=mock_llm_provider)

        assert adapter.llm_provider == mock_llm_provider
        assert adapter.model_name == "test-model"
        assert adapter._llm_type == "custom-llm-provider"

    def test_convert_messages_to_domain(self, adapter):
        """Test conversion of LangChain messages to domain messages."""
        langchain_messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        domain_messages = adapter._convert_messages_to_domain(langchain_messages)

        assert len(domain_messages) == 3
        assert domain_messages[0].role == MessageRole.SYSTEM
        assert domain_messages[0].content == "You are a helpful assistant"
        assert domain_messages[1].role == MessageRole.USER
        assert domain_messages[1].content == "Hello"
        assert domain_messages[2].role == MessageRole.ASSISTANT
        assert domain_messages[2].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_agenerate_success(self, adapter, mock_llm_provider):
        """Test successful async generation."""
        # Setup
        mock_llm_provider.generate.return_value = "Generated response"
        messages = [HumanMessage(content="Test prompt")]

        # Execute
        result = await adapter._agenerate(messages)

        # Verify
        assert len(result.generations) == 1
        assert isinstance(result.generations[0].message, AIMessage)
        assert result.generations[0].message.content == "Generated response"

        # Verify provider was called with correct domain messages
        call_args = mock_llm_provider.generate.call_args
        domain_messages = call_args[0][0]
        assert len(domain_messages) == 1
        assert domain_messages[0].role == MessageRole.USER
        assert domain_messages[0].content == "Test prompt"

    @pytest.mark.asyncio
    async def test_astream_success(self, adapter, mock_llm_provider):
        """Test successful async streaming."""
        # Setup
        async def mock_stream(*args, **kwargs):
            yield "Hello"
            yield " "
            yield "world"

        # Make the mock return the async generator directly
        mock_llm_provider.generate_stream = lambda *args, **kwargs: mock_stream(*args, **kwargs)
        messages = [HumanMessage(content="Stream test")]

        # Execute
        chunks = []
        async for chunk in adapter._astream(messages):
            chunks.append(chunk)

        # Verify
        assert len(chunks) == 3
        assert all(isinstance(chunk.message, AIMessageChunk) for chunk in chunks)
        assert chunks[0].message.content == "Hello"
        assert chunks[1].message.content == " "
        assert chunks[2].message.content == "world"

    def test_generate_not_implemented(self, adapter):
        """Test that synchronous generation raises NotImplementedError."""
        messages = [HumanMessage(content="Test")]

        with pytest.raises(NotImplementedError, match="async methods"):
            adapter._generate(messages)

    def test_stream_not_implemented(self, adapter):
        """Test that synchronous streaming raises NotImplementedError."""
        messages = [HumanMessage(content="Test")]

        with pytest.raises(NotImplementedError, match="async methods"):
            list(adapter._stream(messages))

    @pytest.mark.asyncio
    async def test_agenerate_with_callback_manager(self, adapter, mock_llm_provider):
        """Test async generation with callback manager."""
        # Setup
        mock_llm_provider.generate.return_value = "Response with callbacks"
        mock_callback = AsyncMock()
        messages = [HumanMessage(content="Test")]

        # Execute
        result = await adapter._agenerate(messages, run_manager=mock_callback)

        # Verify callback was triggered
        assert mock_callback.on_llm_new_token.called
        assert result.generations[0].message.content == "Response with callbacks"

    @pytest.mark.asyncio
    async def test_multiple_message_types(self, adapter, mock_llm_provider):
        """Test handling multiple message types in conversation."""
        # Setup
        mock_llm_provider.generate.return_value = "Complex response"
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message"),
            AIMessage(content="Assistant response"),
            HumanMessage(content="Follow-up question"),
        ]

        # Execute
        result = await adapter._agenerate(messages)

        # Verify domain messages conversion
        call_args = mock_llm_provider.generate.call_args
        domain_messages = call_args[0][0]
        assert len(domain_messages) == 4
        assert domain_messages[0].role == MessageRole.SYSTEM
        assert domain_messages[1].role == MessageRole.USER
        assert domain_messages[2].role == MessageRole.ASSISTANT
        assert domain_messages[3].role == MessageRole.USER
