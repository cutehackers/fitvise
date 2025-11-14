"""Tests for LangChain LLM adapter Runnable integration."""

import pytest
from unittest.mock import AsyncMock, Mock, create_autospec

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

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
    provider.generate = AsyncMock(return_value="Test response from LLM")
    provider.generate_stream = AsyncMock()
    return provider


@pytest.fixture
def adapter(mock_llm_provider):
    """Create a LangChain LLM adapter with mocked provider."""
    return LangChainLLMAdapter(llm_provider=mock_llm_provider)


class TestLangChainRunnableIntegration:
    """Test suite for LangChain Runnable pipeline integration."""

    def test_adapter_is_runnable(self, adapter):
        """Test that adapter is a proper Runnable."""
        assert isinstance(adapter, Runnable)

    def test_adapter_has_runnable_methods(self, adapter):
        """Test that adapter has all required Runnable methods."""
        # These are the core Runnable interface methods
        assert hasattr(adapter, 'invoke')
        assert hasattr(adapter, 'ainvoke')
        assert hasattr(adapter, 'stream')
        assert hasattr(adapter, 'astream')
        assert hasattr(adapter, 'batch')
        assert hasattr(adapter, 'abatch')

    @pytest.mark.asyncio
    async def test_adapter_in_simple_chain(self, adapter, mock_llm_provider):
        """Test adapter works in a simple LCEL chain."""
        # Create a simple chain: prompt | llm
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}")
        ])

        chain = prompt | adapter

        # Mock the provider response
        mock_llm_provider.generate.return_value = "Hello from chain!"

        # Invoke the chain
        result = await chain.ainvoke({"input": "Hi there"})

        # Verify result
        assert result.content == "Hello from chain!"
        assert mock_llm_provider.generate.called

    @pytest.mark.asyncio
    async def test_adapter_in_complex_chain(self, adapter, mock_llm_provider):
        """Test adapter works in a complex LCEL chain with multiple steps."""
        # Create a chain with passthrough and prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are {role}"),
            ("human", "{input}")
        ])

        chain = (
            RunnablePassthrough.assign(role=lambda x: "a helpful assistant")
            | prompt
            | adapter
        )

        # Mock the provider response
        mock_llm_provider.generate.return_value = "Complex chain response"

        # Invoke the chain
        result = await chain.ainvoke({"input": "Test input"})

        # Verify result
        assert result.content == "Complex chain response"
        assert mock_llm_provider.generate.called

    @pytest.mark.asyncio
    async def test_adapter_streaming_in_chain(self, adapter, mock_llm_provider):
        """Test adapter streaming works in LCEL chain."""
        # Create a simple chain
        prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}")
        ])

        chain = prompt | adapter

        # Mock streaming response
        async def mock_stream(*args, **kwargs):
            yield "Streaming "
            yield "response "
            yield "works!"

        mock_llm_provider.generate_stream = lambda *args, **kwargs: mock_stream(*args, **kwargs)

        # Stream from the chain
        chunks = []
        async for chunk in chain.astream({"input": "Test"}):
            chunks.append(chunk.content)

        # Verify streaming worked
        assert len(chunks) == 3
        assert chunks == ["Streaming ", "response ", "works!"]

    @pytest.mark.asyncio
    async def test_adapter_with_pipe_operator(self, adapter, mock_llm_provider):
        """Test that pipe operator (|) works correctly with adapter."""
        # The pipe operator is core to LCEL
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Hello")
        ])

        # Test piping works
        chain = prompt | adapter
        assert isinstance(chain, Runnable)

        # Test it executes
        mock_llm_provider.generate.return_value = "Pipe works!"
        result = await chain.ainvoke({})
        assert result.content == "Pipe works!"

    @pytest.mark.asyncio
    async def test_adapter_batch_processing(self, adapter, mock_llm_provider):
        """Test adapter supports batch processing via abatch."""
        # Create simple chain
        prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}")
        ])
        chain = prompt | adapter

        # Mock provider to return different responses
        responses = ["Response 1", "Response 2", "Response 3"]
        mock_llm_provider.generate.side_effect = responses

        # Batch invoke
        inputs = [
            {"input": "First"},
            {"input": "Second"},
            {"input": "Third"}
        ]
        results = await chain.abatch(inputs)

        # Verify batch processing worked
        assert len(results) == 3
        assert results[0].content == "Response 1"
        assert results[1].content == "Response 2"
        assert results[2].content == "Response 3"

    @pytest.mark.asyncio
    async def test_adapter_matches_orchestrator_usage(self, adapter, mock_llm_provider):
        """Test adapter works exactly as used in LangChainOrchestrator."""
        from operator import itemgetter
        from langchain_core.prompts.chat import MessagesPlaceholder

        # Replicate the exact chain structure from orchestrator
        prompt = ChatPromptTemplate([
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])

        # Simple trimmer (no actual trimming for test) - wrap in RunnableLambda
        trimmer = RunnableLambda(lambda msgs: msgs)

        # Build the exact chain structure
        chain = (
            RunnablePassthrough.assign(
                history=itemgetter("history") | trimmer
            ) | prompt | adapter
        )

        # Mock response
        mock_llm_provider.generate.return_value = "Orchestrator-style response"

        # Test the chain
        result = await chain.ainvoke({
            "input": "Test message",
            "history": [HumanMessage(content="Previous message")]
        })

        # Verify it works
        assert result.content == "Orchestrator-style response"
        assert mock_llm_provider.generate.called
