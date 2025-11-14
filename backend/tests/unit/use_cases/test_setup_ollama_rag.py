"""Unit tests for SetupOllamaRagUseCase (Task 3.1.1)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.application.use_cases.llm_infrastructure.setup_ollama_rag import (
    SetupOllamaRagUseCase,
)


class TestSetupOllamaRagUseCase:
    """Test SetupOllamaRagUseCase RAG pipeline orchestration."""

    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock OllamaService."""
        mock = MagicMock()
        mock._llm = MagicMock()
        mock._model = "llama3.2:3b"
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create mock WeaviateLangChainRetriever."""
        mock = MagicMock()
        mock.top_k = 5
        return mock

    @pytest.fixture
    def mock_context_manager(self):
        """Create mock ContextWindowManager."""
        return MagicMock()

    @pytest.fixture
    def rag_use_case(
        self, mock_ollama_service, mock_retriever, mock_context_manager
    ):
        """Create SetupOllamaRagUseCase with mocks."""
        return SetupOllamaRagUseCase(
            ollama_service=mock_ollama_service,
            retriever=mock_retriever,
            context_manager=mock_context_manager,
        )

    @pytest.mark.asyncio
    async def test_execute_rag_query_success(
        self,
        rag_use_case,
        mock_retriever,
        mock_context_manager,
        mock_ollama_service,
    ):
        """Test successful RAG query execution."""
        # Mock retriever results
        mock_docs = [
            Document(
                page_content="Strength training builds muscle",
                metadata={
                    "chunk_id": "1",
                    "document_id": "doc_1",
                    "similarity_score": 0.92,
                },
            ),
            Document(
                page_content="Cardio improves endurance",
                metadata={
                    "chunk_id": "2",
                    "document_id": "doc_2",
                    "similarity_score": 0.85,
                },
            ),
        ]
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=mock_docs)

        # Mock context window fitting
        mock_context_manager.fit_to_window = MagicMock(
            return_value="Strength training builds muscle\n\nCardio improves endurance"
        )

        # Mock LLM response
        with patch(
            "app.application.use_cases.llm_infrastructure.setup_ollama_rag.create_stuff_documents_chain"
        ) as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(
                return_value="Based on the context [1][2], strength training and cardio are important."
            )
            mock_create_chain.return_value = mock_chain

            # Execute
            response, sources = await rag_use_case.execute_rag_query(
                query="What are the benefits of exercise?", top_k=5
            )

        # Verify response
        assert isinstance(response, str)
        assert "strength training" in response.lower()

        # Verify sources
        assert len(sources) == 2
        assert sources[0].page_content == "Strength training builds muscle"
        assert sources[1].page_content == "Cardio improves endurance"

        # Verify retriever called with correct top_k
        assert mock_retriever.top_k == 5
        mock_retriever.aget_relevant_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_rag_query_empty_query(self, rag_use_case):
        """Test RAG query with empty query string."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await rag_use_case.execute_rag_query(query="")

    @pytest.mark.asyncio
    async def test_execute_rag_query_whitespace_query(self, rag_use_case):
        """Test RAG query with whitespace-only query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await rag_use_case.execute_rag_query(query="   \n\t  ")

    @pytest.mark.asyncio
    async def test_execute_rag_query_no_documents(
        self, rag_use_case, mock_retriever, mock_context_manager
    ):
        """Test RAG query when no documents are retrieved."""
        # Mock empty retrieval
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=[])

        # Execute
        response, sources = await rag_use_case.execute_rag_query(
            query="obscure query", top_k=5
        )

        # Verify fallback response
        assert "couldn't find" in response.lower()
        assert sources == []

        # Verify context manager not called when no docs
        mock_context_manager.fit_to_window.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_rag_query_context_fitting(
        self,
        rag_use_case,
        mock_retriever,
        mock_context_manager,
        mock_ollama_service,
    ):
        """Test that context window manager is called correctly."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Content",
                metadata={"chunk_id": "1", "similarity_score": 0.9},
            )
        ]
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=mock_docs)

        # Mock context fitting
        mock_context_manager.fit_to_window = MagicMock(return_value="Fitted context")

        # Mock chain
        with patch(
            "app.application.use_cases.llm_infrastructure.setup_ollama_rag.create_stuff_documents_chain"
        ) as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="Response")
            mock_create_chain.return_value = mock_chain

            # Execute
            await rag_use_case.execute_rag_query(query="test query")

        # Verify context manager called with correct args
        mock_context_manager.fit_to_window.assert_called_once()
        call_args = mock_context_manager.fit_to_window.call_args
        assert call_args[1]["documents"] == mock_docs
        assert call_args[1]["user_query"] == "test query"
        assert "system_prompt" in call_args[1]

    @pytest.mark.asyncio
    async def test_execute_rag_stream_success(
        self,
        rag_use_case,
        mock_retriever,
        mock_context_manager,
        mock_ollama_service,
    ):
        """Test successful streaming RAG query."""
        # Mock retriever
        mock_docs = [
            Document(
                page_content="Exercise content",
                metadata={"chunk_id": "1", "similarity_score": 0.9},
            )
        ]
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=mock_docs)

        # Mock context fitting
        mock_context_manager.fit_to_window = MagicMock(return_value="Exercise content")

        # Mock streaming chain
        async def mock_astream(*args, **kwargs):
            chunks = ["Exercise ", "is ", "beneficial."]
            for chunk in chunks:
                yield chunk

        with patch(
            "app.application.use_cases.llm_infrastructure.setup_ollama_rag.create_stuff_documents_chain"
        ) as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.astream = mock_astream
            mock_create_chain.return_value = mock_chain

            # Execute
            stream_gen, sources = await rag_use_case.execute_rag_stream(
                query="fitness tips"
            )

            # Collect chunks
            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)

        # Verify
        assert chunks == ["Exercise ", "is ", "beneficial."]
        assert len(sources) == 1
        assert sources[0].page_content == "Exercise content"

    @pytest.mark.asyncio
    async def test_execute_rag_stream_empty_query(self, rag_use_case):
        """Test streaming with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await rag_use_case.execute_rag_stream(query="")

    @pytest.mark.asyncio
    async def test_execute_rag_stream_no_documents(self, rag_use_case, mock_retriever):
        """Test streaming when no documents retrieved."""
        # Mock empty retrieval
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=[])

        # Execute
        stream_gen, sources = await rag_use_case.execute_rag_stream(query="query")

        # Collect stream
        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        # Verify fallback message
        full_response = "".join(chunks)
        assert "couldn't find" in full_response.lower()
        assert sources == []

    @pytest.mark.asyncio
    async def test_execute_rag_stream_error_handling(
        self, rag_use_case, mock_retriever, mock_context_manager
    ):
        """Test error handling in streaming."""
        # Mock retriever
        mock_docs = [
            Document(page_content="Content", metadata={"chunk_id": "1"})
        ]
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=mock_docs)
        mock_context_manager.fit_to_window = MagicMock(return_value="Context")

        # Mock chain that raises error
        async def mock_astream_error(*args, **kwargs):
            raise Exception("LLM generation failed")
            yield  # pragma: no cover

        with patch(
            "app.application.use_cases.llm_infrastructure.setup_ollama_rag.create_stuff_documents_chain"
        ) as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.astream = mock_astream_error
            mock_create_chain.return_value = mock_chain

            # Execute
            stream_gen, sources = await rag_use_case.execute_rag_stream(query="query")

            # Collect stream - should include error
            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)

        # Verify error message in stream
        full_response = "".join(chunks)
        assert "Error:" in full_response or "failed" in full_response.lower()

    @pytest.mark.asyncio
    async def test_session_id_parameter(
        self,
        rag_use_case,
        mock_retriever,
        mock_context_manager,
        mock_ollama_service,
    ):
        """Test that session_id parameter is accepted."""
        # Mock components
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=[])

        # Execute with session_id
        response, sources = await rag_use_case.execute_rag_query(
            query="test", top_k=5, session_id="session-123"
        )

        # Should not raise error
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_custom_top_k(
        self,
        rag_use_case,
        mock_retriever,
        mock_context_manager,
        mock_ollama_service,
    ):
        """Test custom top_k parameter."""
        # Mock retriever
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=[])

        # Execute with custom top_k
        await rag_use_case.execute_rag_query(query="test", top_k=10)

        # Verify top_k was set
        assert mock_retriever.top_k == 10

    def test_get_retriever(self, rag_use_case, mock_retriever):
        """Test get_retriever returns correct instance."""
        retriever = rag_use_case.get_retriever()
        assert retriever == mock_retriever

    def test_get_context_manager(self, rag_use_case, mock_context_manager):
        """Test get_context_manager returns correct instance."""
        manager = rag_use_case.get_context_manager()
        assert manager == mock_context_manager

    def test_system_prompt_format(self, rag_use_case):
        """Test that SYSTEM_PROMPT has correct structure."""
        prompt = rag_use_case.SYSTEM_PROMPT

        # Verify prompt contains key elements
        assert "helpful" in prompt.lower()
        assert "cite" in prompt.lower() or "citation" in prompt.lower()
        assert "{context}" in prompt
        assert "[1]" in prompt or "citation" in prompt.lower()

    @pytest.mark.asyncio
    async def test_query_trimming(
        self,
        rag_use_case,
        mock_retriever,
        mock_context_manager,
        mock_ollama_service,
    ):
        """Test that query is trimmed before processing."""
        # Mock components
        mock_docs = [Document(page_content="Content", metadata={"chunk_id": "1"})]
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=mock_docs)
        mock_context_manager.fit_to_window = MagicMock(return_value="Context")

        with patch(
            "app.application.use_cases.llm_infrastructure.setup_ollama_rag.create_stuff_documents_chain"
        ) as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="Response")
            mock_create_chain.return_value = mock_chain

            # Execute with whitespace
            await rag_use_case.execute_rag_query(query="  test query  \n")

        # Verify context manager received trimmed query
        call_args = mock_context_manager.fit_to_window.call_args
        assert call_args[1]["user_query"] == "test query"

    @pytest.mark.asyncio
    async def test_retrieval_exception_propagated(
        self, rag_use_case, mock_retriever
    ):
        """Test that retrieval exceptions are propagated."""
        # Mock retriever error
        mock_retriever.aget_relevant_documents = AsyncMock(
            side_effect=Exception("Weaviate error")
        )

        # Verify exception is raised
        with pytest.raises(Exception, match="Weaviate error"):
            await rag_use_case.execute_rag_query(query="test")

    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks(
        self, rag_use_case, mock_retriever, mock_context_manager
    ):
        """Test streaming filters empty chunks."""
        # Mock retriever
        mock_docs = [Document(page_content="Content", metadata={"chunk_id": "1"})]
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=mock_docs)
        mock_context_manager.fit_to_window = MagicMock(return_value="Context")

        # Mock chain with empty chunks
        async def mock_astream(*args, **kwargs):
            chunks = ["Valid", "", None, "Text"]
            for chunk in chunks:
                yield chunk

        with patch(
            "app.application.use_cases.llm_infrastructure.setup_ollama_rag.create_stuff_documents_chain"
        ) as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.astream = mock_astream
            mock_create_chain.return_value = mock_chain

            # Execute
            stream_gen, _ = await rag_use_case.execute_rag_stream(query="test")

            # Collect chunks
            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)

        # Verify only valid chunks included
        assert "Valid" in chunks
        assert "Text" in chunks
        assert "" not in chunks
        assert None not in chunks
