"""Unit tests for RagOrchestrator."""

import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import AIMessage

from app.infrastructure.llm.services.rag_orchestrator import RagOrchestrator
from app.infrastructure.external_services.context_management.context_window_manager import (
    ContextWindowManager,
    ContextWindow,
)
from app.domain.llm.interfaces.llm_service import LLMService
from app.schemas.chat import ChatRequest, ChatMessage
from app.domain.entities.message_role import MessageRole


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    mock_service = AsyncMock(spec=LLMService)
    mock_service.get_model_spec.return_value = MagicMock(name="llama3.2:3b")

    # Create a mock LLM instance
    mock_llm = AsyncMock()
    mock_llm.get_num_tokens_from_messages = MagicMock(return_value=100)
    mock_llm.astream = AsyncMock()

    # Create an async generator for astream
    async def mock_astream(*args, **kwargs):
        yield AIMessage(content="This is a test response")

    mock_llm.astream = mock_astream

    mock_service.llm_instance = mock_llm
    mock_service.health_check = AsyncMock(return_value=True)

    return mock_service


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    mock_retr = AsyncMock(spec=BaseRetriever)

    # Create test documents
    docs = [
        Document(
            page_content="Test content 1",
            metadata={
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "_distance": 0.9,
            },
        ),
        Document(
            page_content="Test content 2",
            metadata={
                "document_id": "doc2",
                "chunk_id": "chunk2",
                "_distance": 0.85,
            },
        ),
    ]
    mock_retr.ainvoke = AsyncMock(return_value=docs)

    return mock_retr


@pytest.fixture
def context_manager():
    """Create a context window manager."""
    config = ContextWindow(
        max_tokens=4000,
        reserve_tokens=500,
        truncation_strategy="relevant",
    )
    return ContextWindowManager(config)


@pytest.fixture
def rag_orchestrator(mock_llm_service, mock_retriever, context_manager):
    """Create a RagOrchestrator instance."""
    return RagOrchestrator(
        llm_service=mock_llm_service,
        retriever=mock_retriever,
        context_manager=context_manager,
    )


class TestRagOrchestratorInit:
    """Tests for RagOrchestrator initialization."""

    def test_rag_orchestrator_initialization(self, rag_orchestrator):
        """Test that RagOrchestrator initializes correctly."""
        assert rag_orchestrator is not None
        assert rag_orchestrator._llm_service is not None
        assert rag_orchestrator._retriever is not None
        assert rag_orchestrator._context_manager is not None
        assert rag_orchestrator._turns_window == 10
        assert rag_orchestrator._max_session_age_hours == 24

    def test_rag_orchestrator_with_custom_parameters(
        self, mock_llm_service, mock_retriever, context_manager
    ):
        """Test RagOrchestrator initialization with custom parameters."""
        orchestrator = RagOrchestrator(
            llm_service=mock_llm_service,
            retriever=mock_retriever,
            context_manager=context_manager,
            turns_window=5,
            max_session_age_hours=12,
        )

        assert orchestrator.get_turns_window() == 5
        assert orchestrator._max_session_age_hours == 12

    def test_rag_orchestrator_with_shared_session_store(
        self, mock_llm_service, mock_retriever, context_manager
    ):
        """Test RagOrchestrator initialization with shared session store."""
        shared_store = {}
        orchestrator = RagOrchestrator(
            llm_service=mock_llm_service,
            retriever=mock_retriever,
            context_manager=context_manager,
            session_store=shared_store,
        )

        assert orchestrator._session_store is shared_store


class TestSessionManagement:
    """Tests for session management functionality."""

    def test_get_session_history_creates_new_session(self, rag_orchestrator):
        """Test that get_session_history creates a new session if it doesn't exist."""
        session_id = "test_session_1"
        history = rag_orchestrator.get_session_history(session_id)

        assert history is not None
        assert session_id in rag_orchestrator._session_store

    def test_get_session_history_returns_existing_session(self, rag_orchestrator):
        """Test that get_session_history returns existing session."""
        session_id = "test_session_2"
        history1 = rag_orchestrator.get_session_history(session_id)
        history2 = rag_orchestrator.get_session_history(session_id)

        assert history1 is history2

    def test_get_session_history_with_invalid_session_id(self, rag_orchestrator):
        """Test that invalid session ID raises ValueError."""
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            rag_orchestrator.get_session_history("")

        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            rag_orchestrator.get_session_history(None)

    @pytest.mark.asyncio
    async def test_clear_session(self, rag_orchestrator):
        """Test clearing a session."""
        session_id = "test_session_3"
        rag_orchestrator.get_session_history(session_id)

        result = await rag_orchestrator.clear_session(session_id)
        assert result is True
        assert session_id not in rag_orchestrator._session_store

    @pytest.mark.asyncio
    async def test_clear_non_existent_session(self, rag_orchestrator):
        """Test clearing a non-existent session."""
        result = await rag_orchestrator.clear_session("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_all_sessions(self, rag_orchestrator):
        """Test clearing all sessions."""
        rag_orchestrator.get_session_history("session_1")
        rag_orchestrator.get_session_history("session_2")
        rag_orchestrator.get_session_history("session_3")

        count = await rag_orchestrator.clear_all_sessions()
        assert count == 3
        assert len(rag_orchestrator._session_store) == 0

    @pytest.mark.asyncio
    async def test_get_active_session_count(self, rag_orchestrator):
        """Test getting active session count."""
        rag_orchestrator.get_session_history("session_1")
        rag_orchestrator.get_session_history("session_2")

        count = await rag_orchestrator.get_active_session_count()
        assert count == 2


class TestContextRetrieval:
    """Tests for context retrieval functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_context_success(self, rag_orchestrator):
        """Test successful context retrieval."""
        query = "What is fitness?"

        context, docs = await rag_orchestrator._retrieve(query)

        assert context is not None
        assert len(docs) == 2
        assert "Test content 1" in context or "Test content 2" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_no_documents(self, rag_orchestrator, mock_retriever):
        """Test context retrieval when no documents are found."""
        mock_retriever.ainvoke = AsyncMock(return_value=[])

        context, docs = await rag_orchestrator._retrieve("What is fitness?")

        assert context == ""
        assert len(docs) == 0

    @pytest.mark.asyncio
    async def test_retrieve_context_retrieval_error(self, rag_orchestrator, mock_retriever):
        """Test context retrieval when retriever fails."""
        mock_retriever.ainvoke = AsyncMock(
            side_effect=Exception("Retriever error")
        )

        from app.domain.llm.exceptions import ChatOrchestratorError

        with pytest.raises(ChatOrchestratorError, match="Failed to retrieve context"):
            await rag_orchestrator._retrieve("What is fitness?")


class TestSourceCitations:
    """Tests for source citation creation."""

    def test_create_source_citations(self, rag_orchestrator):
        """Test creating source citations from documents."""
        docs = [
            Document(
                page_content="Content 1" * 100,
                metadata={
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "_distance": 0.9,
                },
            ),
            Document(
                page_content="Content 2" * 100,
                metadata={
                    "document_id": "doc2",
                    "chunk_id": "chunk2",
                    "_distance": 0.85,
                },
            ),
        ]

        citations = rag_orchestrator._create_source_citations(docs)

        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[1].index == 2
        assert citations[0].document_id == "doc1"
        assert citations[1].document_id == "doc2"
        assert citations[0].similarity_score == 0.9
        assert citations[1].similarity_score == 0.85

    def test_create_source_citations_empty(self, rag_orchestrator):
        """Test creating source citations from empty document list."""
        citations = rag_orchestrator._create_source_citations([])
        assert len(citations) == 0


class TestChatValidation:
    """Tests for chat request validation."""

    def test_ensure_chat_request_valid(self, rag_orchestrator):
        """Test validation of valid chat request."""
        message = ChatMessage(role="user", content="Hello")
        request = ChatRequest(message=message, session_id="session_1")

        # Should not raise
        rag_orchestrator._ensure_chat_request(request)

    def test_ensure_chat_request_no_message(self, rag_orchestrator):
        """Test validation fails with no message."""
        from app.domain.llm.exceptions import MessageValidationError

        request = ChatRequest(message=None, session_id="session_1")

        with pytest.raises(MessageValidationError, match="Message is required"):
            rag_orchestrator._ensure_chat_request(request)

    def test_ensure_chat_request_empty_content(self, rag_orchestrator):
        """Test validation fails with empty message content."""
        from app.domain.llm.exceptions import MessageValidationError

        message = ChatMessage(role="user", content="")
        request = ChatRequest(message=message, session_id="session_1")

        with pytest.raises(MessageValidationError, match="Message content cannot be empty"):
            rag_orchestrator._ensure_chat_request(request)

    def test_ensure_chat_request_no_session_id(self, rag_orchestrator):
        """Test validation fails with no session ID."""
        from app.domain.llm.exceptions import MessageValidationError

        message = ChatMessage(role="user", content="Hello")
        request = ChatRequest(message=message, session_id="")

        with pytest.raises(MessageValidationError, match="Session ID is required"):
            rag_orchestrator._ensure_chat_request(request)


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, rag_orchestrator):
        """Test health check when system is healthy."""
        result = await rag_orchestrator.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_llm_unhealthy(self, rag_orchestrator):
        """Test health check when LLM service is unhealthy."""
        rag_orchestrator._llm_service.health_check = AsyncMock(return_value=False)

        result = await rag_orchestrator.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_retriever_error(self, rag_orchestrator, mock_retriever):
        """Test health check when retriever fails."""
        mock_retriever.ainvoke = AsyncMock(side_effect=Exception("Retriever error"))

        result = await rag_orchestrator.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, rag_orchestrator):
        """Test health check handles exceptions."""
        rag_orchestrator._llm_service.health_check = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        result = await rag_orchestrator.health_check()
        assert result is False


class TestTrimming:
    """Tests for message trimming functionality."""

    def test_should_apply_trimming_true(self, rag_orchestrator):
        """Test that trimming is applied for long conversations."""
        messages = [MagicMock() for _ in range(25)]
        result = rag_orchestrator._should_apply_trimming(messages)
        assert result is True

    def test_should_apply_trimming_false(self, rag_orchestrator):
        """Test that trimming is not applied for short conversations."""
        messages = [MagicMock() for _ in range(10)]
        result = rag_orchestrator._should_apply_trimming(messages)
        assert result is False

    def test_get_smart_trimmer_long_conversation(self, rag_orchestrator):
        """Test smart trimmer returns trimmer for long conversations."""
        messages = [MagicMock() for _ in range(25)]
        trimmer = rag_orchestrator._get_smart_trimmer(messages)
        assert trimmer is rag_orchestrator._trimmer

    def test_get_smart_trimmer_short_conversation(self, rag_orchestrator):
        """Test smart trimmer returns identity function for short conversations."""
        messages = [MagicMock() for _ in range(10)]
        trimmer = rag_orchestrator._get_smart_trimmer(messages)
        assert trimmer is not rag_orchestrator._trimmer


class TestWindowConfiguration:
    """Tests for window configuration."""

    def test_get_turns_window(self, rag_orchestrator):
        """Test getting turns window."""
        assert rag_orchestrator.get_turns_window() == 10

    def test_get_turns_window_custom(self, mock_llm_service, mock_retriever, context_manager):
        """Test getting custom turns window."""
        orchestrator = RagOrchestrator(
            llm_service=mock_llm_service,
            retriever=mock_retriever,
            context_manager=context_manager,
            turns_window=5,
        )
        assert orchestrator.get_turns_window() == 5
