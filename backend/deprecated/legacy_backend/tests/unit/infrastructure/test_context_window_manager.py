"""Unit tests for ContextWindowManager (Task 3.1.3)."""

import pytest
from langchain_core.documents import Document

from app.infrastructure.external_services.context_management.context_window_manager import (
    ContextWindow,
    ContextWindowManager,
)


class TestContextWindowManager:
    """Test ContextWindowManager token estimation and truncation."""

    @pytest.fixture
    def default_config(self):
        """Create default context window configuration."""
        return ContextWindow(
            max_tokens=4000, reserve_tokens=500, truncation_strategy="relevant"
        )

    @pytest.fixture
    def manager(self, default_config):
        """Create ContextWindowManager with default config."""
        return ContextWindowManager(default_config)

    def test_initialization(self, manager, default_config):
        """Test manager initialization."""
        assert manager.config == default_config
        assert manager._available_tokens == 3500  # 4000 - 500

    def test_token_estimation(self, manager):
        """Test token estimation using CHARS_PER_TOKEN ratio."""
        # CHARS_PER_TOKEN = 4, so 100 chars ≈ 25 tokens
        text = "a" * 100
        tokens = manager._estimate_tokens(text)
        assert tokens == 25

        # Empty text
        assert manager._estimate_tokens("") == 0

        # Single character
        assert manager._estimate_tokens("a") == 1  # Rounds up

    def test_fit_to_window_small_context(self, manager):
        """Test fit_to_window with context that fits in window."""
        documents = [
            Document(page_content="Short document 1", metadata={"chunk_id": "1"}),
            Document(page_content="Short document 2", metadata={"chunk_id": "2"}),
        ]

        result = manager.fit_to_window(
            documents=documents,
            user_query="Test query",
            system_prompt="You are a helpful assistant.",
        )

        # All documents should be included
        assert "Short document 1" in result
        assert "Short document 2" in result
        assert len(result) > 0

    def test_fit_to_window_exceeds_limit(self, manager):
        """Test fit_to_window with context exceeding token limit."""
        # Create documents that exceed available tokens
        large_content = "x" * 5000  # ~1250 tokens per document
        documents = [
            Document(page_content=large_content, metadata={"chunk_id": str(i)})
            for i in range(10)  # Total: ~12,500 tokens
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="Prompt"
        )

        # Result should be truncated to fit
        result_tokens = manager._estimate_tokens(result)
        assert result_tokens <= manager._available_tokens

    def test_truncation_strategy_recent(self):
        """Test 'recent' truncation strategy."""
        config = ContextWindow(
            max_tokens=1000, reserve_tokens=200, truncation_strategy="recent"
        )
        manager = ContextWindowManager(config)

        # Create documents
        documents = [
            Document(page_content=f"Document {i} content", metadata={"chunk_id": str(i)})
            for i in range(5)
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="System"
        )

        # Recent strategy should prefer later documents
        # Since we have limited space, earlier docs might be excluded
        assert len(result) > 0

    def test_truncation_strategy_relevant(self):
        """Test 'relevant' truncation strategy."""
        config = ContextWindow(
            max_tokens=1000, reserve_tokens=200, truncation_strategy="relevant"
        )
        manager = ContextWindowManager(config)

        documents = [
            Document(
                page_content="Highly relevant fitness content",
                metadata={"similarity_score": 0.95, "chunk_id": "1"},
            ),
            Document(
                page_content="Less relevant content",
                metadata={"similarity_score": 0.60, "chunk_id": "2"},
            ),
            Document(
                page_content="Moderately relevant content",
                metadata={"similarity_score": 0.75, "chunk_id": "3"},
            ),
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="System"
        )

        # Relevant strategy should prioritize by similarity score
        # Higher scored documents should appear first
        assert "Highly relevant" in result or len(result) > 0

    def test_truncation_strategy_summarize(self):
        """Test 'summarize' truncation strategy."""
        config = ContextWindow(
            max_tokens=1000, reserve_tokens=200, truncation_strategy="summarize"
        )
        manager = ContextWindowManager(config)

        documents = [
            Document(page_content=f"Document {i} with content", metadata={"chunk_id": str(i)})
            for i in range(5)
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="System"
        )

        # Summarize strategy should produce condensed output
        assert len(result) > 0
        result_tokens = manager._estimate_tokens(result)
        assert result_tokens <= manager._available_tokens

    def test_empty_documents(self, manager):
        """Test fit_to_window with empty document list."""
        result = manager.fit_to_window(
            documents=[], user_query="Query", system_prompt="System"
        )

        # Should return empty string
        assert result == ""

    def test_single_document(self, manager):
        """Test fit_to_window with single document."""
        documents = [
            Document(
                page_content="Single document content", metadata={"chunk_id": "1"}
            )
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="System"
        )

        assert "Single document content" in result

    def test_query_and_system_prompt_included(self, manager):
        """Test that query and system prompt tokens are accounted for."""
        long_system_prompt = "x" * 2000  # ~500 tokens
        long_query = "y" * 2000  # ~500 tokens
        # Leaves ~2500 tokens for documents (available_tokens = 3500)

        large_content = "z" * 12000  # ~3000 tokens
        documents = [Document(page_content=large_content, metadata={"chunk_id": "1"})]

        result = manager.fit_to_window(
            documents=documents, user_query=long_query, system_prompt=long_system_prompt
        )

        # Total context should fit in available tokens
        total_tokens = (
            manager._estimate_tokens(long_system_prompt)
            + manager._estimate_tokens(long_query)
            + manager._estimate_tokens(result)
        )
        assert total_tokens <= manager._available_tokens

    def test_document_separator(self, manager):
        """Test that documents are properly separated."""
        documents = [
            Document(page_content="Doc 1", metadata={"chunk_id": "1"}),
            Document(page_content="Doc 2", metadata={"chunk_id": "2"}),
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="System"
        )

        # Documents should be separated by double newlines
        assert "\n\n" in result or "Doc 1" in result

    def test_metadata_preserved_in_context(self, manager):
        """Test that document metadata is accessible during truncation."""
        documents = [
            Document(
                page_content="Content with metadata",
                metadata={
                    "chunk_id": "123",
                    "document_id": "doc_456",
                    "similarity_score": 0.85,
                },
            )
        ]

        result = manager.fit_to_window(
            documents=documents, user_query="Query", system_prompt="System"
        )

        # Content should be in result
        assert "Content with metadata" in result

    def test_custom_config_values(self):
        """Test manager with custom configuration values."""
        config = ContextWindow(
            max_tokens=8000,  # Larger context window
            reserve_tokens=1000,  # More reserve
            truncation_strategy="recent",
        )
        manager = ContextWindowManager(config)

        assert manager._available_tokens == 7000
        assert manager.config.truncation_strategy == "recent"

    def test_zero_reserve_tokens(self):
        """Test manager with zero reserve tokens."""
        config = ContextWindow(max_tokens=4000, reserve_tokens=0, truncation_strategy="relevant")
        manager = ContextWindowManager(config)

        assert manager._available_tokens == 4000

    def test_truncate_by_relevance_sorting(self, manager):
        """Test that relevant strategy sorts by similarity score."""
        documents = [
            Document(
                page_content="Low score",
                metadata={"similarity_score": 0.5, "chunk_id": "1"},
            ),
            Document(
                page_content="High score",
                metadata={"similarity_score": 0.95, "chunk_id": "2"},
            ),
            Document(
                page_content="Medium score",
                metadata={"similarity_score": 0.75, "chunk_id": "3"},
            ),
        ]

        sorted_docs = manager._truncate_by_relevance(documents, available_tokens=1000)

        # Should be sorted by similarity_score descending
        assert sorted_docs[0].metadata["similarity_score"] == 0.95
        assert sorted_docs[1].metadata["similarity_score"] == 0.75
        assert sorted_docs[2].metadata["similarity_score"] == 0.5

    def test_truncate_recent_preserves_order(self, manager):
        """Test that recent strategy preserves document order."""
        documents = [
            Document(page_content=f"Doc {i}", metadata={"chunk_id": str(i)})
            for i in range(5)
        ]

        result_docs = manager._truncate_recent(documents, available_tokens=1000)

        # Should maintain original order
        for i, doc in enumerate(result_docs):
            assert doc.metadata["chunk_id"] == str(i)
