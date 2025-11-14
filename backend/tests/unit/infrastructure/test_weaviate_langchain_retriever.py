"""Unit tests for WeaviateLangChainRetriever adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from app.infrastructure.adapters.weaviate_langchain_retriever import (
    WeaviateLangChainRetriever,
)
from app.domain.value_objects.search_result import SearchResult
from app.domain.value_objects.chunk_metadata import ChunkMetadata


class TestWeaviateLangChainRetriever:
    """Test WeaviateLangChainRetriever adapter functionality."""

    @pytest.fixture
    def mock_search_repository(self):
        """Create mock WeaviateSearchRepository."""
        return AsyncMock()

    @pytest.fixture
    def retriever(self, mock_search_repository):
        """Create WeaviateLangChainRetriever with mock repository."""
        return WeaviateLangChainRetriever(
            search_repository=mock_search_repository,
            top_k=5,
            similarity_threshold=0.7,
        )

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_success(
        self, retriever, mock_search_repository
    ):
        """Test successful async document retrieval."""
        # Mock search results
        mock_results = [
            SearchResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Fitness content about strength training",
                similarity_score=0.92,
                metadata={
                    "source": "fitness_guide.pdf",
                    "page": 5,
                    "author": "Expert",
                },
            ),
            SearchResult(
                chunk_id="chunk_2",
                document_id="doc_1",
                content="Cardio exercises for endurance",
                similarity_score=0.85,
                metadata={"source": "fitness_guide.pdf", "page": 12},
            ),
        ]
        mock_search_repository.semantic_search = AsyncMock(return_value=mock_results)

        # Execute
        run_manager = MagicMock()
        documents = await retriever._aget_relevant_documents(
            query="fitness tips", run_manager=run_manager
        )

        # Verify
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        # Verify first document
        assert documents[0].page_content == "Fitness content about strength training"
        assert documents[0].metadata["chunk_id"] == "chunk_1"
        assert documents[0].metadata["document_id"] == "doc_1"
        assert documents[0].metadata["similarity_score"] == 0.92
        assert documents[0].metadata["source"] == "fitness_guide.pdf"
        assert documents[0].metadata["page"] == 5
        assert documents[0].metadata["author"] == "Expert"

        # Verify repository was called correctly
        mock_search_repository.semantic_search.assert_called_once_with(
            query="fitness tips", top_k=5, min_similarity=0.7
        )

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_empty_results(
        self, retriever, mock_search_repository
    ):
        """Test retrieval with no matching documents."""
        # Mock empty results
        mock_search_repository.semantic_search = AsyncMock(return_value=[])

        # Execute
        run_manager = MagicMock()
        documents = await retriever._aget_relevant_documents(
            query="obscure query", run_manager=run_manager
        )

        # Verify
        assert documents == []

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_metadata_handling(
        self, retriever, mock_search_repository
    ):
        """Test proper metadata conversion from SearchResult to Document."""
        # Mock result with comprehensive metadata
        mock_result = SearchResult(
            chunk_id="chunk_test",
            document_id="doc_test",
            content="Test content",
            similarity_score=0.88,
            metadata={
                "source": "test.pdf",
                "custom_field": "value",
                "nested": {"key": "nested_value"},
            },
        )
        mock_search_repository.semantic_search = AsyncMock(return_value=[mock_result])

        # Execute
        run_manager = MagicMock()
        documents = await retriever._aget_relevant_documents(
            query="test", run_manager=run_manager
        )

        # Verify all metadata is preserved
        doc = documents[0]
        assert doc.metadata["chunk_id"] == "chunk_test"
        assert doc.metadata["document_id"] == "doc_test"
        assert doc.metadata["similarity_score"] == 0.88
        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["custom_field"] == "value"
        assert doc.metadata["nested"] == {"key": "nested_value"}

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_missing_source(
        self, retriever, mock_search_repository
    ):
        """Test handling of missing source metadata."""
        # Mock result without source
        mock_result = SearchResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content without source",
            similarity_score=0.80,
            metadata={},  # No source
        )
        mock_search_repository.semantic_search = AsyncMock(return_value=[mock_result])

        # Execute
        run_manager = MagicMock()
        documents = await retriever._aget_relevant_documents(
            query="test", run_manager=run_manager
        )

        # Verify source defaults to "unknown"
        assert documents[0].metadata["source"] == "unknown"

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_exception_handling(
        self, retriever, mock_search_repository
    ):
        """Test error handling during retrieval."""
        # Mock repository error
        mock_search_repository.semantic_search = AsyncMock(
            side_effect=Exception("Weaviate connection failed")
        )

        # Verify exception is raised
        run_manager = MagicMock()
        with pytest.raises(Exception, match="Weaviate connection failed"):
            await retriever._aget_relevant_documents(
                query="test", run_manager=run_manager
            )

    @pytest.mark.asyncio
    async def test_configurable_top_k(self, mock_search_repository):
        """Test that top_k parameter is configurable."""
        # Create retriever with custom top_k
        retriever = WeaviateLangChainRetriever(
            search_repository=mock_search_repository,
            top_k=10,
            similarity_threshold=0.7,
        )

        mock_search_repository.semantic_search = AsyncMock(return_value=[])

        # Execute
        run_manager = MagicMock()
        await retriever._aget_relevant_documents(query="test", run_manager=run_manager)

        # Verify top_k is passed correctly
        mock_search_repository.semantic_search.assert_called_once_with(
            query="test", top_k=10, min_similarity=0.7
        )

    @pytest.mark.asyncio
    async def test_configurable_similarity_threshold(self, mock_search_repository):
        """Test that similarity_threshold parameter is configurable."""
        # Create retriever with custom threshold
        retriever = WeaviateLangChainRetriever(
            search_repository=mock_search_repository,
            top_k=5,
            similarity_threshold=0.85,
        )

        mock_search_repository.semantic_search = AsyncMock(return_value=[])

        # Execute
        run_manager = MagicMock()
        await retriever._aget_relevant_documents(query="test", run_manager=run_manager)

        # Verify threshold is passed correctly
        mock_search_repository.semantic_search.assert_called_once_with(
            query="test", top_k=5, min_similarity=0.85
        )

    @pytest.mark.asyncio
    async def test_dynamic_top_k_modification(self, retriever, mock_search_repository):
        """Test that top_k can be modified dynamically."""
        mock_search_repository.semantic_search = AsyncMock(return_value=[])

        # Modify top_k
        retriever.top_k = 20

        # Execute
        run_manager = MagicMock()
        await retriever._aget_relevant_documents(query="test", run_manager=run_manager)

        # Verify new top_k is used
        mock_search_repository.semantic_search.assert_called_once_with(
            query="test", top_k=20, min_similarity=0.7
        )

    def test_sync_retrieval_not_implemented(self, retriever):
        """Test that synchronous retrieval raises NotImplementedError."""
        run_manager = MagicMock()

        with pytest.raises(
            NotImplementedError, match="Use aget_relevant_documents for async"
        ):
            retriever._get_relevant_documents(query="test", run_manager=run_manager)

    @pytest.mark.asyncio
    async def test_multiple_documents_conversion(
        self, retriever, mock_search_repository
    ):
        """Test conversion of multiple search results to documents."""
        # Create 5 mock results
        mock_results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                document_id=f"doc_{i}",
                content=f"Content {i}",
                similarity_score=0.9 - (i * 0.05),
                metadata={"source": f"file_{i}.pdf"},
            )
            for i in range(5)
        ]
        mock_search_repository.semantic_search = AsyncMock(return_value=mock_results)

        # Execute
        run_manager = MagicMock()
        documents = await retriever._aget_relevant_documents(
            query="test", run_manager=run_manager
        )

        # Verify all converted
        assert len(documents) == 5
        for i, doc in enumerate(documents):
            assert doc.page_content == f"Content {i}"
            assert doc.metadata["chunk_id"] == f"chunk_{i}"
            assert doc.metadata["document_id"] == f"doc_{i}"
            assert doc.metadata["similarity_score"] == 0.9 - (i * 0.05)

    @pytest.mark.asyncio
    async def test_page_content_preservation(self, retriever, mock_search_repository):
        """Test that page content is exactly preserved."""
        # Content with special characters and formatting
        special_content = """Line 1 with special chars: @#$%
        Line 2 with   multiple   spaces
        Line 3 with\ttabs\tand\nnewlines"""

        mock_result = SearchResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            content=special_content,
            similarity_score=0.9,
            metadata={},
        )
        mock_search_repository.semantic_search = AsyncMock(return_value=[mock_result])

        # Execute
        run_manager = MagicMock()
        documents = await retriever._aget_relevant_documents(
            query="test", run_manager=run_manager
        )

        # Verify content is exactly preserved
        assert documents[0].page_content == special_content

    @pytest.mark.asyncio
    async def test_default_parameters(self, mock_search_repository):
        """Test default parameter values."""
        # Create retriever with defaults
        retriever = WeaviateLangChainRetriever(
            search_repository=mock_search_repository
        )

        assert retriever.top_k == 5
        assert retriever.similarity_threshold == 0.7
