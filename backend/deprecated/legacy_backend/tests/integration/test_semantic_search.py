"""Integration tests for semantic search functionality (Task 2.4.1).

This module provides integration tests for the semantic search use case
to verify end-to-end functionality with real data.
"""

import pytest
from uuid import uuid4

from app.application.use_cases.retrieval.semantic_search import (
    SemanticSearchRequest,
    SemanticSearchUseCase,
)
from app.domain.value_objects.search_filters import SearchFilters
from app.domain.entities.search_query import SearchQuery


@pytest.mark.asyncio
@pytest.mark.integration
class TestSemanticSearchIntegration:
    """Integration tests for semantic search functionality."""

    @pytest.fixture
    async def search_use_case(self):
        """Create search use case with test dependencies."""
        # Note: In a real test environment, you'd mock or use test databases
        # For now, we'll create a placeholder that shows the structure

        # These would be injected in a real test environment
        # embed_query_uc = mock_embed_query_uc
        # search_repository = mock_search_repository
        # retrieval_service = RetrievalService()

        # return SemanticSearchUseCase(
        #     embed_query_uc=embed_query_uc,
        #     search_repository=search_repository,
        #     retrieval_service=retrieval_service,
        # )

        # Placeholder for now
        pytest.skip("Integration tests require test database setup")

    async def test_basic_semantic_search(self, search_use_case):
        """Test basic semantic search functionality."""
        # Create search request
        request = SemanticSearchRequest(
            query_text="What exercises help with lower back pain?",
            top_k=5,
            min_similarity=0.7,
        )

        # Execute search
        response = await search_use_case.execute(request)

        # Verify response
        assert response.success is True
        assert len(response.results) <= 5
        assert response.processing_time_ms > 0
        assert response.query_vector_dimension > 0

        # Verify results structure
        for result in response.results:
            assert result.content
            assert result.similarity_score.score >= 0.7
            assert result.rank >= 1
            assert result.document_id

    async def test_search_with_filters(self, search_use_case):
        """Test semantic search with filters."""
        # Create search request with filters
        request = SemanticSearchRequest(
            query_text="strength training exercises",
            top_k=10,
            filters={
                "doc_types": ["pdf", "docx"],
                "departments": ["fitness"],
                "quality_threshold": 0.8,
            },
        )

        # Execute search
        response = await search_use_case.execute(request)

        # Verify response
        assert response.success is True
        assert response.total_results >= 0

        # Verify filters were applied
        # In real test, you'd verify that results match filter criteria

    async def test_search_suggestions(self, search_use_case):
        """Test search suggestions functionality."""
        # Get suggestions for partial query
        suggestions = await search_use_case.get_search_suggestions(
            partial_query="lower back",
            max_suggestions=5,
            min_similarity=0.4,
        )

        # Verify suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5

        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert "back" in suggestion.lower() or "lower" in suggestion.lower()

    async def test_similar_chunks(self, search_use_case):
        """Test finding similar chunks."""
        # Find chunks similar to example chunk IDs
        chunk_ids = [str(uuid4()), str(uuid4())]  # Example chunk IDs

        similar_chunks = await search_use_case.find_similar_chunks(
            chunk_ids=chunk_ids,
            top_k=5,
            min_similarity=0.6,
        )

        # Verify results
        assert isinstance(similar_chunks, list)

        for chunk in similar_chunks:
            assert chunk.content
            assert chunk.similarity_score.score >= 0.6
            assert chunk.chunk_id not in chunk_ids  # Should not return original chunks

    async def test_search_feedback_logging(self, search_use_case):
        """Test search feedback logging."""
        # Log feedback for a search
        query_id = uuid4()
        result_ids = [str(uuid4()), str(uuid4())]
        clicked_result_id = result_ids[0]
        feedback_score = 4

        # This should not raise an exception
        await search_use_case.log_search_feedback(
            query_id=query_id,
            result_ids=result_ids,
            clicked_result_id=clicked_result_id,
            feedback_score=feedback_score,
        )

        # In real test, you'd verify that feedback was logged correctly

    async def test_performance_metrics(self, search_use_case):
        """Test performance metrics retrieval."""
        # Get performance metrics
        metrics = await search_use_case.get_performance_metrics()

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "embedding_metrics" in metrics
        assert "search_health" in metrics
        assert "search_statistics" in metrics
        assert "overall_status" in metrics


@pytest.mark.asyncio
@pytest.mark.unit
class TestSearchQueryValidation:
    """Unit tests for search query validation."""

    async def test_search_query_creation(self):
        """Test SearchQuery creation and validation."""
        # Valid query
        query = SearchQuery.create(
            text="What exercises help with lower back pain?",
            top_k=10,
            min_similarity=0.7,
        )

        assert query.text == "What exercises help with lower back pain?"
        assert query.top_k == 10
        assert query.min_similarity == 0.7
        assert query.query_id
        assert query.created_at

    async def test_search_query_validation_errors(self):
        """Test SearchQuery validation error cases."""
        # Empty query
        with pytest.raises(ValueError, match="Search query text cannot be empty"):
            SearchQuery.create(text="   ", top_k=10)

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            SearchQuery.create(text="test", top_k=0)

        # Invalid similarity threshold
        with pytest.raises(ValueError, match="min_similarity must be between 0.0 and 1.0"):
            SearchQuery.create(text="test", min_similarity=1.5)

    async def test_search_query_with_filters(self):
        """Test SearchQuery with filters."""
        filters = SearchFilters(
            doc_types={"pdf", "docx"},
            departments={"fitness"},
            quality_threshold=0.8,
        )

        query = SearchQuery.create(
            text="fitness exercises",
            filters=filters,
            top_k=5,
        )

        assert query.filters.doc_types == {"pdf", "docx"}
        assert query.filters.departments == {"fitness"}
        assert query.filters.quality_threshold == 0.8


@pytest.mark.asyncio
@pytest.mark.unit
class TestSearchFilters:
    """Unit tests for SearchFilters functionality."""

    def test_empty_filters(self):
        """Test empty search filters."""
        filters = SearchFilters.create_empty()

        assert filters.is_empty()
        assert not filters.doc_types
        assert not filters.departments
        assert not filters.tags

    def test_filters_by_document_types(self):
        """Test creating filters by document types."""
        filters = SearchFilters.by_document_types(["pdf", "docx", "txt"])

        assert not filters.is_empty()
        assert filters.doc_types == {"pdf", "docx", "txt"}

    def test_filters_by_departments(self):
        """Test creating filters by departments."""
        filters = SearchFilters.by_departments(["fitness", "nutrition"])

        assert not filters.is_empty()
        assert filters.departments == {"fitness", "nutrition"}

    def test_filter_chaining(self):
        """Test chaining filter modifications."""
        filters = SearchFilters.create_empty()
        filters = filters.with_doc_types(["pdf"]).with_tags(["exercise"])

        assert filters.doc_types == {"pdf"}
        assert filters.tags == {"exercise"}

    def test_weaviate_filter_conversion(self):
        """Test conversion to Weaviate filter format."""
        filters = SearchFilters(
            doc_types={"pdf"},
            quality_threshold=0.8,
            min_token_count=50,
        )

        weaviate_filters = filters.to_weaviate_filters()

        assert "operator" in weaviate_filters  # Should be "And" operator
        assert "operands" in weaviate_filters

        # Check individual filter conditions
        operands = weaviate_filters["operands"]
        doc_type_filter = next((op for op in operands if op["path"] == ["doc_type"]), None)
        quality_filter = next((op for op in operands if op["path"] == ["quality_score"]), None)

        assert doc_type_filter is not None
        assert doc_type_filter["valueString"] == "pdf"
        assert quality_filter is not None
        assert quality_filter["valueNumber"] == 0.8


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/integration/test_semantic_search.py -v
    pass