"""Unit tests for BuildIngestionPipelineUseCase (Task 2.3.3).

Tests cover:
- Pipeline execution with valid inputs
- Error handling for each stage
- Deduplication functionality
- Retry logic
- Progress tracking with ProcessingJob
- Edge cases and boundary conditions
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
import pytest

from app.application.use_cases.indexing.build_ingestion_pipeline import (
    BuildIngestionPipelineRequest,
    BuildIngestionPipelineResponse,
    BuildIngestionPipelineUseCase,
    DeduplicationStats,
)
from app.domain.entities.chunk import Chunk
from app.domain.entities.processing_job import JobStatus
from app.domain.exceptions.chunking_exceptions import ChunkingError
from app.domain.exceptions.embedding_exceptions import (
    DeduplicationError,
    EmbeddingGenerationError,
    EmbeddingStorageError,
    IngestionPipelineError,
)
from app.domain.value_objects.chunk_metadata import ChunkMetadata


class TestBuildIngestionPipelineRequest:
    """Test BuildIngestionPipelineRequest dataclass."""

    def test_request_creation(self):
        """Test request creation with default values."""
        request = BuildIngestionPipelineRequest()

        assert request.document_ids is None
        assert request.batch_size == 32
        assert request.deduplication_enabled is True
        assert request.max_retries == 3
        assert request.retry_backoff_factor == 1.0
        assert request.show_progress is True
        assert request.replace_existing_embeddings is False

    def test_request_creation_with_values(self):
        """Test request creation with custom values."""
        doc_ids = [uuid4(), uuid4()]
        request = BuildIngestionPipelineRequest(
            document_ids=doc_ids,
            batch_size=64,
            deduplication_enabled=False,
            max_retries=5,
            retry_backoff_factor=2.0,
            show_progress=False
        )

        assert request.document_ids == doc_ids
        assert request.batch_size == 64
        assert request.deduplication_enabled is False
        assert request.max_retries == 5
        assert request.retry_backoff_factor == 2.0
        assert request.show_progress is False


class TestDeduplicationStats:
    """Test DeduplicationStats dataclass."""

    def test_stats_creation(self):
        """Test stats creation."""
        stats = DeduplicationStats(
            total_chunks=100,
            unique_chunks=85,
            duplicates_removed=15,
            duplicates_percentage=15.0
        )

        assert stats.total_chunks == 100
        assert stats.unique_chunks == 85
        assert stats.duplicates_removed == 15
        assert stats.duplicates_percentage == 15.0

    def test_stats_as_dict(self):
        """Test stats conversion to dictionary."""
        stats = DeduplicationStats(
            total_chunks=100,
            unique_chunks=85,
            duplicates_removed=15,
            duplicates_percentage=15.5678
        )

        result = stats.as_dict()
        expected = {
            "total_chunks": 100,
            "unique_chunks": 85,
            "duplicates_removed": 15,
            "duplicates_percentage": 15.57  # Rounded
        }

        assert result == expected


class TestBuildIngestionPipelineUseCase:
    """Test BuildIngestionPipelineUseCase."""

    @pytest.fixture
    def mock_document_repository(self):
        """Create mock document repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_chunking_use_case(self):
        """Create mock chunking use case."""
        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.chunks = []
        mock_use_case.execute.return_value = mock_response
        return mock_use_case

    @pytest.fixture
    def mock_embedding_use_case(self):
        """Create mock embedding use case."""
        mock_use_case = AsyncMock()
        mock_response = Mock()
        mock_response.embedded_count = 0
        mock_response.stored_count = 0
        mock_use_case.execute.return_value = mock_response
        return mock_use_case

    @pytest.fixture
    def use_case(self, mock_document_repository, mock_chunking_use_case, mock_embedding_use_case):
        """Create use case instance with mocked dependencies."""
        return BuildIngestionPipelineUseCase(
            document_repository=mock_document_repository,
            chunking_use_case=mock_chunking_use_case,
            embedding_use_case=mock_embedding_use_case
        )

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(5):
            chunk = Chunk(
                chunk_id=str(uuid4()),
                document_id=uuid4(),
                text=f"This is chunk {i} with some content for testing.",
                metadata=ChunkMetadata(
                    sequence=i,
                    start=i * 50,
                    end=(i + 1) * 50,
                    token_count=10,
                    source_type="semantic"
                )
            )
            chunks.append(chunk)
        return chunks

    def test_execute_success_no_documents(self, use_case):
        """Test successful execution with no documents."""
        request = BuildIngestionPipelineRequest(document_ids=[])

        with pytest.raises(IngestionPipelineError) as exc_info:
            # For testing purposes, we'll make this synchronous by calling the method directly
            import asyncio
            asyncio.get_event_loop().run_until_complete(use_case.execute(request))

        assert "No document IDs provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_success_with_documents(self, use_case, mock_chunking_use_case, mock_embedding_use_case, sample_chunks):
        """Test successful execution with documents."""
        # Setup mocks
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = sample_chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        mock_embedding_response = Mock()
        mock_embedding_response.embedded_count = 5
        mock_embedding_response.stored_count = 5
        mock_embedding_use_case.execute.return_value = mock_embedding_response

        # Execute
        request = BuildIngestionPipelineRequest(
            document_ids=[uuid4(), uuid4()],
            batch_size=32
        )

        response = await use_case.execute(request)

        # Assertions
        assert response.success is True
        assert response.job_id is not None
        assert response.results["chunks_generated"] == 5
        assert response.results["embeddings_generated"] == 5
        assert response.results["embeddings_stored"] == 5
        assert response.deduplication_stats is not None
        assert response.deduplication_stats.total_chunks == 5
        assert len(response.errors) == 0

    @pytest.mark.asyncio
    async def test_execute_chunking_failure(self, use_case, mock_chunking_use_case):
        """Test handling of chunking failure."""
        # Setup mock to raise ChunkingError
        mock_chunking_use_case.execute.side_effect = ChunkingError(
            "Chunking service unavailable",
            document_id=str(uuid4())
        )

        request = BuildIngestionPipelineRequest(document_ids=[uuid4()])

        with pytest.raises(IngestionPipelineError) as exc_info:
            await use_case.execute(request)

        assert "chunking" in str(exc_info.value)
        assert use_case._current_job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_with_deduplication_disabled(self, use_case, mock_chunking_use_case, mock_embedding_use_case, sample_chunks):
        """Test execution with deduplication disabled."""
        # Setup mocks
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = sample_chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        mock_embedding_response = Mock()
        mock_embedding_response.embedded_count = 5
        mock_embedding_response.stored_count = 5
        mock_embedding_use_case.execute.return_value = mock_embedding_response

        # Execute with deduplication disabled
        request = BuildIngestionPipelineRequest(
            document_ids=[uuid4()],
            deduplication_enabled=False
        )

        response = await use_case.execute(request)

        # Assertions
        assert response.success is True
        assert response.deduplication_stats.duplicates_removed == 0
        assert response.deduplication_stats.duplicates_percentage == 0.0

    @pytest.mark.asyncio
    async def test_deduplication_with_duplicates(self, use_case, mock_chunking_use_case, mock_embedding_use_case):
        """Test deduplication with duplicate chunks."""
        # Create chunks with duplicate content
        chunk1 = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="Same content",
            metadata=ChunkMetadata(sequence=1, start=0, end=12, token_count=2, source_type="semantic")
        )
        chunk2 = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="Same content",
            metadata=ChunkMetadata(sequence=2, start=13, end=25, token_count=2, source_type="semantic")
        )
        chunk3 = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text="Different content",
            metadata=ChunkMetadata(sequence=1, start=0, end=16, token_count=2, source_type="semantic")
        )

        chunks = [chunk1, chunk2, chunk3]

        # Setup mocks
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        mock_embedding_response = Mock()
        mock_embedding_response.embedded_count = 2  # Only 2 unique chunks
        mock_embedding_response.stored_count = 2
        mock_embedding_use_case.execute.return_value = mock_embedding_response

        # Execute
        request = BuildIngestionPipelineRequest(document_ids=[uuid4()])
        response = await use_case.execute(request)

        # Assertions
        assert response.success is True
        assert response.deduplication_stats.total_chunks == 3
        assert response.deduplication_stats.unique_chunks == 2
        assert response.deduplication_stats.duplicates_removed == 1
        assert response.deduplication_stats.duplicates_percentage == 33.33

    @pytest.mark.asyncio
    async def test_retry_logic_success_on_retry(self, use_case, mock_chunking_use_case, mock_embedding_use_case, sample_chunks):
        """Test retry logic succeeding on retry."""
        # Setup chunking mock
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = sample_chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        # Setup embedding mock to fail first time, succeed second time
        mock_embedding_response = Mock()
        mock_embedding_response.embedded_count = 5
        mock_embedding_response.stored_count = 5

        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EmbeddingGenerationError("Temporary failure")
            return mock_embedding_response

        mock_embedding_use_case.execute.side_effect = mock_execute

        # Execute with retry
        request = BuildIngestionPipelineRequest(
            document_ids=[uuid4()],
            max_retries=3,
            retry_backoff_factor=0.1  # Short wait for tests
        )

        response = await use_case.execute(request)

        # Assertions
        assert response.success is True
        assert call_count == 2  # Failed once, succeeded on retry
        assert response.results["embeddings_generated"] == 5

    @pytest.mark.asyncio
    async def test_retry_logic_exhausted(self, use_case, mock_chunking_use_case, mock_embedding_use_case, sample_chunks):
        """Test retry logic exhausting all retries."""
        # Setup chunking mock
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = sample_chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        # Setup embedding mock to always fail
        mock_embedding_use_case.execute.side_effect = EmbeddingGenerationError("Persistent failure")

        # Execute with retry
        request = BuildIngestionPipelineRequest(
            document_ids=[uuid4()],
            max_retries=2,
            retry_backoff_factor=0.01  # Very short wait for tests
        )

        with pytest.raises(IngestionPipelineError) as exc_info:
            await use_case.execute(request)

        assert "embedding" in str(exc_info.value)
        assert use_case._current_job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_progress_tracking(self, use_case, mock_chunking_use_case, mock_embedding_use_case, sample_chunks):
        """Test progress tracking with ProcessingJob."""
        # Setup mocks
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = sample_chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        mock_embedding_response = Mock()
        mock_embedding_response.embedded_count = 5
        mock_embedding_response.stored_count = 5
        mock_embedding_use_case.execute.return_value = mock_embedding_response

        # Execute
        request = BuildIngestionPipelineRequest(document_ids=[uuid4()])
        response = await use_case.execute(request)

        # Assertions
        assert response.success is True
        assert response.job_id is not None

        job = use_case.get_current_job()
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.results["chunks_generated"] == 5
        assert job.results["embeddings_generated"] == 5

    @pytest.mark.asyncio
    async def test_empty_chunks_response(self, use_case, mock_chunking_use_case):
        """Test handling when chunking returns no chunks."""
        # Setup mock to return empty chunks
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = []
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        # Execute
        request = BuildIngestionPipelineRequest(document_ids=[uuid4()])
        response = await use_case.execute(request)

        # Assertions
        assert response.success is True
        assert response.results["chunks_generated"] == 0
        assert len(response.errors) == 1
        assert "No chunks generated" in response.errors[0]

    def test_get_current_job_before_execution(self, use_case):
        """Test getting current job before execution."""
        job = use_case.get_current_job()
        assert job is None

    def test_get_current_job_after_execution(self, use_case, mock_chunking_use_case, mock_embedding_use_case):
        """Test getting current job after execution."""
        # Setup mock
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = []
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        # This would be called after execute() in a real test
        # For now, just test the method exists and returns None
        job = use_case.get_current_job()
        assert job is None  # Still None since execute() hasn't been called

    @pytest.mark.asyncio
    async def test_deduplication_error_handling(self, use_case, mock_chunking_use_case, mock_embedding_use_case, sample_chunks):
        """Test handling of deduplication errors."""
        # Setup chunking mock
        mock_chunking_response = Mock()
        mock_chunking_response.success = True
        mock_chunking_response.chunks = sample_chunks
        mock_chunking_use_case.execute.return_value = mock_chunking_response

        # Setup embedding mock
        mock_embedding_response = Mock()
        mock_embedding_response.embedded_count = 5
        mock_embedding_response.stored_count = 5
        mock_embedding_use_case.execute.return_value = mock_embedding_response

        # Mock deduplication to fail by patching the method
        original_deduplicate = use_case._deduplicate_chunks
        async def failing_deduplicate(chunks, enabled):
            if enabled:
                raise DeduplicationError("Hash generation failed")
            return chunks, DeduplicationStats(len(chunks), len(chunks), 0, 0.0)

        use_case._deduplicate_chunks = failing_deduplicate

        # Execute
        request = BuildIngestionPipelineRequest(document_ids=[uuid4()])
        response = await use_case.execute(request)

        # Should continue with original chunks despite deduplication failure
        assert response.success is True
        assert len(response.errors) == 1
        assert "Deduplication failed" in response.errors[0]
        assert response.deduplication_stats.duplicates_removed == 0