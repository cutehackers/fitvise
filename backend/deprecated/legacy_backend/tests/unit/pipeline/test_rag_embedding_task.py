"""Unit tests for RagEmbeddingTask (Embedding Generation Phase).

Tests cover:
- Task initialization and configuration
- Processed document retrieval from shared repository
- Chunk availability validation (Task 2 handover validation)
- Embedding generation and storage
- Deduplication statistics
- Error handling and recovery
- Progress tracking and timing
- Edge cases and boundary conditions
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from datetime import datetime, timezone
from uuid import UUID, uuid4
import json
import pytest

from app.pipeline.phases.rag_embedding_task import (
    RagEmbeddingTask,
    RagEmbeddingTaskReport,
    EmbeddingResult,
    EmbeddingPipelineError,
)
from app.pipeline.config import PipelineSpec, ChunkingOptions
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.entities.document import Document
from app.domain.entities.chunk import Chunk
from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentFormat
from app.domain.exceptions import ProcessingError
from app.infrastructure.external_services import ExternalServicesContainer
from app.application.use_cases.indexing.build_ingestion_pipeline import (
    BuildIngestionPipelineResponse,
    DeduplicationStats,
)


class TestRagEmbeddingTaskInitialization:
    """Test RagEmbeddingTask initialization."""

    def test_task_initialization_with_defaults(self):
        """Test task initialization with default parameters."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        assert task.external_services is external_services
        assert task.document_repository is document_repository
        assert task.verbose is False

    def test_task_initialization_with_verbose(self):
        """Test task initialization with verbose logging enabled."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
            verbose=True,
        )

        assert task.verbose is True

    def test_task_initialization_stores_repositories(self):
        """Test that task stores both external services and repository."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        assert hasattr(task, "external_services")
        assert hasattr(task, "document_repository")


class TestGetProcessedDocumentIds:
    """Test _get_processed_document_ids method."""

    @pytest.mark.asyncio
    async def test_get_processed_documents_success(self):
        """Test retrieving processed document IDs successfully."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        # Create mock documents
        doc_id_1 = uuid4()
        doc_id_2 = uuid4()
        doc_id_3 = uuid4()

        mock_doc_1 = Mock(spec=Document, id=doc_id_1)
        mock_doc_2 = Mock(spec=Document, id=doc_id_2)
        mock_doc_3 = Mock(spec=Document, id=doc_id_3)

        document_repository.find_processed_documents.return_value = [
            mock_doc_1,
            mock_doc_2,
            mock_doc_3,
        ]

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        result = await task._get_processed_document_ids()

        assert len(result) == 3
        assert doc_id_1 in result
        assert doc_id_2 in result
        assert doc_id_3 in result
        document_repository.find_processed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_processed_documents_with_limit(self):
        """Test retrieving processed documents with limit."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_ids = [uuid4() for _ in range(5)]
        mock_docs = [Mock(spec=Document, id=doc_id) for doc_id in doc_ids]

        document_repository.find_processed_documents.return_value = mock_docs

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        result = await task._get_processed_document_ids(limit=2)

        assert len(result) == 2
        assert result[0] == doc_ids[0]
        assert result[1] == doc_ids[1]

    @pytest.mark.asyncio
    async def test_get_processed_documents_empty(self):
        """Test when no processed documents exist."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        document_repository.find_processed_documents.return_value = []

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        result = await task._get_processed_document_ids()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_processed_documents_repository_error(self):
        """Test error handling when repository fails."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        document_repository.find_processed_documents.side_effect = Exception(
            "Database connection failed"
        )

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        with pytest.raises(EmbeddingPipelineError, match="Document retrieval failed"):
            await task._get_processed_document_ids()


class TestValidateChunkAvailability:
    """Test _validate_chunk_availability method (Task 2 handover validation)."""

    @pytest.mark.asyncio
    async def test_validate_chunks_all_present(self):
        """Test validation when all documents have chunks."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id_1 = uuid4()
        doc_id_2 = uuid4()
        document_ids = [doc_id_1, doc_id_2]

        # Mock documents with chunks
        chunk_1 = Mock(spec=Chunk)
        chunk_2 = Mock(spec=Chunk)
        chunk_3 = Mock(spec=Chunk)

        mock_doc_1 = Mock(spec=Document, id=doc_id_1, chunks=[chunk_1, chunk_2])
        mock_doc_2 = Mock(spec=Document, id=doc_id_2, chunks=[chunk_3])

        document_repository.find_by_id.side_effect = [mock_doc_1, mock_doc_2]

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        result = await task._validate_chunk_availability(document_ids)

        assert result["total_documents"] == 2
        assert result["documents_with_chunks"] == 2
        assert result["total_chunks"] == 3
        assert result["documents_without_chunks"] == []

    @pytest.mark.asyncio
    async def test_validate_chunks_some_missing(self):
        """Test validation when some documents lack chunks."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id_1 = uuid4()
        doc_id_2 = uuid4()
        doc_id_3 = uuid4()
        document_ids = [doc_id_1, doc_id_2, doc_id_3]

        chunk_1 = Mock(spec=Chunk)

        mock_doc_1 = Mock(spec=Document, id=doc_id_1, chunks=[chunk_1])
        mock_doc_2 = Mock(spec=Document, id=doc_id_2, chunks=[])  # No chunks
        mock_doc_3 = Mock(spec=Document, id=doc_id_3, chunks=None)  # No chunks

        document_repository.find_by_id.side_effect = [mock_doc_1, mock_doc_2, mock_doc_3]

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        result = await task._validate_chunk_availability(document_ids)

        assert result["total_documents"] == 3
        assert result["documents_with_chunks"] == 1
        assert result["total_chunks"] == 1
        assert len(result["documents_without_chunks"]) == 2
        assert str(doc_id_2) in result["documents_without_chunks"]
        assert str(doc_id_3) in result["documents_without_chunks"]

    @pytest.mark.asyncio
    async def test_validate_chunks_document_not_found(self):
        """Test validation when document doesn't exist in repository."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id_1 = uuid4()
        document_ids = [doc_id_1]

        document_repository.find_by_id.return_value = None

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        result = await task._validate_chunk_availability(document_ids)

        assert result["total_documents"] == 1
        assert result["documents_with_chunks"] == 0
        assert result["total_chunks"] == 0
        assert str(doc_id_1) in result["documents_without_chunks"]

    @pytest.mark.asyncio
    async def test_validate_chunks_repository_error(self):
        """Test error handling when repository fails during validation."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id_1 = uuid4()
        document_ids = [doc_id_1]

        document_repository.find_by_id.side_effect = Exception("Database error")

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        with pytest.raises(
            EmbeddingPipelineError, match="Chunk validation failed"
        ):
            await task._validate_chunk_availability(document_ids)


class TestEmbeddingExecution:
    """Test execute method for embedding generation."""

    @pytest.mark.asyncio
    async def test_execute_success_basic(self):
        """Test basic successful embedding generation."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        # Mock embedding service
        embedding_service = AsyncMock()
        embedding_service.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        # Mock processed documents
        chunk = Mock(spec=Chunk)
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[chunk])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        # Mock pipeline response
        dedup_stats = DeduplicationStats(
            total_chunks=1,
            unique_chunks=1,
            duplicates_removed=0,
            duplicates_percentage=0.0,
        )
        pipeline_response = Mock(
            success=True,
            deduplication_stats=dedup_stats,
            results={"embeddings_generated": 1, "embeddings_stored": 1},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            # Create properly configured mock spec with chunking options
            spec = Mock(spec=PipelineSpec)
            spec.chunking = ChunkingOptions(preset="balanced", enable_semantic_chunking=None)
            result = await task.execute(spec)

            assert result.success is True
            assert result.phase_result.success is True
            assert result.phase_result.documents_processed == 1
            assert result.phase_result.embeddings_stored == 1

    @pytest.mark.asyncio
    async def test_execute_no_documents(self):
        """Test execution when no processed documents exist."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        document_repository.find_processed_documents.return_value = []

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        spec = Mock(spec=PipelineSpec)
        result = await task.execute(spec)

        assert result.success is True
        assert result.phase_result.documents_processed == 0
        assert result.phase_result.embeddings_stored == 0
        assert len(result.phase_result.warnings) == 1

    @pytest.mark.asyncio
    async def test_execute_no_chunks_available(self):
        """Test execution when documents have no chunks (Task 2 failed)."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        # Mock processed documents without chunks
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        spec = Mock(spec=PipelineSpec)
        result = await task.execute(spec)

        assert result.success is False
        assert result.phase_result.success is False
        assert len(result.phase_result.errors) > 0
        assert "No chunks available" in result.phase_result.errors[0]

    @pytest.mark.asyncio
    async def test_execute_embedding_initialization_fails(self):
        """Test error handling when embedding service fails to initialize."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock(
            side_effect=Exception("Model loading failed")
        )

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        mock_doc = Mock(spec=Document, id=doc_id, chunks=[Mock(spec=Chunk)])
        document_repository.find_processed_documents.return_value = [mock_doc]

        task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        spec = Mock(spec=PipelineSpec)
        result = await task.execute(spec)

        assert result.success is False
        assert len(result.phase_result.errors) > 0
        assert "Failed to initialize embedding model" in result.phase_result.errors[0]

    @pytest.mark.asyncio
    async def test_execute_with_deduplication(self):
        """Test execution with deduplication enabled."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        chunk = Mock(spec=Chunk)
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[chunk])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        # Mock deduplication stats
        dedup_stats = DeduplicationStats(
            total_chunks=100,
            unique_chunks=85,
            duplicates_removed=15,
            duplicates_percentage=15.0,
        )
        pipeline_response = Mock(
            success=True,
            deduplication_stats=dedup_stats,
            results={"embeddings_generated": 85, "embeddings_stored": 85},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            # Create properly configured mock spec with chunking options
            spec = Mock(spec=PipelineSpec)
            spec.chunking = ChunkingOptions(preset="balanced", enable_semantic_chunking=None)
            result = await task.execute(spec, deduplication_enabled=True)

            assert result.success is True
            assert result.phase_result.total_chunks == 100
            assert result.phase_result.unique_chunks == 85
            assert result.phase_result.duplicates_removed == 15

    @pytest.mark.asyncio
    async def test_execute_with_batch_size(self):
        """Test execution with custom batch size."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        chunk = Mock(spec=Chunk)
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[chunk])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        pipeline_response = Mock(
            success=True,
            deduplication_stats=DeduplicationStats(1, 1, 0, 0.0),
            results={"embeddings_generated": 1, "embeddings_stored": 1},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            # Create properly configured mock spec with chunking options
            spec = Mock(spec=PipelineSpec)
            spec.chunking = ChunkingOptions(preset="balanced", enable_semantic_chunking=None)
            result = await task.execute(spec, batch_size=64)

            assert result.success is True
            # Verify the batch size was passed to the pipeline
            call_args = mock_instance.execute.call_args
            assert call_args[0][0].batch_size == 64

    @pytest.mark.asyncio
    async def test_execute_with_chunk_load_policy(self):
        """Test execution with different chunk load policies."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        chunk = Mock(spec=Chunk)
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[chunk])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        pipeline_response = Mock(
            success=True,
            deduplication_stats=DeduplicationStats(1, 1, 0, 0.0),
            results={"embeddings_generated": 1, "embeddings_stored": 1},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            # Create properly configured mock spec with chunking options
            spec = Mock(spec=PipelineSpec)
            spec.chunking = ChunkingOptions(preset="balanced", enable_semantic_chunking=None)
            result = await task.execute(
                spec, chunk_load_policy=ChunkLoadPolicy.SEMANTIC_FALLBACK
            )

            assert result.success is True
            call_args = mock_instance.execute.call_args
            assert (
                call_args[0][0].chunk_load_policy == ChunkLoadPolicy.SEMANTIC_FALLBACK
            )


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation."""
        result = EmbeddingResult(
            success=True,
            documents_processed=10,
            total_chunks=50,
            unique_chunks=45,
            duplicates_removed=5,
            embeddings_generated=45,
            embeddings_stored=45,
            processing_time_seconds=5.5,
        )

        assert result.success is True
        assert result.documents_processed == 10
        assert result.total_chunks == 50
        assert result.unique_chunks == 45
        assert result.duplicates_removed == 5

    def test_embedding_result_as_dict(self):
        """Test EmbeddingResult conversion to dictionary."""
        result = EmbeddingResult(
            success=True,
            documents_processed=10,
            total_chunks=50,
            unique_chunks=45,
            duplicates_removed=5,
            embeddings_generated=45,
            embeddings_stored=45,
            processing_time_seconds=5.5,
        )

        result_dict = result.as_dict()

        assert result_dict["success"] is True
        assert result_dict["documents_processed"] == 10
        assert result_dict["total_chunks"] == 50
        assert result_dict["unique_chunks"] == 45
        assert result_dict["duplicates_removed"] == 5
        assert result_dict["embeddings_generated"] == 45
        assert result_dict["embeddings_stored"] == 45
        assert result_dict["duplicates_percentage"] == 10.0  # 5/50 * 100
        assert result_dict["chunks_per_document"] == 5.0  # 50/10
        assert result_dict["embedding_success_rate"] == 100.0  # 45/45 * 100

    def test_embedding_result_duplicates_percentage_calculation(self):
        """Test duplicates percentage calculation in result dict."""
        result = EmbeddingResult(
            success=True,
            documents_processed=5,
            total_chunks=100,
            unique_chunks=75,
            duplicates_removed=25,
            embeddings_generated=75,
            embeddings_stored=75,
            processing_time_seconds=2.0,
        )

        result_dict = result.as_dict()

        assert result_dict["duplicates_percentage"] == 25.0  # 25/100 * 100

    def test_embedding_result_zero_documents(self):
        """Test result dict when no documents processed."""
        result = EmbeddingResult(
            success=True,
            documents_processed=0,
            total_chunks=0,
            unique_chunks=0,
            duplicates_removed=0,
            embeddings_generated=0,
            embeddings_stored=0,
            processing_time_seconds=0.0,
        )

        result_dict = result.as_dict()

        assert result_dict["chunks_per_document"] == 0
        assert result_dict["embedding_success_rate"] == 0


class TestRagEmbeddingTaskReport:
    """Test RagEmbeddingTaskReport dataclass."""

    def test_report_creation(self):
        """Test report creation."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)

        embedding_result = EmbeddingResult(
            success=True,
            documents_processed=5,
            total_chunks=10,
            unique_chunks=10,
            duplicates_removed=0,
            embeddings_generated=10,
            embeddings_stored=10,
            processing_time_seconds=2.5,
        )

        report = RagEmbeddingTaskReport(
            success=True,
            execution_time_seconds=2.5,
            start_time=start_time,
            end_time=end_time,
            phase_result=embedding_result,
            total_errors=0,
            total_warnings=0,
        )

        assert report.success is True
        assert report.execution_time_seconds == 2.5
        assert report.phase_result is embedding_result

    def test_report_as_dict(self):
        """Test report conversion to dictionary."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)

        embedding_result = EmbeddingResult(
            success=True,
            documents_processed=5,
            total_chunks=10,
            unique_chunks=10,
            duplicates_removed=0,
            embeddings_generated=10,
            embeddings_stored=10,
            processing_time_seconds=2.5,
        )

        report = RagEmbeddingTaskReport(
            success=True,
            execution_time_seconds=2.5,
            start_time=start_time,
            end_time=end_time,
            phase_result=embedding_result,
            total_errors=0,
            total_warnings=0,
        )

        report_dict = report.as_dict()

        assert report_dict["task_name"] == "Embedding Generation"
        assert report_dict["success"] is True
        assert report_dict["total_errors"] == 0
        assert report_dict["total_warnings"] == 0
        assert report_dict["phase_result"]["documents_processed"] == 5

    def test_report_as_json(self):
        """Test report conversion to JSON string."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)

        embedding_result = EmbeddingResult(
            success=True,
            documents_processed=5,
            total_chunks=10,
            unique_chunks=10,
            duplicates_removed=0,
            embeddings_generated=10,
            embeddings_stored=10,
            processing_time_seconds=2.5,
        )

        report = RagEmbeddingTaskReport(
            success=True,
            execution_time_seconds=2.5,
            start_time=start_time,
            end_time=end_time,
            phase_result=embedding_result,
            total_errors=0,
            total_warnings=0,
        )

        json_str = report.as_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["task_name"] == "Embedding Generation"
        assert parsed["success"] is True


class TestEmbeddingPipelineError:
    """Test EmbeddingPipelineError exception."""

    def test_error_creation(self):
        """Test error creation."""
        error = EmbeddingPipelineError("Pipeline execution failed")

        assert str(error) == "Pipeline execution failed"

    def test_error_inheritance(self):
        """Test that error is an Exception."""
        error = EmbeddingPipelineError("Test error")

        assert isinstance(error, Exception)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_execute_with_max_retries(self):
        """Test execution with custom max retries."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        chunk = Mock(spec=Chunk)
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[chunk])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        pipeline_response = Mock(
            success=True,
            deduplication_stats=DeduplicationStats(1, 1, 0, 0.0),
            results={"embeddings_generated": 1, "embeddings_stored": 1},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            # Create properly configured mock spec with chunking options
            spec = Mock(spec=PipelineSpec)
            spec.chunking = ChunkingOptions(preset="balanced", enable_semantic_chunking=None)
            result = await task.execute(spec, max_retries=5)

            assert result.success is True
            call_args = mock_instance.execute.call_args
            assert call_args[0][0].max_retries == 5

    @pytest.mark.asyncio
    async def test_execute_timing_calculation(self):
        """Test that execution timing is calculated correctly."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        doc_id = uuid4()

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        chunk = Mock(spec=Chunk)
        mock_doc = Mock(spec=Document, id=doc_id, chunks=[chunk])
        document_repository.find_processed_documents.return_value = [mock_doc]
        document_repository.find_by_id.return_value = mock_doc

        pipeline_response = Mock(
            success=True,
            deduplication_stats=DeduplicationStats(1, 1, 0, 0.0),
            results={"embeddings_generated": 1, "embeddings_stored": 1},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            spec = Mock(spec=PipelineSpec)
            result = await task.execute(spec)

            assert result.execution_time_seconds >= 0
            assert result.start_time is not None
            assert result.end_time is not None

    @pytest.mark.asyncio
    async def test_execute_large_document_set(self):
        """Test execution with large number of documents."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = AsyncMock()

        # Create 100 documents
        doc_ids = [uuid4() for _ in range(100)]
        mock_docs = [
            Mock(spec=Document, id=doc_id, chunks=[Mock(spec=Chunk)])
            for doc_id in doc_ids
        ]

        embedding_service = AsyncMock()
        embedding_service.model_name = "test-model"
        embedding_service.initialize = AsyncMock()

        external_services.sentence_transformer_service = embedding_service
        external_services.embedding_repository = Mock()
        external_services.embedding_domain_service = Mock()
        external_services.embedding_model = Mock()

        document_repository.find_processed_documents.return_value = mock_docs
        document_repository.find_by_id.side_effect = mock_docs

        pipeline_response = Mock(
            success=True,
            deduplication_stats=DeduplicationStats(100, 100, 0, 0.0),
            results={"embeddings_generated": 100, "embeddings_stored": 100},
            processing_stats={},
            errors=[],
        )

        with patch(
            "app.pipeline.phases.rag_embedding_task.BuildIngestionPipelineUseCase"
        ) as mock_pipeline:
            mock_instance = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=pipeline_response)
            mock_pipeline.return_value = mock_instance

            task = RagEmbeddingTask(
                external_services=external_services,
                document_repository=document_repository,
            )

            # Create properly configured mock spec with chunking options
            spec = Mock(spec=PipelineSpec)
            spec.chunking = ChunkingOptions(preset="balanced", enable_semantic_chunking=None)
            result = await task.execute(spec, document_limit=100)

            assert result.success is True
            assert result.phase_result.documents_processed == 100
