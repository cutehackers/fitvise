"""Integration tests for RecursiveChunkingUseCase (Task 2.1.3)."""

from uuid import uuid4

import pytest

from app.application.use_cases.chunking.recursive_chunking import (
    RecursiveChunkingRequest,
    RecursiveChunkingResponse,
    RecursiveChunkingUseCase,
)
from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.document_metadata import DocumentFormat, DocumentMetadata
from app.infrastructure.external_services.ml_services.chunking_services.llama_hierarchical_chunker import (
    HierarchicalChunker,
    LlamaHierarchicalChunker,
)
from tests.fixtures.hierarchical_documents.sample_documents import (
    POLICY_DOCUMENT,
    SHORT_POLICY,
)


class MockDocumentRepository:
    """Mock repository for testing."""

    def __init__(self):
        self.documents = {}

    async def find_by_id(self, doc_id):
        return self.documents.get(doc_id)

    async def save(self, document):
        self.documents[document.id] = document

    async def find_processed_documents(self):
        return [doc for doc in self.documents.values() if doc.extracted_text]

    async def find_failed_documents(self):
        return []


@pytest.fixture
def mock_repository():
    """Fixture providing mock repository."""
    return MockDocumentRepository()


@pytest.fixture
def sample_document():
    """Fixture providing a sample policy document."""
    doc_id = uuid4()
    metadata = DocumentMetadata(
        file_name="policy.md",
        file_path="/docs/policy.md",
        format=DocumentFormat.TEXT,
        language="en",
    )
    document = Document(
        id=doc_id,
        source_id=uuid4(),
        content=POLICY_DOCUMENT,
        metadata=metadata,
    )
    document.mark_as_processed(extracted_text=POLICY_DOCUMENT)
    return document


@pytest.mark.asyncio
class TestRecursiveChunkingUseCase:
    """Integration tests for RecursiveChunkingUseCase."""

    async def test_execute_with_single_document(self, mock_repository, sample_document):
        """Test recursive chunking execution with a single document."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id])

        response = await use_case.execute(request)

        assert response.success is True
        assert response.total_chunks > 0
        assert len(response.results) == 1
        assert response.results[0].document_id == sample_document.id
        assert response.results[0].chunk_count > 0
        assert len(response.results[0].hierarchy_stats) > 0

    async def test_execute_creates_hierarchical_chunks(self, mock_repository, sample_document):
        """Test that execution creates chunks with hierarchy metadata."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id])

        response = await use_case.execute(request)

        # Check that hierarchy statistics are present
        assert response.total_hierarchy_stats
        assert any(key.startswith("depth_") for key in response.total_hierarchy_stats.keys())

        # Verify document was updated with chunks
        updated_doc = await mock_repository.find_by_id(sample_document.id)
        assert updated_doc.chunks
        assert len(updated_doc.chunks) == response.total_chunks

    async def test_execute_with_custom_config(self, mock_repository, sample_document):
        """Test execution with custom chunker configuration."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        custom_config = {"chunk_sizes": [1024, 256], "chunk_overlap": 50}
        request = RecursiveChunkingRequest(
            document_ids=[sample_document.id], chunker_config=custom_config
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.total_chunks > 0

    async def test_execute_dry_run_mode(self, mock_repository, sample_document):
        """Test execution in dry run mode doesn't persist chunks."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id], dry_run=True)

        response = await use_case.execute(request)

        assert response.success is True
        assert response.dry_run is True
        assert response.total_chunks > 0

        # Verify document chunks were NOT updated
        updated_doc = await mock_repository.find_by_id(sample_document.id)
        assert not updated_doc.chunks

    async def test_execute_with_no_documents(self, mock_repository):
        """Test execution with no documents returns empty response."""
        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(document_ids=[uuid4()])  # Non-existent ID

        response = await use_case.execute(request)

        assert response.success is True
        assert response.total_chunks == 0
        assert len(response.results) == 0

    async def test_execute_replaces_existing_chunks(self, mock_repository, sample_document):
        """Test that execution replaces existing chunks when requested."""
        sample_document.set_chunks([{"id": "old-chunk", "text": "old"}])
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(
            document_ids=[sample_document.id], replace_existing_chunks=True
        )

        response = await use_case.execute(request)

        assert response.success is True
        updated_doc = await mock_repository.find_by_id(sample_document.id)
        # Old chunks should be replaced
        assert not any(chunk.get("id") == "old-chunk" for chunk in updated_doc.chunks)

    async def test_execute_merges_with_existing_chunks(self, mock_repository, sample_document):
        """Test that execution merges with existing chunks when requested."""
        old_chunks = [{"id": "old-chunk", "text": "old"}]
        sample_document.set_chunks(old_chunks)
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(
            document_ids=[sample_document.id], replace_existing_chunks=False
        )

        response = await use_case.execute(request)

        assert response.success is True
        updated_doc = await mock_repository.find_by_id(sample_document.id)
        # Old chunks should still be present
        assert any(chunk.get("id") == "old-chunk" for chunk in updated_doc.chunks)
        # New chunks should be added
        assert len(updated_doc.chunks) > len(old_chunks)

    async def test_execute_with_metadata_overrides(self, mock_repository, sample_document):
        """Test execution with metadata overrides."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        metadata_overrides = {"custom_field": "custom_value"}
        request = RecursiveChunkingRequest(
            document_ids=[sample_document.id], metadata_overrides=metadata_overrides
        )

        response = await use_case.execute(request)

        assert response.success is True
        assert response.total_chunks > 0

    async def test_hierarchy_stats_calculation(self, mock_repository, sample_document):
        """Test that hierarchy statistics are calculated correctly."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id])

        response = await use_case.execute(request)

        # Verify hierarchy stats structure
        assert response.total_hierarchy_stats
        total_chunks_in_stats = sum(response.total_hierarchy_stats.values())
        assert total_chunks_in_stats == response.total_chunks

        # Verify document-level stats match
        assert response.results[0].hierarchy_stats == response.total_hierarchy_stats

    async def test_chunk_entity_conversion(self, mock_repository, sample_document):
        """Test that HierarchicalChunks are correctly converted to Chunk entities."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id])

        response = await use_case.execute(request)

        updated_doc = await mock_repository.find_by_id(sample_document.id)
        chunks = updated_doc.chunks

        # Verify chunk structure
        assert len(chunks) > 0
        first_chunk = chunks[0]
        assert "id" in first_chunk
        assert "text" in first_chunk
        assert "metadata" in first_chunk

        # Verify hierarchical metadata is present
        metadata = first_chunk["metadata"]
        assert "sequence" in metadata
        assert "start" in metadata
        assert "end" in metadata
        assert "extra" in metadata
        assert "depth_level" in metadata["extra"]

    async def test_execute_with_custom_chunker(self, mock_repository, sample_document):
        """Test execution with a custom chunker instance."""
        mock_repository.documents[sample_document.id] = sample_document

        custom_chunker = LlamaHierarchicalChunker()
        use_case = RecursiveChunkingUseCase(mock_repository, chunker=custom_chunker)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id])

        response = await use_case.execute(request)

        assert response.success is True
        assert response.total_chunks > 0

    async def test_execute_with_preset_config(self, mock_repository, sample_document):
        """Test execution with hierarchical preset configuration."""
        mock_repository.documents[sample_document.id] = sample_document

        use_case = RecursiveChunkingUseCase(mock_repository)
        # Use hierarchical preset by passing None (default)
        request = RecursiveChunkingRequest(document_ids=[sample_document.id])

        response = await use_case.execute(request)

        assert response.success is True
        assert response.total_chunks > 0
        # Should have multiple depth levels with default hierarchical config
        assert len(response.total_hierarchy_stats) >= 1


@pytest.mark.asyncio
async def test_end_to_end_recursive_chunking_workflow(mock_repository):
    """End-to-end test of the recursive chunking workflow."""
    # Create document
    doc_id = uuid4()
    metadata = DocumentMetadata(
        file_name="policy.md",
        file_path="/docs/policy.md",
        format=DocumentFormat.TEXT,
    )
    document = Document(
        id=doc_id,
        source_id=uuid4(),
        content=POLICY_DOCUMENT,
        metadata=metadata,
    )
    document.mark_as_processed(extracted_text=POLICY_DOCUMENT)
    mock_repository.documents[doc_id] = document

    # Execute chunking
    use_case = RecursiveChunkingUseCase(mock_repository)
    request = RecursiveChunkingRequest(document_ids=[doc_id])
    response = await use_case.execute(request)

    # Verify results
    assert response.success
    assert response.total_chunks > 0
    assert len(response.results) == 1

    # Verify document was updated
    updated_doc = await mock_repository.find_by_id(doc_id)
    assert updated_doc.chunks
    assert len(updated_doc.chunks) == response.total_chunks

    # Verify hierarchy preservation
    hierarchy_stats = response.total_hierarchy_stats
    assert len(hierarchy_stats) > 0  # Multiple depth levels
    assert all(key.startswith("depth_") for key in hierarchy_stats.keys())

    # Verify chunk structure
    first_chunk = updated_doc.chunks[0]
    assert first_chunk["text"]
    assert first_chunk["metadata"]["extra"]["depth_level"] >= 0
