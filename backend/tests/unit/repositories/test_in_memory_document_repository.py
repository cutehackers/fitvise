"""
Comprehensive unit tests for InMemoryDocumentRepository.

These tests validate the behavior and performance of the in-memory repository implementation,
ensuring it correctly implements the DocumentRepository interface and handles edge cases.
"""

import asyncio
import pytest
from typing import List
from uuid import uuid4, UUID
from datetime import datetime

from app.domain.entities.document import Document
from app.domain.value_objects.document_metadata import (
    DocumentMetadata,
    DocumentFormat,
    DocumentStatus,
)
from app.domain.value_objects.quality_metrics import DataQualityMetrics, ContentQualityMetrics
from app.infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository
from app.infrastructure.repositories.factory import RepositoryFactory


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Create sample document metadata for testing."""
    return DocumentMetadata(
        file_name="test_document.pdf",
        file_path="/docs/test_document.pdf",
        file_size=1024,
        format=DocumentFormat.PDF,
        status=DocumentStatus.PENDING,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_document(sample_metadata: DocumentMetadata) -> Document:
    """Create a sample document for testing."""
    return Document(
        source_id=uuid4(),
        metadata=sample_metadata,
        content="This is a test document content for testing purposes.",
    )


@pytest.fixture
def sample_quality_metrics() -> DataQualityMetrics:
    """Create sample quality metrics for testing."""
    return DataQualityMetrics(
        content_quality=ContentQualityMetrics(
            completeness_score=0.95,
            readability_score=0.88,
            structure_score=0.92,
            error_count=2,
            warning_count=1,
        ),
        overall_quality_score=0.92,
        relevance_score=0.87,
        confidence_score=0.90,
        validation_passed=True,
        validation_timestamp=datetime.utcnow(),
    )


class TestInMemoryDocumentRepository:
    """Test suite for InMemoryDocumentRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryDocumentRepository:
        """Create a new in-memory repository instance."""
        return InMemoryDocumentRepository()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_and_retrieve_document(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test basic save and retrieve operations."""
        # Save document
        saved_document = await repository.save(sample_document)

        # Verify returned document matches original
        assert saved_document.id == sample_document.id
        assert saved_document.content == sample_document.content
        assert saved_document.metadata.file_path == sample_document.metadata.file_path

        # Retrieve document
        retrieved_document = await repository.find_by_id(sample_document.id)
        assert retrieved_document is not None
        assert retrieved_document.id == sample_document.id
        assert retrieved_document.content == sample_document.content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_updates_metadata_status(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test that save updates document status correctly."""
        # Original status should be pending
        assert sample_document.metadata.status == DocumentStatus.PENDING

        # Save document
        saved_document = await repository.save(sample_document)

        # Status should remain pending (no processing happened yet)
        assert saved_document.metadata.status == DocumentStatus.PENDING

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_by_source_id(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding documents by source ID."""
        # Save document
        await repository.save(sample_document)

        # Find by source ID
        documents = await repository.find_by_source_id(sample_document.source_id)

        assert len(documents) == 1
        assert documents[0].id == sample_document.id

        # Find by non-existent source ID
        non_existent_documents = await repository.find_by_source_id(uuid4())
        assert len(non_existent_documents) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_by_file_path(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding documents by file path."""
        # Save document
        await repository.save(sample_document)

        # Find by file path
        found_document = await repository.find_by_file_path(sample_document.metadata.file_path)
        assert found_document is not None
        assert found_document.id == sample_document.id

        # Find by non-existent file path
        non_existent_document = await repository.find_by_file_path("/non/existent/path")
        assert non_existent_document is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_by_status(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding documents by status."""
        # Save document
        await repository.save(sample_document)

        # Find by pending status
        pending_documents = await repository.find_by_status(DocumentStatus.PENDING)
        assert len(pending_documents) == 1
        assert pending_documents[0].id == sample_document.id

        # Find by processed status (should be empty)
        processed_documents = await repository.find_by_status(DocumentStatus.PROCESSED)
        assert len(processed_documents) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_by_format(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding documents by format."""
        # Save document
        await repository.save(sample_document)

        # Find by PDF format
        pdf_documents = await repository.find_by_format(DocumentFormat.PDF)
        assert len(pdf_documents) == 1
        assert pdf_documents[0].id == sample_document.id

        # Find by TXT format (should be empty)
        txt_documents = await repository.find_by_format(DocumentFormat.TXT)
        assert len(txt_documents) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_by_categories(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding documents by categories."""
        # Add categories to document
        sample_document.set_predicted_categories(["test", "sample"], 0.85)

        # Save document
        await repository.save(sample_document)

        # Find by categories
        documents = await repository.find_by_categories(["test"])
        assert len(documents) == 1
        assert documents[0].id == sample_document.id

        # Find by non-existent categories
        empty_documents = await repository.find_by_categories(["nonexistent"])
        assert len(empty_documents) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_processed_documents(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding processed documents."""
        # Mark document as processed
        sample_document.complete_processing()
        await repository.save(sample_document)

        # Find processed documents
        processed_documents = await repository.find_processed_documents()
        assert len(processed_documents) == 1
        assert processed_documents[0].is_processed()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_failed_documents(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding failed documents."""
        # Mark document as failed
        sample_document.fail_processing("Test error")
        await repository.save(sample_document)

        # Find failed documents
        failed_documents = await repository.find_failed_documents()
        assert len(failed_documents) == 1
        assert failed_documents[0].has_failed()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_ready_for_rag(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test finding documents ready for RAG."""
        # Mark document as processed with embeddings and chunks
        sample_document.complete_processing()
        sample_document.set_embeddings([0.1, 0.2, 0.3])
        sample_document.set_chunks([{"text": "chunk 1", "tokens": 10}])
        await repository.save(sample_document)

        # Find RAG-ready documents
        rag_ready_documents = await repository.find_ready_for_rag()
        assert len(rag_ready_documents) == 1
        assert rag_ready_documents[0].is_ready_for_rag()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_needing_reprocessing(
        self, repository: InMemoryDocumentRepository
    ):
        """Test finding documents needing reprocessing."""
        # In-memory repository always returns empty list
        needing_reprocessing = await repository.find_needing_reprocessing()
        assert len(needing_reprocessing) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_find_by_quality_score_range(
        self,
        repository: InMemoryDocumentRepository,
        sample_document: Document,
        sample_quality_metrics: DataQualityMetrics,
    ):
        """Test finding documents by quality score range."""
        # Set quality metrics
        sample_document.set_quality_metrics(sample_quality_metrics)
        await repository.save(sample_document)

        # Find by quality score range
        documents = await repository.find_by_quality_score_range(0.8, 0.95)
        assert len(documents) == 1
        assert documents[0].quality_metrics.overall_quality_score == 0.92

        # Find by non-overlapping range
        empty_documents = await repository.find_by_quality_score_range(0.1, 0.5)
        assert len(empty_documents) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_document(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test document deletion."""
        # Save document
        await repository.save(sample_document)

        # Delete document
        deleted = await repository.delete(sample_document.id)
        assert deleted is True

        # Verify document is deleted
        retrieved_document = await repository.find_by_id(sample_document.id)
        assert retrieved_document is None

        # Try to delete non-existent document
        non_existent_delete = await repository.delete(uuid4())
        assert non_existent_delete is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_by_source_id(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test deletion by source ID."""
        # Save document
        await repository.save(sample_document)

        # Delete by source ID
        deleted_count = await repository.delete_by_source_id(sample_document.source_id)
        assert deleted_count == 1

        # Verify documents are deleted
        remaining_documents = await repository.find_by_source_id(sample_document.source_id)
        assert len(remaining_documents) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_all_documents(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test counting all documents."""
        # Count before adding document
        count_before = await repository.count_all()
        assert count_before == 0

        # Save document
        await repository.save(sample_document)

        # Count after adding document
        count_after = await repository.count_all()
        assert count_after == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_by_source_id(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test counting documents by source ID."""
        # Save document
        await repository.save(sample_document)

        # Count by source ID
        count = await repository.count_by_source_id(sample_document.source_id)
        assert count == 1

        # Count by non-existent source ID
        non_existent_count = await repository.count_by_source_id(uuid4())
        assert non_existent_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_by_status(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test counting documents by status."""
        # Save document
        await repository.save(sample_document)

        # Count by pending status
        count = await repository.count_by_status(DocumentStatus.PENDING)
        assert count == 1

        # Count by processed status (should be empty)
        processed_count = await repository.count_by_status(DocumentStatus.PROCESSED)
        assert processed_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_by_format(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test counting documents by format."""
        # Save document
        await repository.save(sample_document)

        # Count by PDF format
        count = await repository.count_by_format(DocumentFormat.PDF)
        assert count == 1

        # Count by TXT format (should be empty)
        txt_count = await repository.count_by_format(DocumentFormat.TXT)
        assert txt_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_processing_stats(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test getting processing statistics."""
        # Save document
        await repository.save(sample_document)

        # Get processing stats
        stats = await repository.get_processing_stats()

        assert stats["total"] == 1
        assert stats["processed"] == 0
        assert stats["failed"] == 0
        assert stats["pending"] == 1

        # Mark document as processed
        sample_document.complete_processing()
        await repository.save(sample_document)

        # Get updated stats
        updated_stats = await repository.get_processing_stats()
        assert updated_stats["processed"] == 1
        assert updated_stats["pending"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_documents_operations(
        self, repository: InMemoryDocumentRepository
    ):
        """Test operations with multiple documents."""
        # Create multiple documents
        documents = []
        for i in range(5):
            metadata = DocumentMetadata(
                file_name=f"doc_{i}.pdf",
                file_path=f"/docs/doc_{i}.pdf",
                file_size=1024,
                format=DocumentFormat.PDF,
                status=DocumentStatus.PENDING,
                created_at=datetime.utcnow(),
            )
            doc = Document(
                source_id=uuid4(),
                metadata=metadata,
                content=f"Document {i} content",
            )
            documents.append(doc)
            await repository.save(doc)

        # Verify all documents are saved
        assert await repository.count_all() == 5

        # Test finding multiple documents
        retrieved_docs = await repository.find_by_source_id(documents[0].source_id)
        assert len(retrieved_docs) == 1

        # Test bulk deletion
        deleted_count = await repository.delete_by_source_id(documents[0].source_id)
        assert deleted_count == 1

        # Verify remaining documents
        assert await repository.count_all() == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_document_handling(
        self, repository: InMemoryDocumentRepository
    ):
        """Test handling of large documents."""
        # Create a large document
        large_content = "Large document content. " * 1000  # ~25KB content
        metadata = DocumentMetadata(
            file_name="large_document.pdf",
            file_path="/docs/large_document.pdf",
            file_size=25600,  # 25KB
            format=DocumentFormat.PDF,
            status=DocumentStatus.PENDING,
            created_at=datetime.utcnow(),
        )

        document = Document(
            source_id=uuid4(),
            metadata=metadata,
            content=large_content,
        )

        # Save and retrieve large document
        saved_document = await repository.save(document)
        retrieved_document = await repository.find_by_id(document.id)

        assert retrieved_document is not None
        assert retrieved_document.content == large_content
        assert retrieved_document.metadata.file_size == 25600

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, repository: InMemoryDocumentRepository
    ):
        """Test concurrent document operations."""
        # Create multiple documents with same source ID
        source_id = uuid4()
        documents = []

        for i in range(10):
            metadata = DocumentMetadata(
                file_name=f"concurrent_doc_{i}.pdf",
                file_path=f"/docs/concurrent_doc_{i}.pdf",
                file_size=1024,
                format=DocumentFormat.PDF,
                status=DocumentStatus.PENDING,
                created_at=datetime.utcnow(),
            )
            doc = Document(
                source_id=source_id,
                metadata=metadata,
                content=f"Concurrent document {i} content",
            )
            documents.append(doc)

        # Concurrent save operations
        save_tasks = [repository.save(doc) for doc in documents]
        saved_documents = await asyncio.gather(*save_tasks)

        # Verify all documents were saved
        assert len(saved_documents) == 10

        # Concurrent read operations
        find_tasks = [repository.find_by_source_id(source_id) for _ in range(5)]
        results = await asyncio.gather(*find_tasks)

        # All concurrent reads should return same data
        for result in results:
            assert len(result) == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling(
        self, repository: InMemoryDocumentRepository
    ):
        """Test error handling in repository operations."""
        # Test finding non-existent document
        non_existent = await repository.find_by_id(uuid4())
        assert non_existent is None

        # Test deleting non-existent document
        deleted = await repository.delete(uuid4())
        assert deleted is False

        # Test deleting by non-existent source ID
        deleted_count = await repository.delete_by_source_id(uuid4())
        assert deleted_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repository_interface_compliance(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test that repository properly implements DocumentRepository interface."""
        # Test all required methods exist and are callable
        methods = [
            'save', 'find_by_id', 'find_by_source_id', 'find_by_file_path',
            'find_by_status', 'find_by_format', 'find_by_categories',
            'find_processed_documents', 'find_failed_documents', 'find_ready_for_rag',
            'find_needing_reprocessing', 'find_by_quality_score_range', 'delete',
            'delete_by_source_id', 'count_all', 'count_by_source_id', 'count_by_status',
            'count_by_format', 'get_processing_stats'
        ]

        for method_name in methods:
            assert hasattr(repository, method_name)
            method = getattr(repository, method_name)
            assert callable(method)

    @pytest.mark.unit
    def test_factory_selection_memory_repository(self):
        """Test that factory selects in-memory repository for memory URL."""
        # Set memory URL environment
        import os
        original_url = os.environ.get('DATABASE_URL')
        os.environ['DATABASE_URL'] = 'memory'

        try:
            # Reset factory cache
            RepositoryFactory._repositories.clear()

            # Get repository
            repo = RepositoryFactory.get_repository()
            assert isinstance(repo, InMemoryDocumentRepository)
        finally:
            # Restore original URL
            if original_url:
                os.environ['DATABASE_URL'] = original_url
            else:
                os.environ.pop('DATABASE_URL', None)


class TestInMemoryRepositoryPerformance:
    """Performance tests for InMemoryDocumentRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryDocumentRepository:
        """Create a new in-memory repository instance."""
        return InMemoryDocumentRepository()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_performance(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test save operation performance."""
        import time

        # Time save operation
        start_time = time.time()
        await repository.save(sample_document)
        end_time = time.time()

        save_duration = end_time - start_time
        assert save_duration < 0.1  # Should complete within 100ms

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_performance(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test retrieve operation performance."""
        # Save document first
        await repository.save(sample_document)

        # Time retrieve operation
        start_time = time.time()
        retrieved_document = await repository.find_by_id(sample_document.id)
        end_time = time.time()

        retrieve_duration = end_time - start_time
        assert retrieve_duration < 0.1  # Should complete within 100ms
        assert retrieved_document is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_dataset_performance(
        self, repository: InMemoryDocumentRepository
    ):
        """Test performance with large dataset."""
        import time

        # Create and save 100 documents
        documents = []
        for i in range(100):
            metadata = DocumentMetadata(
                file_name=f"bulk_doc_{i}.pdf",
                file_path=f"/docs/bulk_doc_{i}.pdf",
                file_size=1024,
                format=DocumentFormat.PDF,
                status=DocumentStatus.PENDING,
                created_at=datetime.utcnow(),
            )
            doc = Document(
                source_id=uuid4(),
                metadata=metadata,
                content=f"Document {i} content for bulk testing.",
            )
            documents.append(doc)
            await repository.save(doc)

        # Time count operation on large dataset
        start_time = time.time()
        count = await repository.count_all()
        end_time = time.time()

        count_duration = end_time - start_time
        assert count == 100
        assert count_duration < 0.1  # Should complete within 100ms