"""Pytest configuration and fixtures for indexing use case tests."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document
from app.domain.value_objects.chunk_metadata import ChunkMetadata
from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentFormat


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        source_id=uuid4(),
        title="Test Document",
        content="This is a test document with some content for testing purposes.",
        metadata=DocumentMetadata(
            file_name="test.txt",
            file_path="/test/test.txt",
            format=DocumentFormat.TEXT,
            language="en",
            title="Test Document",
            author="Test Author",
            keywords=["test", "document"],
            custom_fields={}
        )
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = []
    content_parts = [
        "This is the first chunk of content.",
        "This is the second chunk with different content.",
        "This is the third chunk of the test document."
    ]

    for i, content in enumerate(content_parts):
        chunk = Chunk(
            chunk_id=str(uuid4()),
            document_id=uuid4(),
            text=content,
            metadata=ChunkMetadata(
                sequence=i,
                start=i * len(content),
                end=(i + 1) * len(content),
                token_count=len(content.split()),
                source_type="semantic"
            )
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def duplicate_chunks():
    """Create chunks with duplicate content for deduplication testing."""
    chunks = []

    # First chunk
    chunk1 = Chunk(
        chunk_id=str(uuid4()),
        document_id=uuid4(),
        text="Duplicate content here",
        metadata=ChunkMetadata(sequence=1, start=0, end=21, token_count=3, source_type="semantic")
    )

    # Duplicate content with different ID
    chunk2 = Chunk(
        chunk_id=str(uuid4()),
        document_id=uuid4(),
        text="Duplicate content here",
        metadata=ChunkMetadata(sequence=1, start=0, end=21, token_count=3, source_type="semantic")
    )

    # Unique content
    chunk3 = Chunk(
        chunk_id=str(uuid4()),
        document_id=uuid4(),
        text="Unique content here",
        metadata=ChunkMetadata(sequence=1, start=0, end=19, token_count=3, source_type="semantic")
    )

    chunks = [chunk1, chunk2, chunk3]
    return chunks


@pytest.fixture
def mock_chunking_use_case():
    """Create a mock chunking use case."""
    mock_use_case = AsyncMock()
    mock_response = Mock()
    mock_response.success = True
    mock_response.chunks = []
    mock_use_case.execute.return_value = mock_response
    return mock_use_case


@pytest.fixture
def mock_embedding_use_case():
    """Create a mock embedding use case."""
    mock_use_case = AsyncMock()
    mock_response = Mock()
    mock_response.embedded_count = 0
    mock_response.stored_count = 0
    mock_response.success = True
    mock_use_case.execute.return_value = mock_response
    return mock_use_case


@pytest.fixture
def mock_document_repository():
    """Create a mock document repository."""
    return AsyncMock()