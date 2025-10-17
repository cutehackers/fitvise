from __future__ import annotations

import asyncio
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from app.application.use_cases.chunking.semantic_chunking import (
    SemanticChunkingRequest,
    SemanticChunkingUseCase,
)
from app.domain.entities.document import Document
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.document_metadata import DocumentFormat, DocumentMetadata
from app.infrastructure.external_services.ml_services.chunking_services.llama_index_chunker import (
    SemanticChunk,
)


class StubChunker:
    """Stub chunker returning deterministic chunks."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Optional[str]]] = []

    def chunk(self, text: str, *, metadata: Optional[Dict[str, str]] = None, document_metadata: Optional[Dict[str, str]] = None) -> List[SemanticChunk]:
        self.calls.append({"text": text, "document_id": metadata.get("document_id") if metadata else None})
        return [
            SemanticChunk(
                chunk_id="chunk-1",
                sequence=0,
                text=text[:20] or "fallback",
                start=0,
                end=min(len(text), 20),
                metadata=metadata or {},
            ),
            SemanticChunk(
                chunk_id="chunk-2",
                sequence=1,
                text=text[20:40] or "fallback-2",
                start=min(len(text), 20),
                end=min(len(text), 40),
                metadata=metadata or {},
            ),
        ]


class InMemoryDocumentRepository(DocumentRepository):
    """Lightweight in-memory repository for testing."""

    def __init__(self, documents: Optional[List[Document]] = None) -> None:
        self._documents: Dict[UUID, Document] = {doc.id: doc for doc in documents or []}
        self.saved_documents: List[Document] = []

    async def save(self, document: Document) -> Document:
        self._documents[document.id] = document
        self.saved_documents.append(document)
        return document

    async def find_by_id(self, document_id: UUID) -> Optional[Document]:
        return self._documents.get(document_id)

    async def find_by_source_id(self, source_id: UUID) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.source_id == source_id]

    async def find_by_file_path(self, file_path: str) -> Optional[Document]:
        return next((doc for doc in self._documents.values() if doc.metadata.file_path == file_path), None)

    async def find_by_status(self, status):
        return [doc for doc in self._documents.values() if doc.metadata.status == status]

    async def find_by_format(self, format):
        return [doc for doc in self._documents.values() if doc.metadata.format == format]

    async def find_by_categories(self, categories: List[str]) -> List[Document]:
        return [doc for doc in self._documents.values() if any(cat in doc.all_categories for cat in categories)]

    async def find_processed_documents(self) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.is_processed()]

    async def find_failed_documents(self) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.has_failed()]

    async def find_ready_for_rag(self) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.is_ready_for_rag()]

    async def find_needing_reprocessing(self) -> List[Document]:
        return []

    async def delete(self, document_id: UUID) -> bool:
        return self._documents.pop(document_id, None) is not None

    async def delete_by_source_id(self, source_id: UUID) -> int:
        to_delete = [doc_id for doc_id, doc in self._documents.items() if doc.source_id == source_id]
        for doc_id in to_delete:
            self._documents.pop(doc_id)
        return len(to_delete)

    async def count_all(self) -> int:
        return len(self._documents)

    async def count_by_source_id(self, source_id: UUID) -> int:
        return sum(1 for doc in self._documents.values() if doc.source_id == source_id)

    async def count_by_status(self, status) -> int:
        return sum(1 for doc in self._documents.values() if doc.metadata.status == status)

    async def count_by_format(self, format) -> int:
        return sum(1 for doc in self._documents.values() if doc.metadata.format == format)

    async def get_processing_stats(self) -> dict:
        return {}


def _make_document(extracted_text: str = "Section 1\nContent\n") -> Document:
    metadata = DocumentMetadata(
        file_name="doc.txt",
        file_path="/docs/doc.txt",
        file_size=64,
        format=DocumentFormat.TXT,
    )
    document = Document(source_id=uuid4(), metadata=metadata, content=extracted_text)
    document.complete_processing(extracted_text=extracted_text)
    return document


@pytest.mark.asyncio
async def test_execute_updates_chunks_and_persists():
    document = _make_document("This is a long enough piece of text to create multiple chunks.")
    repo = InMemoryDocumentRepository([document])
    chunker = StubChunker()
    use_case = SemanticChunkingUseCase(document_repository=repo, chunker=chunker)

    response = await use_case.execute(SemanticChunkingRequest())

    assert response.success is True
    assert response.total_chunks == 2
    assert len(repo.saved_documents) == 1
    saved_doc = repo.saved_documents[0]
    assert saved_doc.chunk_count == 2
    assert chunker.calls[0]["document_id"] == str(document.id)


@pytest.mark.asyncio
async def test_execute_dry_run_does_not_persist_changes():
    document = _make_document("Dry run keeps existing state.")
    repo = InMemoryDocumentRepository([document])
    chunker = StubChunker()
    use_case = SemanticChunkingUseCase(document_repository=repo, chunker=chunker)

    response = await use_case.execute(SemanticChunkingRequest(dry_run=True))

    assert response.dry_run is True
    assert len(repo.saved_documents) == 0
    assert document.chunk_count == 0


@pytest.mark.asyncio
async def test_execute_append_chunks_when_replace_disabled():
    document = _make_document("Original text for merging.")
    document.set_chunks(
        [
            {
                "id": "existing-1",
                "sequence": 0,
                "text": "Existing chunk",
                "start": 0,
                "end": 14,
                "metadata": {"document_id": str(document.id)},
            }
        ]
    )
    repo = InMemoryDocumentRepository([document])
    chunker = StubChunker()
    use_case = SemanticChunkingUseCase(document_repository=repo, chunker=chunker)

    await use_case.execute(SemanticChunkingRequest(replace_existing_chunks=False))

    saved_doc = repo.saved_documents[0]
    assert saved_doc.chunk_count == 3  # original + two new
    chunk_ids = {chunk["id"] for chunk in saved_doc.chunks}
    assert "existing-1" in chunk_ids
    assert "chunk-1" in chunk_ids
