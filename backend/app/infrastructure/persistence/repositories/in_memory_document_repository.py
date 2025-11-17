"""In-memory implementation of DocumentRepository for orchestration flows."""
from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from app.domain.entities.document import Document
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.document_metadata import DocumentFormat, DocumentStatus


class InMemoryDocumentRepository(DocumentRepository):
    """Simple in-memory repository used by the ingestion pipeline."""

    def __init__(self) -> None:
        self._documents: Dict[UUID, Document] = {}

    async def save(self, document: Document) -> Document:
        self._documents[document.id] = document
        return document

    async def find_by_id(self, document_id: UUID) -> Optional[Document]:
        return self._documents.get(document_id)

    async def find_by_source_id(self, source_id: UUID) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.source_id == source_id]

    async def find_by_file_path(self, file_path: str) -> Optional[Document]:
        for document in self._documents.values():
            if document.metadata.file_path == file_path:
                return document
        return None

    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.metadata.status == status]

    async def find_by_format(self, format: DocumentFormat) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.metadata.format == format]

    async def find_by_categories(self, categories: List[str]) -> List[Document]:
        return [
            doc
            for doc in self._documents.values()
            if any(category in doc.all_categories for category in categories)
        ]

    async def find_processed_documents(self) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.is_processed()]

    async def find_failed_documents(self) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.has_failed()]

    async def find_ready_for_rag(self) -> List[Document]:
        return [doc for doc in self._documents.values() if doc.is_ready_for_rag()]

    async def find_needing_reprocessing(self) -> List[Document]:
        return []

    async def find_by_quality_score_range(self, min_score: float, max_score: float) -> List[Document]:
        return [
            doc
            for doc in self._documents.values()
            if doc.quality_metrics and min_score <= doc.quality_metrics.overall_score <= max_score
        ]

    async def delete(self, document_id: UUID) -> bool:
        return self._documents.pop(document_id, None) is not None

    async def delete_by_source_id(self, source_id: UUID) -> int:
        keys = [doc_id for doc_id, doc in self._documents.items() if doc.source_id == source_id]
        for key in keys:
            self._documents.pop(key, None)
        return len(keys)

    async def count_all(self) -> int:
        return len(self._documents)

    async def count_by_source_id(self, source_id: UUID) -> int:
        return sum(1 for doc in self._documents.values() if doc.source_id == source_id)

    async def count_by_status(self, status: DocumentStatus) -> int:
        return sum(1 for doc in self._documents.values() if doc.metadata.status == status)

    async def count_by_format(self, format: DocumentFormat) -> int:
        return sum(1 for doc in self._documents.values() if doc.metadata.format == format)

    async def get_processing_stats(self) -> dict:
        processed = await self.count_by_status(DocumentStatus.PROCESSED)
        failed = await self.count_by_status(DocumentStatus.FAILED)
        pending = await self.count_by_status(DocumentStatus.PENDING)
        return {
            "total": await self.count_all(),
            "processed": processed,
            "failed": failed,
            "pending": pending,
        }
