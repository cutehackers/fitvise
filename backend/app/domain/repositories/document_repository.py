"""Document repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from app.domain.entities.document import Document
from app.domain.value_objects.document_metadata import DocumentStatus, DocumentFormat


class DocumentRepository(ABC):
    """Abstract repository for document persistence."""
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """Save a document."""
        pass
    
    @abstractmethod
    async def find_by_id(self, document_id: UUID) -> Optional[Document]:
        """Find a document by ID."""
        pass
    
    @abstractmethod
    async def find_by_source_id(self, source_id: UUID) -> List[Document]:
        """Find documents by source ID."""
        pass
    
    @abstractmethod
    async def find_by_file_path(self, file_path: str) -> Optional[Document]:
        """Find a document by file path."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        """Find documents by processing status."""
        pass
    
    @abstractmethod
    async def find_by_format(self, format: DocumentFormat) -> List[Document]:
        """Find documents by format."""
        pass
    
    @abstractmethod
    async def find_by_categories(self, categories: List[str]) -> List[Document]:
        """Find documents by categories."""
        pass
    
    @abstractmethod
    async def find_processed_documents(self) -> List[Document]:
        """Find all processed documents."""
        pass
    
    @abstractmethod
    async def find_failed_documents(self) -> List[Document]:
        """Find documents with processing failures."""
        pass
    
    @abstractmethod
    async def find_ready_for_rag(self) -> List[Document]:
        """Find documents ready for RAG (with chunks and embeddings)."""
        pass
    
    @abstractmethod
    async def find_needing_reprocessing(self) -> List[Document]:
        """Find documents that need reprocessing."""
        pass
    
    @abstractmethod
    async def find_by_quality_score_range(self, min_score: float, max_score: float) -> List[Document]:
        """Find documents by quality score range."""
        pass
    
    @abstractmethod
    async def delete(self, document_id: UUID) -> bool:
        """Delete a document."""
        pass
    
    @abstractmethod
    async def delete_by_source_id(self, source_id: UUID) -> int:
        """Delete all documents from a source."""
        pass
    
    @abstractmethod
    async def count_all(self) -> int:
        """Count total number of documents."""
        pass
    
    @abstractmethod
    async def count_by_source_id(self, source_id: UUID) -> int:
        """Count documents by source ID."""
        pass
    
    @abstractmethod
    async def count_by_status(self, status: DocumentStatus) -> int:
        """Count documents by status."""
        pass
    
    @abstractmethod
    async def count_by_format(self, format: DocumentFormat) -> int:
        """Count documents by format."""
        pass
    
    @abstractmethod
    async def get_processing_stats(self) -> dict:
        """Get processing statistics."""
        pass