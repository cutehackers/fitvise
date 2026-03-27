"""Document domain entity for RAG system."""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentStatus
from app.domain.value_objects.quality_metrics import DataQualityMetrics


class Document:
    """Document entity - manages file content, processing status & ML categorization for RAG.
    
    Examples:
        >>> from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentFormat
        >>> metadata = DocumentMetadata("report.pdf", "/docs/report.pdf", 1024, DocumentFormat.PDF)
        >>> doc = Document(source_id, metadata, content="Annual report content...")
        >>> doc.start_processing()  # Begin document processing
        >>> doc.set_predicted_categories(["financial", "annual"], 0.85)
        >>> doc.is_ready_for_rag()  # Check if ready for retrieval
        False
    """
    
    def __init__(
        self,
        source_id: UUID,
        metadata: DocumentMetadata,
        content: Optional[str] = None,
        id: Optional[UUID] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """Initialize a document entity."""
        self._id = id or uuid4()
        self._source_id = source_id
        self._metadata = metadata
        self._content = content
        self._created_at = created_at or datetime.now(timezone.utc)
        self._updated_at = updated_at or datetime.now(timezone.utc)
        
        # Processing tracking
        self._processing_attempts: int = 0
        self._last_processed_at: Optional[datetime] = None
        self._processing_duration: Optional[float] = None
        
        # Content analysis
        self._extracted_text: Optional[str] = None
        self._structured_content: Optional[Dict[str, Any]] = None
        self._embeddings: Optional[List[float]] = None
        
        # Categorization
        self._predicted_categories: List[str] = []
        self._category_confidence: Optional[float] = None
        self._manual_categories: List[str] = []
        
        # Quality and validation
        self._quality_metrics: Optional[DataQualityMetrics] = None
        self._validation_errors: List[str] = []
        
        # Chunks for RAG
        self._chunks: List[Dict[str, Any]] = []
        self._chunk_count: int = 0
        
        # Version control
        self._version: int = 1
        self._checksum: Optional[str] = None
        
        # Validation
        self._validate()
    
    def _validate(self) -> None:
        """Validate the document entity."""
        if self._metadata.file_size < 0:
            raise ValueError("File size cannot be negative")
        
        if self._processing_attempts < 0:
            raise ValueError("Processing attempts cannot be negative")
        
        if self._chunk_count < 0:
            raise ValueError("Chunk count cannot be negative")
    
    # Properties (read-only)
    @property
    def id(self) -> UUID:
        """Get the unique identifier."""
        return self._id
    
    @property
    def source_id(self) -> UUID:
        """Get the source identifier."""
        return self._source_id
    
    @property
    def metadata(self) -> DocumentMetadata:
        """Get the document metadata."""
        return self._metadata
    
    @property
    def content(self) -> Optional[str]:
        """Get the raw content."""
        return self._content
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def processing_attempts(self) -> int:
        """Get number of processing attempts."""
        return self._processing_attempts
    
    @property
    def last_processed_at(self) -> Optional[datetime]:
        """Get last processing timestamp."""
        return self._last_processed_at
    
    @property
    def processing_duration(self) -> Optional[float]:
        """Get last processing duration in seconds."""
        return self._processing_duration
    
    @property
    def extracted_text(self) -> Optional[str]:
        """Get extracted text content."""
        return self._extracted_text
    
    @property
    def structured_content(self) -> Optional[Dict[str, Any]]:
        """Get structured content."""
        return self._structured_content
    
    @property
    def embeddings(self) -> Optional[List[float]]:
        """Get document embeddings."""
        return self._embeddings
    
    @property
    def predicted_categories(self) -> List[str]:
        """Get ML-predicted categories."""
        return self._predicted_categories.copy()
    
    @property
    def category_confidence(self) -> Optional[float]:
        """Get category prediction confidence."""
        return self._category_confidence
    
    @property
    def manual_categories(self) -> List[str]:
        """Get manually assigned categories."""
        return self._manual_categories.copy()
    
    @property
    def all_categories(self) -> List[str]:
        """Get all categories (manual + predicted)."""
        return list(set(self._manual_categories + self._predicted_categories))
    
    @property
    def quality_metrics(self) -> Optional[DataQualityMetrics]:
        """Get quality metrics."""
        return self._quality_metrics
    
    @property
    def validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()
    
    @property
    def chunks(self) -> List[Dict[str, Any]]:
        """Get document chunks."""
        return self._chunks.copy()
    
    @property
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return self._chunk_count
    
    @property
    def version(self) -> int:
        """Get document version."""
        return self._version
    
    @property
    def checksum(self) -> Optional[str]:
        """Get document checksum."""
        return self._checksum
    
    # Business methods
    def update_content(self, content: str, checksum: Optional[str] = None) -> None:
        """Update document content."""
        if content != self._content:
            self._content = content
            self._version += 1
            self._checksum = checksum
            self._updated_at = datetime.now(timezone.utc)
            
            # Reset processing-related fields when content changes
            self._extracted_text = None
            self._structured_content = None
            self._embeddings = None
            self._chunks.clear()
            self._chunk_count = 0
            
            # Update metadata status
            self._metadata = self._metadata.with_status(DocumentStatus.PENDING)
    
    def update_metadata(self, metadata: DocumentMetadata) -> None:
        """Update document metadata."""
        self._metadata = metadata
        self._updated_at = datetime.now(timezone.utc)
    
    def start_processing(self) -> None:
        """Mark document as starting processing."""
        self._processing_attempts += 1
        self._metadata = self._metadata.with_status(DocumentStatus.PROCESSING)
        self._last_processed_at = datetime.now(timezone.utc)
        self._updated_at = datetime.now(timezone.utc)
    
    def complete_processing(
        self,
        extracted_text: Optional[str] = None,
        structured_content: Optional[Dict[str, Any]] = None,
        processing_duration: Optional[float] = None
    ) -> None:
        """Mark document processing as completed."""
        self._extracted_text = extracted_text
        self._structured_content = structured_content
        self._processing_duration = processing_duration
        self._metadata = self._metadata.with_status(DocumentStatus.PROCESSED)
        self._updated_at = datetime.now(timezone.utc)
    
    def fail_processing(self, error_message: str) -> None:
        """Mark document processing as failed."""
        self._metadata = self._metadata.with_status(DocumentStatus.FAILED, error_message)
        self._validation_errors.append(f"{datetime.now(timezone.utc).isoformat()}: {error_message}")
        self._updated_at = datetime.now(timezone.utc)
    
    def set_embeddings(self, embeddings: List[float]) -> None:
        """Set document embeddings."""
        self._embeddings = embeddings.copy()
        self._updated_at = datetime.now(timezone.utc)
    
    def set_predicted_categories(self, categories: List[str], confidence: float) -> None:
        """Set ML-predicted categories."""
        self._predicted_categories = categories.copy()
        self._category_confidence = confidence
        self._updated_at = datetime.now(timezone.utc)
    
    def add_manual_category(self, category: str) -> None:
        """Add a manual category."""
        if category not in self._manual_categories:
            self._manual_categories.append(category)
            self._updated_at = datetime.now(timezone.utc)
    
    def remove_manual_category(self, category: str) -> None:
        """Remove a manual category."""
        if category in self._manual_categories:
            self._manual_categories.remove(category)
            self._updated_at = datetime.now(timezone.utc)
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Set document chunks for RAG."""
        self._chunks = chunks.copy()
        self._chunk_count = len(chunks)
        self._updated_at = datetime.now(timezone.utc)
    
    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a single chunk."""
        self._chunks.append(chunk)
        self._chunk_count = len(self._chunks)
        self._updated_at = datetime.now(timezone.utc)
    
    def set_quality_metrics(self, metrics: DataQualityMetrics) -> None:
        """Set quality metrics."""
        self._quality_metrics = metrics
        self._updated_at = datetime.now(timezone.utc)
    
    def add_validation_error(self, error: str) -> None:
        """Add a validation error."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self._validation_errors.append(f"{timestamp}: {error}")
        
        # Keep only the last 20 errors
        if len(self._validation_errors) > 20:
            self._validation_errors = self._validation_errors[-20:]
        
        self._updated_at = datetime.now(timezone.utc)
    
    def clear_validation_errors(self) -> None:
        """Clear all validation errors."""
        self._validation_errors.clear()
        self._updated_at = datetime.now(timezone.utc)
    
    def archive(self) -> None:
        """Archive the document."""
        self._metadata = self._metadata.with_status(DocumentStatus.ARCHIVED)
        self._updated_at = datetime.now(timezone.utc)
    
    # Status methods
    def is_processed(self) -> bool:
        """Check if document has been processed successfully."""
        return self._metadata.status == DocumentStatus.PROCESSED
    
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self._metadata.status == DocumentStatus.PROCESSING
    
    def has_failed(self) -> bool:
        """Check if document processing has failed."""
        return self._metadata.status == DocumentStatus.FAILED
    
    def is_ready_for_rag(self) -> bool:
        """Check if document is ready for RAG (has chunks and embeddings)."""
        return (self.is_processed() and 
                self._chunk_count > 0 and 
                self._embeddings is not None)
    
    def needs_reprocessing(self) -> bool:
        """Check if document needs reprocessing."""
        return (self._metadata.status in [DocumentStatus.PENDING, DocumentStatus.FAILED] or
                (self.is_processed() and self._chunk_count == 0))
    
    def get_processing_status(self) -> dict:
        """Get comprehensive processing status."""
        return {
            "status": self._metadata.status.value,
            "is_processed": self.is_processed(),
            "is_processing": self.is_processing(),
            "has_failed": self.has_failed(),
            "is_ready_for_rag": self.is_ready_for_rag(),
            "needs_reprocessing": self.needs_reprocessing(),
            "processing_attempts": self._processing_attempts,
            "last_processed_at": self._last_processed_at.isoformat() if self._last_processed_at else None,
            "processing_duration": self._processing_duration,
            "chunk_count": self._chunk_count,
            "has_embeddings": self._embeddings is not None,
            "validation_errors": len(self._validation_errors),
            "quality_score": self._quality_metrics.overall_quality_score if self._quality_metrics else None,
            "categories": self.all_categories,
            "version": self._version
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"Document(id={self._id}, file='{self._metadata.file_name}', status={self._metadata.status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Document(id={self._id}, source_id={self._source_id}, "
                f"file='{self._metadata.file_name}', status={self._metadata.status.value}, "
                f"chunks={self._chunk_count})")