"""Document metadata value object for RAG system."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class DocumentFormat(str, Enum):
    """File formats supported by RAG pipeline - PDF, DOCX, HTML, TXT, MD, CSV, JSON, XML."""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    JSON = "json"
    XML = "xml"


class DocumentStatus(str, Enum):
    """Processing pipeline states - pending → processing → processed/failed/archived."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class DocumentMetadata:
    """Immutable doc metadata - file info, content stats, processing status & timestamps.
    
    Examples:
        >>> metadata = DocumentMetadata("contract.pdf", "/legal/contract.pdf", 2048, 
        ...                             DocumentFormat.PDF, title="Service Contract")
        >>> metadata.is_processed
        False
        >>> updated = metadata.with_status(DocumentStatus.PROCESSED)
        >>> updated.is_processed
        True
    """
    
    # Basic identification
    file_name: str
    file_path: str
    file_size: int  # in bytes
    format: DocumentFormat
    
    # Content metadata
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    keywords: List[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    
    # Processing metadata
    status: DocumentStatus = DocumentStatus.PENDING
    processing_time: Optional[float] = None  # in seconds
    error_message: Optional[str] = None
    
    # Content analysis
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    encoding: Optional[str] = None
    
    # Custom metadata
    custom_fields: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.keywords is None:
            object.__setattr__(self, 'keywords', [])
        if self.custom_fields is None:
            object.__setattr__(self, 'custom_fields', {})
    
    @property
    def file_extension(self) -> str:
        """Get file extension from filename."""
        return self.file_name.split('.')[-1].lower() if '.' in self.file_name else ''
    
    @property
    def is_processed(self) -> bool:
        """Check if document has been processed successfully."""
        return self.status == DocumentStatus.PROCESSED
    
    @property
    def has_error(self) -> bool:
        """Check if document processing failed."""
        return self.status == DocumentStatus.FAILED
    
    def with_status(self, status: DocumentStatus, error_message: Optional[str] = None) -> 'DocumentMetadata':
        """Create new metadata with updated status."""
        return DocumentMetadata(
            file_name=self.file_name,
            file_path=self.file_path,
            file_size=self.file_size,
            format=self.format,
            title=self.title,
            author=self.author,
            language=self.language,
            keywords=self.keywords.copy(),
            created_at=self.created_at,
            modified_at=self.modified_at,
            accessed_at=self.accessed_at,
            status=status,
            processing_time=self.processing_time,
            error_message=error_message,
            word_count=self.word_count,
            page_count=self.page_count,
            encoding=self.encoding,
            custom_fields=self.custom_fields.copy()
        )
    
    def add_keyword(self, keyword: str) -> 'DocumentMetadata':
        """Create new metadata with additional keyword."""
        new_keywords = self.keywords.copy()
        if keyword not in new_keywords:
            new_keywords.append(keyword)
        
        return DocumentMetadata(
            file_name=self.file_name,
            file_path=self.file_path,
            file_size=self.file_size,
            format=self.format,
            title=self.title,
            author=self.author,
            language=self.language,
            keywords=new_keywords,
            created_at=self.created_at,
            modified_at=self.modified_at,
            accessed_at=self.accessed_at,
            status=self.status,
            processing_time=self.processing_time,
            error_message=self.error_message,
            word_count=self.word_count,
            page_count=self.page_count,
            encoding=self.encoding,
            custom_fields=self.custom_fields.copy()
        )