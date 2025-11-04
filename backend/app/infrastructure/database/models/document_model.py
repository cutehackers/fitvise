"""SQLAlchemy model for Document entity."""
import uuid as uuid_pkg
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import ARRAY, CheckConstraint, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
from sqlalchemy.types import JSON as SQLAlchemyJSON

from app.infrastructure.database.database import Base


class DocumentModel(Base):
    """SQLAlchemy model for Document entity.

    This model supports both PostgreSQL (with native UUID, JSONB, ARRAY) and
    SQLite (with TEXT-based equivalents) through SQLAlchemy's type adapters.

    Table Structure:
        - Primary key: id (UUID)
        - Foreign key: source_id (UUID)
        - Text fields: content, extracted_text (large text)
        - JSON fields: structured_content, quality_metrics, chunks, metadata
        - Array fields: embeddings, predicted_categories, manual_categories, validation_errors
        - Timestamps: created_at, updated_at, last_processed_at
        - Processing: processing_attempts, processing_duration
        - Categorization: predicted_categories, category_confidence, manual_categories
        - Version control: version, checksum
    """

    __tablename__ = "documents"

    # Core identifiers
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid_pkg.uuid4,
        comment="Unique document identifier",
    )
    source_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Source/collection identifier",
    )

    # Content fields (large text)
    content = Column(Text, nullable=True, comment="Original raw content")
    extracted_text = Column(Text, nullable=True, comment="Extracted text content")

    # JSON fields (PostgreSQL: JSONB, SQLite: JSON as TEXT)
    structured_content = Column(
        JSONB().with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        comment="Structured content as JSON",
    )

    # Embeddings (PostgreSQL: native ARRAY, SQLite: JSON array)
    embeddings = Column(
        ARRAY(Float).with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        comment="Document embeddings (768 dimensions)",
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Creation timestamp",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last update timestamp",
    )
    last_processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last processing timestamp",
    )

    # Processing tracking
    processing_attempts = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of processing attempts",
    )
    processing_duration = Column(
        Float,
        nullable=True,
        comment="Last processing duration in seconds",
    )

    # Categorization (PostgreSQL: native ARRAY, SQLite: JSON array)
    predicted_categories = Column(
        ARRAY(String).with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        default=list,
        comment="ML-predicted categories",
    )
    category_confidence = Column(
        Float,
        nullable=True,
        comment="Category prediction confidence (0-1)",
    )
    manual_categories = Column(
        ARRAY(String).with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        default=list,
        comment="Manually assigned categories",
    )

    # Quality and validation
    quality_metrics = Column(
        JSONB().with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        comment="Data quality metrics as JSON",
    )
    validation_errors = Column(
        ARRAY(Text).with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        default=list,
        comment="Validation errors (last 20)",
    )

    # Chunks for RAG (stored as JSON array)
    chunks = Column(
        JSONB().with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=True,
        default=list,
        comment="Document chunks as JSON array",
    )
    chunk_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of chunks",
    )

    # Version control
    version = Column(
        Integer,
        nullable=False,
        default=1,
        comment="Document version number",
    )
    checksum = Column(
        String(64),
        nullable=True,
        comment="Content checksum for integrity",
    )

    # Metadata (embedded as JSON)
    # Note: renamed to document_metadata because 'metadata' is reserved in SQLAlchemy
    document_metadata = Column(
        JSONB().with_variant(SQLAlchemyJSON(), "sqlite"),
        nullable=False,
        comment="Document metadata (file info, status, etc.)",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("chunk_count >= 0", name="check_chunk_count_non_negative"),
        CheckConstraint("processing_attempts >= 0", name="check_processing_attempts_non_negative"),
        CheckConstraint(
            "category_confidence IS NULL OR (category_confidence >= 0 AND category_confidence <= 1)",
            name="check_category_confidence_range",
        ),
        {"comment": "Persistent storage for Document entities"},
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<DocumentModel(id={self.id}, "
            f"source_id={self.source_id}, "
            f"status={self.document_metadata.get('status') if isinstance(self.document_metadata, dict) else 'unknown'}, "
            f"chunks={self.chunk_count})>"
        )


# Create indexes separately for better control
from sqlalchemy import Index

# Single column indexes
Index("idx_documents_source_id", DocumentModel.source_id)
Index("idx_documents_created_at", DocumentModel.created_at.desc())
Index("idx_documents_chunk_count", DocumentModel.chunk_count)

# JSON field indexes (PostgreSQL GIN indexes, SQLite will ignore)
# Note: These will only work on PostgreSQL with JSONB
# Index("idx_documents_metadata_gin", DocumentModel.metadata, postgresql_using="gin")
# Index("idx_documents_chunks_gin", DocumentModel.chunks, postgresql_using="gin")

# Composite indexes for common queries
Index(
    "idx_documents_status",
    DocumentModel.document_metadata["status"].astext,
    postgresql_where=(DocumentModel.document_metadata["status"].astext.isnot(None)),
)
Index(
    "idx_documents_ready_for_rag",
    DocumentModel.document_metadata["status"].astext,
    DocumentModel.chunk_count,
    postgresql_where=(
        (DocumentModel.document_metadata["status"].astext == "processed")
        & (DocumentModel.chunk_count > 0)
    ),
)