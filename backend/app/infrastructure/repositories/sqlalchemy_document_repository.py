"""SQLAlchemy implementation of DocumentRepository supporting PostgreSQL and SQLite."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import Float, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.document import Document
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.document_metadata import (
    DocumentFormat,
    DocumentMetadata,
    DocumentStatus,
)
from app.domain.value_objects.quality_metrics import (
    ContentQualityMetrics,
    DataQualityMetrics,
    QualityLevel,
    ValidationResult,
    ValidationRule,
)
from app.infrastructure.database.models import DocumentModel


class SQLAlchemyDocumentRepository(DocumentRepository):
    """SQLAlchemy-based document repository supporting multiple databases.

    This repository handles persistence and retrieval of Document entities
    with full support for:
    - PostgreSQL (production): Native JSONB, arrays, UUIDs, full-text search
    - SQLite (development): JSON emulation via SQLAlchemy type variants

    The repository automatically adapts to the configured database driver:
    - sqlite+aiosqlite:/// for development
    - postgresql+asyncpg:// for production

    Examples:
        >>> # Works with both PostgreSQL and SQLite
        >>> async with async_session_maker() as session:
        ...     repo = SQLAlchemyDocumentRepository(session)
        ...     document = await repo.find_by_id(doc_id)
        ...     document.start_processing()
        ...     await repo.save(document)
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session for database operations
        """
        self._session = session

    async def save(self, document: Document) -> Document:
        """Save or update a document.

        Args:
            document: Document entity to persist

        Returns:
            Saved document entity

        Raises:
            SQLAlchemyError: If database operation fails
        """
        # Check if document exists
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.id == document.id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing document
            self._update_model_from_entity(existing, document)
        else:
            # Create new document
            model = self._to_model(document)
            self._session.add(model)

        await self._session.flush()
        await self._session.commit()
        return document

    async def find_by_id(self, document_id: UUID) -> Optional[Document]:
        """Find document by ID.

        Args:
            document_id: Unique document identifier

        Returns:
            Document entity if found, None otherwise
        """
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_source_id(self, source_id: UUID) -> List[Document]:
        """Find all documents from a source.

        Args:
            source_id: Source identifier

        Returns:
            List of documents from the source
        """
        result = await self._session.execute(
            select(DocumentModel)
            .where(DocumentModel.source_id == source_id)
            .order_by(DocumentModel.created_at.desc())
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_by_file_path(self, file_path: str) -> Optional[Document]:
        """Find document by file path.

        Args:
            file_path: File path to search for

        Returns:
            Document entity if found, None otherwise
        """
        result = await self._session.execute(
            select(DocumentModel).where(
                DocumentModel.document_metadata["file_path"].astext == file_path
            )
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        """Find documents by processing status.

        Args:
            status: Document status to filter by

        Returns:
            List of documents with the specified status
        """
        result = await self._session.execute(
            select(DocumentModel)
            .where(DocumentModel.document_metadata["status"].astext == status.value)
            .order_by(DocumentModel.created_at.desc())
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_by_format(self, format: DocumentFormat) -> List[Document]:
        """Find documents by format.

        Args:
            format: Document format to filter by

        Returns:
            List of documents with the specified format
        """
        result = await self._session.execute(
            select(DocumentModel)
            .where(DocumentModel.document_metadata["format"].astext == format.value)
            .order_by(DocumentModel.created_at.desc())
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_by_categories(self, categories: List[str]) -> List[Document]:
        """Find documents by categories.

        Args:
            categories: List of categories to match (any match)

        Returns:
            List of documents matching any of the categories
        """
        # This works differently for PostgreSQL (array overlap) vs SQLite (JSON contains)
        result = await self._session.execute(
            select(DocumentModel).where(
                DocumentModel.predicted_categories.overlap(categories)
                | DocumentModel.manual_categories.overlap(categories)
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_processed_documents(self) -> List[Document]:
        """Find all successfully processed documents.

        Returns:
            List of processed documents
        """
        return await self.find_by_status(DocumentStatus.PROCESSED)

    async def find_failed_documents(self) -> List[Document]:
        """Find all failed documents.

        Returns:
            List of failed documents
        """
        return await self.find_by_status(DocumentStatus.FAILED)

    async def find_ready_for_rag(self) -> List[Document]:
        """Find documents ready for RAG (processed with chunks and embeddings).

        Returns:
            List of RAG-ready documents
        """
        result = await self._session.execute(
            select(DocumentModel)
            .where(
                DocumentModel.document_metadata["status"].astext == DocumentStatus.PROCESSED.value,
                DocumentModel.chunk_count > 0,
                DocumentModel.embeddings.isnot(None),
            )
            .order_by(DocumentModel.created_at.desc())
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_needing_reprocessing(self) -> List[Document]:
        """Find documents needing reprocessing.

        Returns:
            List of documents needing reprocessing
        """
        result = await self._session.execute(
            select(DocumentModel).where(
                (
                    DocumentModel.document_metadata["status"].astext.in_(
                        [DocumentStatus.PENDING.value, DocumentStatus.FAILED.value]
                    )
                )
                | (
                    (DocumentModel.document_metadata["status"].astext == DocumentStatus.PROCESSED.value)
                    & (DocumentModel.chunk_count == 0)
                )
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def find_by_quality_score_range(
        self, min_score: float, max_score: float
    ) -> List[Document]:
        """Find documents by quality score range.

        Args:
            min_score: Minimum quality score (inclusive)
            max_score: Maximum quality score (inclusive)

        Returns:
            List of documents within quality score range
        """
        result = await self._session.execute(
            select(DocumentModel).where(
                DocumentModel.quality_metrics["overall_quality_score"]
                .astext.cast(Float)
                .between(min_score, max_score)
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def delete(self, document_id: UUID) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Document ID to delete

        Returns:
            True if document was deleted, False if not found
        """
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        model = result.scalar_one_or_none()

        if model:
            await self._session.delete(model)
            await self._session.flush()
            await self._session.commit()
            return True
        return False

    async def delete_by_source_id(self, source_id: UUID) -> int:
        """Delete all documents from a source.

        Args:
            source_id: Source ID to delete documents from

        Returns:
            Number of documents deleted
        """
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.source_id == source_id)
        )
        models = result.scalars().all()

        count = len(models)
        for model in models:
            await self._session.delete(model)

        await self._session.flush()
        await self._session.commit()
        return count

    async def count_all(self) -> int:
        """Count all documents.

        Returns:
            Total number of documents
        """
        result = await self._session.execute(select(func.count(DocumentModel.id)))
        return result.scalar_one()

    async def count_by_source_id(self, source_id: UUID) -> int:
        """Count documents from a source.

        Args:
            source_id: Source ID to count

        Returns:
            Number of documents from source
        """
        result = await self._session.execute(
            select(func.count(DocumentModel.id)).where(
                DocumentModel.source_id == source_id
            )
        )
        return result.scalar_one()

    async def count_by_status(self, status: DocumentStatus) -> int:
        """Count documents by status.

        Args:
            status: Document status to count

        Returns:
            Number of documents with status
        """
        result = await self._session.execute(
            select(func.count(DocumentModel.id)).where(
                DocumentModel.document_metadata["status"].astext == status.value
            )
        )
        return result.scalar_one()

    async def count_by_format(self, format: DocumentFormat) -> int:
        """Count documents by format.

        Args:
            format: Document format to count

        Returns:
            Number of documents with format
        """
        result = await self._session.execute(
            select(func.count(DocumentModel.id)).where(
                DocumentModel.document_metadata["format"].astext == format.value
            )
        )
        return result.scalar_one()

    async def get_processing_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        total = await self.count_all()
        processed = await self.count_by_status(DocumentStatus.PROCESSED)
        failed = await self.count_by_status(DocumentStatus.FAILED)
        pending = await self.count_by_status(DocumentStatus.PENDING)

        return {
            "total": total,
            "processed": processed,
            "failed": failed,
            "pending": pending,
        }

    # Conversion methods

    def _to_model(self, entity: Document) -> DocumentModel:
        """Convert domain entity to database model.

        Args:
            entity: Document entity

        Returns:
            SQLAlchemy model instance
        """
        return DocumentModel(
            id=entity.id,
            source_id=entity.source_id,
            content=entity.content,
            extracted_text=entity.extracted_text,
            structured_content=entity.structured_content,
            embeddings=entity.embeddings,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            last_processed_at=entity.last_processed_at,
            processing_attempts=entity.processing_attempts,
            processing_duration=entity.processing_duration,
            predicted_categories=entity.predicted_categories,
            category_confidence=entity.category_confidence,
            manual_categories=entity.manual_categories,
            quality_metrics=self._serialize_quality_metrics(entity.quality_metrics),
            validation_errors=entity.validation_errors,
            chunks=entity.chunks,
            chunk_count=entity.chunk_count,
            version=entity.version,
            checksum=entity.checksum,
            document_metadata=self._serialize_metadata(entity.metadata),
        )

    def _update_model_from_entity(self, model: DocumentModel, entity: Document) -> None:
        """Update existing model from entity.

        Args:
            model: Existing SQLAlchemy model
            entity: Document entity with updated data
        """
        model.source_id = entity.source_id
        model.content = entity.content
        model.extracted_text = entity.extracted_text
        model.structured_content = entity.structured_content
        model.embeddings = entity.embeddings
        model.updated_at = entity.updated_at
        model.last_processed_at = entity.last_processed_at
        model.processing_attempts = entity.processing_attempts
        model.processing_duration = entity.processing_duration
        model.predicted_categories = entity.predicted_categories
        model.category_confidence = entity.category_confidence
        model.manual_categories = entity.manual_categories
        model.quality_metrics = self._serialize_quality_metrics(entity.quality_metrics)
        model.validation_errors = entity.validation_errors
        model.chunks = entity.chunks
        model.chunk_count = entity.chunk_count
        model.version = entity.version
        model.checksum = entity.checksum
        model.document_metadata = self._serialize_metadata(entity.metadata)

    def _to_entity(self, model: DocumentModel) -> Document:
        """Convert database model to domain entity.

        Args:
            model: SQLAlchemy model instance

        Returns:
            Document entity
        """
        metadata = self._deserialize_metadata(model.document_metadata)
        quality_metrics = self._deserialize_quality_metrics(model.quality_metrics)

        # Create entity with all fields
        document = Document(
            source_id=model.source_id,
            metadata=metadata,
            content=model.content,
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

        # Set internal fields directly (bypassing business methods)
        document._extracted_text = model.extracted_text
        document._structured_content = model.structured_content
        document._embeddings = model.embeddings
        document._last_processed_at = model.last_processed_at
        document._processing_attempts = model.processing_attempts
        document._processing_duration = model.processing_duration
        document._predicted_categories = model.predicted_categories or []
        document._category_confidence = model.category_confidence
        document._manual_categories = model.manual_categories or []
        document._quality_metrics = quality_metrics
        document._validation_errors = model.validation_errors or []
        document._chunks = model.chunks or []
        document._chunk_count = model.chunk_count
        document._version = model.version
        document._checksum = model.checksum

        return document

    def _serialize_metadata(self, metadata: DocumentMetadata) -> dict:
        """Serialize DocumentMetadata to JSON-compatible dict.

        Args:
            metadata: DocumentMetadata value object

        Returns:
            JSON-serializable dictionary
        """
        return {
            "file_name": metadata.file_name,
            "file_path": metadata.file_path,
            "file_size": metadata.file_size,
            "format": metadata.format.value,
            "title": metadata.title,
            "author": metadata.author,
            "language": metadata.language,
            "keywords": metadata.keywords,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
            "modified_at": metadata.modified_at.isoformat()
            if metadata.modified_at
            else None,
            "accessed_at": metadata.accessed_at.isoformat()
            if metadata.accessed_at
            else None,
            "status": metadata.status.value,
            "processing_time": metadata.processing_time,
            "error_message": metadata.error_message,
            "word_count": metadata.word_count,
            "page_count": metadata.page_count,
            "encoding": metadata.encoding,
            "custom_fields": metadata.custom_fields,
        }

    def _deserialize_metadata(self, data: dict) -> DocumentMetadata:
        """Deserialize JSON dict to DocumentMetadata.

        Args:
            data: JSON dictionary from database

        Returns:
            DocumentMetadata value object
        """
        return DocumentMetadata(
            file_name=data["file_name"],
            file_path=data["file_path"],
            file_size=data["file_size"],
            format=DocumentFormat(data["format"]),
            title=data.get("title"),
            author=data.get("author"),
            language=data.get("language"),
            keywords=data.get("keywords", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            modified_at=datetime.fromisoformat(data["modified_at"])
            if data.get("modified_at")
            else None,
            accessed_at=datetime.fromisoformat(data["accessed_at"])
            if data.get("accessed_at")
            else None,
            status=DocumentStatus(data["status"]),
            processing_time=data.get("processing_time"),
            error_message=data.get("error_message"),
            word_count=data.get("word_count"),
            page_count=data.get("page_count"),
            encoding=data.get("encoding"),
            custom_fields=data.get("custom_fields", {}),
        )

    def _serialize_quality_metrics(
        self, metrics: Optional[DataQualityMetrics]
    ) -> Optional[dict]:
        """Serialize DataQualityMetrics to JSON-compatible dict.

        Args:
            metrics: DataQualityMetrics value object or None

        Returns:
            JSON-serializable dictionary or None
        """
        if metrics is None:
            return None

        return {
            "measured_at": metrics.measured_at.isoformat(),
            "validation_results": [
                {
                    "rule": r.rule.value,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "details": r.details,
                }
                for r in metrics.validation_results
            ],
            "content_metrics": {
                "total_characters": metrics.content_metrics.total_characters,
                "total_words": metrics.content_metrics.total_words,
                "total_sentences": metrics.content_metrics.total_sentences,
                "total_paragraphs": metrics.content_metrics.total_paragraphs,
                "detected_language": metrics.content_metrics.detected_language,
                "language_confidence": metrics.content_metrics.language_confidence,
                "readability_score": metrics.content_metrics.readability_score,
                "has_title": metrics.content_metrics.has_title,
                "has_headings": metrics.content_metrics.has_headings,
                "has_tables": metrics.content_metrics.has_tables,
                "has_images": metrics.content_metrics.has_images,
                "has_links": metrics.content_metrics.has_links,
                "empty_sections": metrics.content_metrics.empty_sections,
                "broken_links": metrics.content_metrics.broken_links,
                "missing_metadata_fields": metrics.content_metrics.missing_metadata_fields,
                "encoding_issues": metrics.content_metrics.encoding_issues,
                "format_errors": metrics.content_metrics.format_errors,
            },
            "processing_time_seconds": metrics.processing_time_seconds,
            "memory_usage_mb": metrics.memory_usage_mb,
            "errors_encountered": metrics.errors_encountered,
            "warnings_encountered": metrics.warnings_encountered,
            "source_availability": metrics.source_availability,
            "source_response_time_ms": metrics.source_response_time_ms,
            "category_confidence": metrics.category_confidence,
            "predicted_categories": metrics.predicted_categories,
            "overall_quality_score": metrics.overall_quality_score,
            "quality_level": metrics.quality_level.value,
        }

    def _deserialize_quality_metrics(
        self, data: Optional[dict]
    ) -> Optional[DataQualityMetrics]:
        """Deserialize JSON dict to DataQualityMetrics.

        Args:
            data: JSON dictionary from database or None

        Returns:
            DataQualityMetrics value object or None
        """
        if data is None:
            return None

        content_metrics_data = data.get("content_metrics", {})
        content_metrics = ContentQualityMetrics(
            total_characters=content_metrics_data.get("total_characters", 0),
            total_words=content_metrics_data.get("total_words", 0),
            total_sentences=content_metrics_data.get("total_sentences", 0),
            total_paragraphs=content_metrics_data.get("total_paragraphs", 0),
            detected_language=content_metrics_data.get("detected_language"),
            language_confidence=content_metrics_data.get("language_confidence"),
            readability_score=content_metrics_data.get("readability_score"),
            has_title=content_metrics_data.get("has_title", False),
            has_headings=content_metrics_data.get("has_headings", False),
            has_tables=content_metrics_data.get("has_tables", False),
            has_images=content_metrics_data.get("has_images", False),
            has_links=content_metrics_data.get("has_links", False),
            empty_sections=content_metrics_data.get("empty_sections", 0),
            broken_links=content_metrics_data.get("broken_links", 0),
            missing_metadata_fields=content_metrics_data.get(
                "missing_metadata_fields", 0
            ),
            encoding_issues=content_metrics_data.get("encoding_issues", 0),
            format_errors=content_metrics_data.get("format_errors", 0),
        )

        validation_results = [
            ValidationResult(
                rule=ValidationRule(r["rule"]),
                passed=r["passed"],
                score=r["score"],
                message=r["message"],
                details=r.get("details"),
            )
            for r in data.get("validation_results", [])
        ]

        return DataQualityMetrics(
            measured_at=datetime.fromisoformat(data["measured_at"]),
            validation_results=validation_results,
            content_metrics=content_metrics,
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            memory_usage_mb=data.get("memory_usage_mb"),
            errors_encountered=data.get("errors_encountered", []),
            warnings_encountered=data.get("warnings_encountered", []),
            source_availability=data.get("source_availability", True),
            source_response_time_ms=data.get("source_response_time_ms"),
            category_confidence=data.get("category_confidence"),
            predicted_categories=data.get("predicted_categories", []),
        )
