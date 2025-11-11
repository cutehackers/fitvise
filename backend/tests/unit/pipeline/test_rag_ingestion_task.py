"""Unit tests for RagIngestionTask (Document Ingestion Phase).

Tests cover:
- Task initialization and configuration
- Document discovery (files, databases, web, APIs)
- Document processing pipeline (extract, normalize, enrich, validate)
- Storage and repository operations
- Semantic chunking integration
- Error handling and recovery
- Performance tracking and timing
- Edge cases and boundary conditions
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID, uuid4
import json
import pytest

from app.pipeline.phases.rag_ingestion_task import (
    RagIngestionTask,
    RagIngestionTaskReport,
    UseCaseBundle,
)
from app.pipeline.config import PipelineSpec
from app.pipeline.contracts import (
    DocumentSource,
    MetadataAnnotation,
    QualityReport,
    SourceContent,
    StandardSourceContent,
    StorageObject,
    RunSummary,
)
from app.domain.entities.document import Document
from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentFormat
from app.domain.exceptions import (
    DocumentPathNotFoundError,
    ContentExtractionError,
    NormalizationError,
    MetadataEnrichmentError,
    QualityValidationError,
    StorageWriteError,
    ProcessingError,
    SourceNotReachableError,
    ChunkingError,
)
from app.infrastructure.external_services import ExternalServicesContainer


class TestRagIngestionTaskInitialization:
    """Test RagIngestionTask initialization."""

    def test_task_initialization_with_defaults(self):
        """Test task initialization with default parameters."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )

        assert task.external_services is external_services
        assert task.document_repository is document_repository
        assert task.verbose is False

    def test_task_initialization_with_verbose(self):
        """Test task initialization with verbose logging enabled."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
            verbose=True,
        )

        assert task.verbose is True

    def test_task_initialization_with_data_source_repository(self):
        """Test task initialization with custom data source repository."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()
        data_source_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
            data_source_repository=data_source_repository,
        )

        assert task.data_source_repository is data_source_repository


class TestDocumentDiscovery:
    """Test document discovery methods."""

    @pytest.fixture
    def task_setup(self, tmp_path):
        """Setup task with temporary directory."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )
        return task, tmp_path

    def test_discover_file_documents_with_pdf_files(self, task_setup):
        """Test file discovery with PDF files."""
        task, tmp_path = task_setup

        # Create test PDF files
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_text("PDF content")

        spec = Mock(spec=PipelineSpec)
        spec.documents = Mock()
        spec.documents.path = tmp_path
        spec.documents.include = ["*.pdf"]
        spec.documents.recurse = False

        documents = task._discover_file_documents(spec)

        assert len(documents) == 1
        assert documents[0].uri == str(pdf_file)
        assert documents[0].path == str(pdf_file)
        assert documents[0].content_type == "application/pdf"
        assert documents[0].origin == "file"

    def test_discover_file_documents_with_multiple_patterns(self, task_setup):
        """Test file discovery with multiple file patterns."""
        task, tmp_path = task_setup

        # Create various file types
        (tmp_path / "doc1.pdf").write_text("PDF")
        (tmp_path / "doc2.txt").write_text("Text")
        (tmp_path / "doc3.md").write_text("Markdown")

        spec = Mock(spec=PipelineSpec)
        spec.documents = Mock()
        spec.documents.path = tmp_path
        spec.documents.include = ["*.pdf", "*.txt", "*.md"]
        spec.documents.recurse = False

        documents = task._discover_file_documents(spec)

        assert len(documents) == 3
        assert all(doc.origin == "file" for doc in documents)

    def test_discover_file_documents_with_recursive_search(self, task_setup):
        """Test file discovery with recursive directory search."""
        task, tmp_path = task_setup

        # Create nested directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "doc1.pdf").write_text("PDF1")
        (subdir / "doc2.pdf").write_text("PDF2")

        spec = Mock(spec=PipelineSpec)
        spec.documents = Mock()
        spec.documents.path = tmp_path
        spec.documents.include = ["*.pdf"]
        spec.documents.recurse = True

        documents = task._discover_file_documents(spec)

        assert len(documents) == 2

    def test_discover_file_documents_missing_path_creates_directory(self, task_setup, tmp_path):
        """Test that missing path is created."""
        task, _ = task_setup
        nonexistent_path = tmp_path / "new_dir"

        spec = Mock(spec=PipelineSpec)
        spec.documents = Mock()
        spec.documents.path = nonexistent_path
        spec.documents.include = ["*.pdf"]
        spec.documents.recurse = False

        task._discover_file_documents(spec)

        assert nonexistent_path.exists()

    def test_discover_file_documents_empty_directory(self, task_setup):
        """Test file discovery with empty directory."""
        task, tmp_path = task_setup

        spec = Mock(spec=PipelineSpec)
        spec.documents = Mock()
        spec.documents.path = tmp_path
        spec.documents.include = ["*.pdf"]
        spec.documents.recurse = False

        documents = task._discover_file_documents(spec)

        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_discover_database_documents_successful_connection(self, task_setup):
        """Test successful database discovery."""
        task, _ = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.connect_databases = AsyncMock()
        use_cases.connect_databases.execute.return_value = Mock(
            results=[Mock(
                sample=Mock(
                    rows=[{"id": 1, "name": "test"}],
                    as_dict=lambda: {"id": 1, "name": "test"}
                ),
                error=None,
                collections=None,
            )]
        )

        spec = Mock(spec=PipelineSpec)
        spec.sources = Mock()
        spec.sources.databases = [Mock(
            name="test_db",
            driver="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="testdb",
            db_schema="public",
            params={},
            use_ssl=False,
            connector_type="sql",
            sample_collection=None,
            sample_limit=10,
            fetch_samples=True,
        )]

        documents = await task._discover_database_documents(spec, use_cases)

        assert len(documents) == 1
        assert documents[0].origin == "database"
        assert documents[0].content_type == "application/json"

    @pytest.mark.asyncio
    async def test_discover_database_documents_connection_failure(self, task_setup):
        """Test database discovery with connection failure."""
        task, _ = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.connect_databases = AsyncMock()
        use_cases.connect_databases.execute.side_effect = Exception("Connection failed")

        spec = Mock(spec=PipelineSpec)
        spec.sources = Mock()
        spec.sources.databases = [Mock(
            name="test_db",
            driver="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="testdb",
            db_schema="public",
            params={},
            use_ssl=False,
            connector_type="sql",
            sample_collection=None,
            sample_limit=10,
            fetch_samples=True,
        )]

        errors = []
        documents = await task._discover_database_documents(
            spec, use_cases, errors
        )

        assert len(documents) == 0
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_discover_web_documents_successful_crawl(self, task_setup):
        """Test successful web discovery."""
        task, _ = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.web_scraping = AsyncMock()
        use_cases.web_scraping.execute.return_value = Mock(
            result=Mock(pages=[
                Mock(url="http://example.com/page1", content="<html>Page 1</html>"),
                Mock(url="http://example.com/page2", content="<html>Page 2</html>"),
            ])
        )

        spec = Mock(spec=PipelineSpec)
        spec.sources = Mock()
        spec.sources.web = [Mock(
            start_urls=["http://example.com"],
            allowed_domains=["example.com"],
            max_depth=2,
            max_pages=10,
            include_patterns=[],
            exclude_patterns=[],
            follow_css_selectors=[],
            follow_xpath=[],
            authentication=None,
            headers={},
            cookies={},
            follow_external_links=False,
        )]

        documents = await task._discover_web_documents(spec, use_cases)

        assert len(documents) == 2
        assert all(doc.origin == "web" for doc in documents)
        assert all(doc.content_type == "text/html" for doc in documents)


class TestDocumentProcessing:
    """Test document processing pipeline."""

    @pytest.fixture
    def task_setup(self):
        """Setup task with mocked dependencies."""
        external_services = Mock(spec=ExternalServicesContainer)
        external_services.embedding_model = Mock()
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )
        return task

    @pytest.mark.asyncio
    async def test_extract_content_from_pdf_file(self, task_setup):
        """Test content extraction from PDF file."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.process_pdfs = AsyncMock()
        use_cases.process_pdfs.execute.return_value = Mock(
            documents=[Mock(
                text="PDF text content",
                markdown="# PDF Header\nContent",
                tables=[],
                warnings=[],
            )]
        )

        doc = DocumentSource(
            source_id=None,
            uri="file.pdf",
            path="/path/to/file.pdf",
            content_type="application/pdf",
            origin="file",
        )

        spec = Mock(spec=PipelineSpec)
        content = await task._extract_content(doc, spec, use_cases)

        # When markdown is available, it takes precedence (as per the implementation)
        assert content.text == "# PDF Header\nContent"
        assert content.markdown == "# PDF Header\nContent"
        assert content.content_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_extract_content_from_non_pdf_file(self, task_setup):
        """Test content extraction from non-PDF file using Tika."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.tika = AsyncMock()
        use_cases.tika.execute.return_value = Mock(
            documents=[Mock(
                result=Mock(
                    text="Extracted text content",
                    metadata={"author": "John Doe"},
                    content_type="text/plain",
                    language="en",
                    warnings=[],
                )
            )]
        )

        doc = DocumentSource(
            source_id=None,
            uri="document.docx",
            path="/path/to/document.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            origin="file",
        )

        spec = Mock(spec=PipelineSpec)
        content = await task._extract_content(doc, spec, use_cases)

        assert content.text == "Extracted text content"
        assert content.language == "en"

    @pytest.mark.asyncio
    async def test_extract_content_with_empty_result(self, task_setup):
        """Test content extraction returning empty result."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.process_pdfs = AsyncMock()
        use_cases.process_pdfs.execute.return_value = Mock(documents=[])

        doc = DocumentSource(
            source_id=None,
            uri="empty.pdf",
            path="/path/to/empty.pdf",
            content_type="application/pdf",
            origin="file",
        )

        spec = Mock(spec=PipelineSpec)
        content = await task._extract_content(doc, spec, use_cases)

        assert content.text == ""
        assert content.markdown == ""

    @pytest.mark.asyncio
    async def test_normalize_content(self, task_setup):
        """Test text normalization."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.normalize_text = AsyncMock()
        use_cases.normalize_text.execute.return_value = Mock(
            results=[Mock(
                normalized="normalized text",
                tokens=["normalized", "text"],
                lemmas=["normalize", "text"],
                entities=["ENTITY"],
            )]
        )

        text = "Some text to normalize"
        result = await task._normalize_content(text, use_cases)

        assert result.normalized == "normalized text"
        assert len(result.tokens) == 2
        assert len(result.lemmas) == 2

    @pytest.mark.asyncio
    async def test_enrich_content_with_metadata(self, task_setup):
        """Test metadata enrichment."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.extract_metadata = AsyncMock()
        use_cases.extract_metadata.execute.return_value = Mock(
            results=[Mock(
                keywords=["keyword1", "keyword2"],
                entities=["ENTITY1", "ENTITY2"],
                dates=["2024-01-01"],
                authors=["John Doe"],
                language="en",
            )]
        )

        text = "Document content with metadata"
        result = await task._enrich_content(text, use_cases)

        assert len(result.keywords) == 2
        assert len(result.entities) == 2
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_validate_content_quality(self, task_setup):
        """Test content quality validation."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.validate_quality = AsyncMock()
        use_cases.validate_quality.execute.return_value = Mock(
            reports=[Mock(
                overall_score=0.85,
                quality_level="high",
                metrics={"readability": 0.85},
                validations=[],
                ge_report=None,
            )]
        )

        text = "High quality content"
        result = await task._validate_content(text, use_cases)

        assert result.overall_score == 0.85
        assert result.quality_level == "high"

    def test_content_hash_generation(self, task_setup):
        """Test content hash generation."""
        task = task_setup

        content = b"test content"
        hash_result = task._content_hash(content)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 produces 64 hex characters


class TestDocumentEntity:
    """Test document entity construction."""

    @pytest.fixture
    def task_setup(self):
        """Setup task."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )
        return task

    def test_build_document_entity(self, task_setup):
        """Test building a document entity."""
        task = task_setup

        doc = DocumentSource(
            source_id=None,
            uri="test.pdf",
            path="/path/to/test.pdf",
            content_type="application/pdf",
            origin="file",
        )

        normalized = StandardSourceContent(
            normalized="Normalized content",
            tokens=["normalized", "content"],
            lemmas=["normalize", "content"],
            entities=[],
            warnings=[],
        )

        enriched = MetadataAnnotation(
            keywords=["keyword1"],
            entities=["ENTITY"],
            dates=[],
            authors=["John Doe"],
            language="en",
            warnings=[],
        )

        quality = QualityReport(
            overall_score=0.85,
            quality_level="high",
            metrics={},
            validations=[],
            ge_report=None,
        )

        content = SourceContent(
            text="test content",
            markdown=None,
            tables=[],
            metadata={},
            content_type="application/pdf",
            language="en",
        )

        run_id = "test-run-id"
        object_key = "2024/01/01/abc123.md"
        payload_size = 1024

        entity = task._build_document_entity(
            doc=doc,
            normalized=normalized,
            enriched=enriched,
            quality=quality,
            content=content,
            object_key=object_key,
            payload_size=payload_size,
            run_id=run_id,
        )

        assert isinstance(entity, Document)
        assert entity.metadata.file_name == "test.pdf"
        assert entity.metadata.format == DocumentFormat.PDF
        assert entity.metadata.language == "en"

    def test_infer_document_format_from_content_type(self, task_setup):
        """Test document format inference from content type."""
        task = task_setup

        # Test PDF
        fmt = task._infer_document_format("application/pdf", None)
        assert fmt == DocumentFormat.PDF

        # Test Markdown
        fmt = task._infer_document_format("text/markdown", None)
        assert fmt == DocumentFormat.MD

        # Test HTML
        fmt = task._infer_document_format("text/html", None)
        assert fmt == DocumentFormat.HTML

    def test_infer_document_format_from_path(self, task_setup):
        """Test document format inference from file path."""
        task = task_setup

        # Test PDF
        fmt = task._infer_document_format(None, "/path/to/file.pdf")
        assert fmt == DocumentFormat.PDF

        # Test Markdown
        fmt = task._infer_document_format(None, "/path/to/file.md")
        assert fmt == DocumentFormat.MD

        # Test default to TXT
        fmt = task._infer_document_format(None, "/path/to/file.unknown")
        assert fmt == DocumentFormat.TXT


class TestRagIngestionTaskReport:
    """Test RagIngestionTaskReport."""

    def test_report_creation(self):
        """Test report creation."""
        report = RagIngestionTaskReport(
            success=True,
            execution_time_seconds=10.5,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            total_errors=0,
            total_warnings=2,
        )

        assert report.success is True
        assert report.execution_time_seconds == 10.5
        assert report.total_errors == 0
        assert report.total_warnings == 2

    def test_report_as_dict(self):
        """Test report serialization to dictionary."""
        start = datetime.now(timezone.utc)
        end = datetime.now(timezone.utc)

        report = RagIngestionTaskReport(
            success=True,
            execution_time_seconds=5.0,
            start_time=start,
            end_time=end,
            total_errors=1,
            total_warnings=2,
            timing_breakdown={"extraction": 1.0, "processing": 3.0},
        )

        report_dict = report.as_dict()

        assert report_dict["success"] is True
        assert report_dict["execution_time_seconds"] == 5.0
        assert report_dict["total_errors"] == 1
        assert "pipeline_timing" in report_dict

    def test_report_as_json(self):
        """Test report serialization to JSON."""
        report = RagIngestionTaskReport(
            success=True,
            execution_time_seconds=5.0,
            total_errors=0,
        )

        json_str = report.as_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["success"] is True

    def test_report_properties(self):
        """Test report convenience properties."""
        stored_objects = [Mock(spec=StorageObject)]
        summary = RunSummary(
            processed=1,
            skipped=0,
            failed=0,
            stored=stored_objects,
            errors=[],
            counters={"discovered": 1, "chunking": {"total_chunks": 5}},
        )

        report = RagIngestionTaskReport(
            success=True,
            phase_result=summary,
        )

        assert report.processed == 1
        assert report.skipped == 0
        assert report.failed == 0
        assert report.discovered == 1
        assert report.chunking_summary["total_chunks"] == 5


class TestErrorHandling:
    """Test error handling in RagIngestionTask."""

    @pytest.fixture
    def task_setup(self):
        """Setup task."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )
        return task

    @pytest.mark.asyncio
    async def test_extraction_error_handling(self, task_setup):
        """Test handling of content extraction errors."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.process_pdfs = AsyncMock()
        use_cases.process_pdfs.execute.side_effect = ContentExtractionError(
            "Failed to extract",
            stage="extract",
            source="test.pdf",
        )

        doc = DocumentSource(
            source_id=None,
            uri="test.pdf",
            path="/path/to/test.pdf",
            content_type="application/pdf",
            origin="file",
        )

        spec = Mock(spec=PipelineSpec)

        with pytest.raises(ContentExtractionError):
            await task._extract_content(doc, spec, use_cases)

    @pytest.mark.asyncio
    async def test_normalization_error_handling(self, task_setup):
        """Test handling of normalization errors."""
        task = task_setup

        use_cases = Mock(spec=UseCaseBundle)
        use_cases.normalize_text = AsyncMock()
        use_cases.normalize_text.execute.side_effect = NormalizationError(
            "Normalization failed",
            stage="normalize",
        )

        with pytest.raises(NormalizationError):
            await task._normalize_content("text", use_cases)

    def test_guess_content_type(self, task_setup):
        """Test content type guessing from file extension."""
        task = task_setup

        assert task._guess_content_type(Path("file.pdf")) == "application/pdf"
        assert task._guess_content_type(Path("file.txt")) == "text/plain"
        assert task._guess_content_type(Path("file.md")) == "text/markdown"
        assert task._guess_content_type(Path("file.html")) == "text/html"
        assert task._guess_content_type(Path("file.csv")) == "text/csv"


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @pytest.fixture
    def task_setup(self):
        """Setup task with mocked use cases."""
        external_services = Mock(spec=ExternalServicesContainer)
        external_services.embedding_model = Mock()
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )
        return task

    @pytest.mark.asyncio
    async def test_full_document_processing_pipeline(self, task_setup):
        """Test complete document processing pipeline."""
        task = task_setup

        # Mock all dependencies
        use_cases = Mock(spec=UseCaseBundle)
        use_cases.process_pdfs = AsyncMock()
        use_cases.normalize_text = AsyncMock()
        use_cases.extract_metadata = AsyncMock()
        use_cases.validate_quality = AsyncMock()
        use_cases.document_repository = AsyncMock()
        use_cases.chunk_documents = AsyncMock()

        # Setup return values
        use_cases.process_pdfs.execute.return_value = Mock(
            documents=[Mock(
                text="Document content",
                markdown="# Content",
                tables=[],
                warnings=[],
            )]
        )

        use_cases.normalize_text.execute.return_value = Mock(
            results=[Mock(
                normalized="normalized content",
                tokens=["normalized", "content"],
                lemmas=["normalize", "content"],
                entities=[],
                warnings=[],
            )]
        )

        use_cases.extract_metadata.execute.return_value = Mock(
            results=[Mock(
                keywords=["keyword1"],
                entities=["ENTITY"],
                dates=[],
                authors=["John Doe"],
                language="en",
                warnings=[],
            )]
        )

        use_cases.validate_quality.execute.return_value = Mock(
            reports=[Mock(
                overall_score=0.85,
                quality_level="high",
                metrics={},
                validations=[],
                ge_report=None,
            )]
        )

        # Create test document
        doc = DocumentSource(
            source_id=None,
            uri="test.pdf",
            path="/path/to/test.pdf",
            content_type="application/pdf",
            origin="file",
        )

        # Test extraction through validation
        spec = Mock(spec=PipelineSpec)

        content = await task._extract_content(doc, spec, use_cases)
        # Markdown takes precedence when available
        assert content.text == "# Content"

        normalized = await task._normalize_content(content.text, use_cases)
        assert normalized.normalized == "normalized content"

        enriched = await task._enrich_content(normalized.normalized, use_cases)
        assert len(enriched.keywords) > 0

        quality = await task._validate_content(normalized.normalized, use_cases)
        assert quality.overall_score == 0.85


class TestPerformanceTracking:
    """Test performance tracking functionality."""

    @pytest.fixture
    def task_setup(self):
        """Setup task."""
        external_services = Mock(spec=ExternalServicesContainer)
        document_repository = Mock()

        task = RagIngestionTask(
            external_services=external_services,
            document_repository=document_repository,
        )
        return task

    def test_performance_tracker_initialization(self, task_setup):
        """Test that PerformanceTracker is initialized."""
        task = task_setup

        assert task.tracker is not None
        assert hasattr(task.tracker, 'with_tracker')
        assert hasattr(task.tracker, 'get_stats')
        assert hasattr(task.tracker, 'get_all_stats')


class TestUseCaseBundle:
    """Test UseCaseBundle dataclass."""

    def test_use_case_bundle_creation(self):
        """Test UseCaseBundle creation."""
        repository = Mock()
        document_repository = Mock()
        storage = Mock()
        process_pdfs = Mock()
        normalize_text = Mock()
        extract_metadata = Mock()
        validate_quality = Mock()
        tika = Mock()
        connect_databases = Mock()
        web_scraping = Mock()
        audit_sources = Mock()
        categorize_sources = Mock()
        document_apis = Mock()
        chunk_documents = Mock()

        bundle = UseCaseBundle(
            repository=repository,
            document_repository=document_repository,
            storage=storage,
            process_pdfs=process_pdfs,
            normalize_text=normalize_text,
            extract_metadata=extract_metadata,
            validate_quality=validate_quality,
            tika=tika,
            connect_databases=connect_databases,
            web_scraping=web_scraping,
            audit_sources=audit_sources,
            categorize_sources=categorize_sources,
            document_apis=document_apis,
            chunk_documents=chunk_documents,
        )

        assert bundle.repository is repository
        assert bundle.document_repository is document_repository
        assert bundle.storage is storage
        assert bundle.process_pdfs is process_pdfs
