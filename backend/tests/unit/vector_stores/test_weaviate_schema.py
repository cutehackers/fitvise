"""Unit tests for Weaviate schema (Task 2.3.2)."""

import pytest
from datetime import datetime
from uuid import uuid4, UUID

from app.infrastructure.external_services.vector_stores.weaviate_schema import (
    create_chunk_class_schema,
    validate_chunk_metadata,
    create_chunk_metadata_template,
    get_filterable_fields,
    get_searchable_fields,
    SCHEMA_VERSION,
)


class TestChunkClassSchema:
    """Tests for Chunk class schema creation."""

    def test_create_default_schema(self):
        """Test creating schema with default parameters."""
        schema = create_chunk_class_schema()

        assert schema["class"] == "Chunk"
        assert schema["vectorizer"] == "none"
        assert schema["vectorIndexConfig"]["distance"] == "cosine"
        assert len(schema["properties"]) == 21  # Enhanced schema

    def test_create_schema_with_custom_dimension(self):
        """Test creating schema with custom vector dimension."""
        schema = create_chunk_class_schema(dimension=768)

        # Dimension is not in schema, but tracked in config
        assert schema["class"] == "Chunk"
        assert schema["vectorizer"] == "none"

    def test_create_schema_with_custom_distance(self):
        """Test creating schema with custom distance metric."""
        schema = create_chunk_class_schema(distance_metric="dot")

        assert schema["vectorIndexConfig"]["distance"] == "dot"

    def test_schema_has_required_core_fields(self):
        """Test schema includes required core fields."""
        schema = create_chunk_class_schema()
        property_names = [prop["name"] for prop in schema["properties"]]

        # Core fields
        assert "text" in property_names
        assert "chunk_id" in property_names
        assert "document_id" in property_names
        assert "sequence" in property_names

    def test_schema_has_model_metadata_fields(self):
        """Test schema includes model metadata fields."""
        schema = create_chunk_class_schema()
        property_names = [prop["name"] for prop in schema["properties"]]

        assert "model_name" in property_names
        assert "model_version" in property_names

    def test_schema_has_document_metadata_fields(self):
        """Test schema includes document metadata fields."""
        schema = create_chunk_class_schema()
        property_names = [prop["name"] for prop in schema["properties"]]

        # Enhanced metadata for RAG filtering
        assert "doc_type" in property_names
        assert "source_type" in property_names
        assert "file_name" in property_names
        assert "section" in property_names
        assert "category" in property_names
        assert "department" in property_names
        assert "author" in property_names
        assert "language" in property_names

    def test_schema_has_hierarchical_fields(self):
        """Test schema includes hierarchical context fields."""
        schema = create_chunk_class_schema()
        property_names = [prop["name"] for prop in schema["properties"]]

        assert "parent_chunk_id" in property_names
        assert "tags" in property_names

    def test_schema_has_quality_metrics(self):
        """Test schema includes quality metric fields."""
        schema = create_chunk_class_schema()
        property_names = [prop["name"] for prop in schema["properties"]]

        assert "token_count" in property_names
        assert "confidence_score" in property_names
        assert "quality_score" in property_names

    def test_schema_has_timestamps(self):
        """Test schema includes timestamp fields."""
        schema = create_chunk_class_schema()
        property_names = [prop["name"] for prop in schema["properties"]]

        assert "created_at" in property_names
        assert "updated_at" in property_names

    def test_text_field_is_searchable(self):
        """Test text field is configured for search."""
        schema = create_chunk_class_schema()
        text_field = next(p for p in schema["properties"] if p["name"] == "text")

        assert text_field["indexSearchable"] is True
        assert text_field["indexFilterable"] is True

    def test_uuid_fields_are_filterable(self):
        """Test UUID fields are configured for filtering."""
        schema = create_chunk_class_schema()
        uuid_fields = ["chunk_id", "document_id", "parent_chunk_id"]

        for field_name in uuid_fields:
            field = next(p for p in schema["properties"] if p["name"] == field_name)
            assert field["dataType"] == ["uuid"]
            assert field["indexFilterable"] is True

    def test_numeric_fields_have_range_filters(self):
        """Test numeric fields support range filtering."""
        schema = create_chunk_class_schema()
        range_fields = ["sequence", "token_count", "confidence_score", "quality_score"]

        for field_name in range_fields:
            field = next(p for p in schema["properties"] if p["name"] == field_name)
            assert field["indexRangeFilters"] is True

    def test_hnsw_configuration(self):
        """Test HNSW index is properly configured."""
        schema = create_chunk_class_schema()
        hnsw_config = schema["vectorIndexConfig"]

        assert hnsw_config["ef"] == -1  # Dynamic
        assert hnsw_config["efConstruction"] == 128
        assert hnsw_config["maxConnections"] == 64
        assert hnsw_config["vectorCacheMaxObjects"] == 1000000


class TestMetadataValidation:
    """Tests for metadata validation."""

    def test_validate_valid_metadata(self):
        """Test validation passes for valid metadata."""
        metadata = {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "text": "Sample chunk text",
            "sequence": 0,
            "model_name": "Alibaba-NLP/gte-multilingual-base",
            "token_count": 10,
        }

        # Should not raise
        validate_chunk_metadata(metadata)

    def test_validate_missing_required_field(self):
        """Test validation fails for missing required fields."""
        metadata = {
            "chunk_id": uuid4(),
            # Missing document_id and text
        }

        with pytest.raises(ValueError, match="Missing required field"):
            validate_chunk_metadata(metadata)

    def test_validate_invalid_uuid(self):
        """Test validation fails for invalid UUID."""
        metadata = {
            "chunk_id": "not-a-uuid",
            "document_id": uuid4(),
            "text": "Sample text",
        }

        with pytest.raises(ValueError, match="Invalid UUID"):
            validate_chunk_metadata(metadata)

    def test_validate_negative_sequence(self):
        """Test validation fails for negative sequence."""
        metadata = {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "text": "Sample text",
            "sequence": -1,
        }

        with pytest.raises(ValueError, match="sequence must be non-negative"):
            validate_chunk_metadata(metadata)

    def test_validate_invalid_confidence_score(self):
        """Test validation fails for invalid confidence score."""
        metadata = {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "text": "Sample text",
            "confidence_score": 1.5,  # Out of range
        }

        with pytest.raises(ValueError, match="confidence_score must be float in"):
            validate_chunk_metadata(metadata)

    def test_validate_invalid_quality_score(self):
        """Test validation fails for invalid quality score."""
        metadata = {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "text": "Sample text",
            "quality_score": -0.1,  # Out of range
        }

        with pytest.raises(ValueError, match="quality_score must be float in"):
            validate_chunk_metadata(metadata)

    def test_validate_non_string_field(self):
        """Test validation fails for non-string in string field."""
        metadata = {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "text": "Sample text",
            "doc_type": 123,  # Should be string
        }

        with pytest.raises(ValueError, match="doc_type must be string"):
            validate_chunk_metadata(metadata)

    def test_validate_invalid_tags(self):
        """Test validation fails for invalid tags."""
        metadata = {
            "chunk_id": uuid4(),
            "document_id": uuid4(),
            "text": "Sample text",
            "tags": ["valid", 123, "another"],  # Mixed types
        }

        with pytest.raises(ValueError, match="All tags must be strings"):
            validate_chunk_metadata(metadata)


class TestMetadataTemplate:
    """Tests for metadata template creation."""

    def test_create_basic_template(self):
        """Test creating basic metadata template."""
        chunk_id = uuid4()
        doc_id = uuid4()
        text = "Sample chunk text"

        metadata = create_chunk_metadata_template(
            chunk_id=chunk_id, document_id=doc_id, text=text
        )

        assert metadata["chunk_id"] == chunk_id
        assert metadata["document_id"] == doc_id
        assert metadata["text"] == text
        assert metadata["sequence"] == 0
        assert metadata["model_name"] == "Alibaba-NLP/gte-multilingual-base"
        assert metadata["source_type"] == "chunk"

    def test_create_template_with_optional_fields(self):
        """Test creating template with optional metadata."""
        chunk_id = uuid4()
        doc_id = uuid4()

        metadata = create_chunk_metadata_template(
            chunk_id=chunk_id,
            document_id=doc_id,
            text="Sample text",
            doc_type="pdf",
            department="engineering",
            category="policy",
            tags=["important", "urgent"],
        )

        assert metadata["doc_type"] == "pdf"
        assert metadata["department"] == "engineering"
        assert metadata["category"] == "policy"
        assert metadata["tags"] == ["important", "urgent"]

    def test_template_has_timestamps(self):
        """Test template includes timestamps."""
        metadata = create_chunk_metadata_template(
            chunk_id=uuid4(), document_id=uuid4(), text="Sample text"
        )

        assert "created_at" in metadata
        assert "updated_at" in metadata
        assert isinstance(metadata["created_at"], datetime)

    def test_template_calculates_token_count(self):
        """Test template calculates token count from text."""
        text = "This is a sample chunk with several words"
        metadata = create_chunk_metadata_template(
            chunk_id=uuid4(), document_id=uuid4(), text=text
        )

        # Simple word count
        assert metadata["token_count"] == len(text.split())

    def test_template_removes_none_values(self):
        """Test template removes None values for optional fields."""
        metadata = create_chunk_metadata_template(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Sample text",
            # Don't provide optional fields
        )

        # Optional fields should not be present if None
        assert "file_name" not in metadata or metadata["file_name"] is not None

    def test_template_default_scores(self):
        """Test template sets default quality scores."""
        metadata = create_chunk_metadata_template(
            chunk_id=uuid4(), document_id=uuid4(), text="Sample text"
        )

        assert metadata["confidence_score"] == 1.0
        assert metadata["quality_score"] == 1.0

    def test_template_custom_scores(self):
        """Test template accepts custom quality scores."""
        metadata = create_chunk_metadata_template(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="Sample text",
            confidence_score=0.85,
            quality_score=0.92,
        )

        assert metadata["confidence_score"] == 0.85
        assert metadata["quality_score"] == 0.92


class TestSchemaUtilities:
    """Tests for schema utility functions."""

    def test_get_filterable_fields(self):
        """Test retrieving list of filterable fields."""
        fields = get_filterable_fields()

        assert isinstance(fields, list)
        assert len(fields) == 20  # All metadata fields except text

        # Check key fields
        assert "chunk_id" in fields
        assert "document_id" in fields
        assert "department" in fields
        assert "category" in fields
        assert "tags" in fields

    def test_get_searchable_fields(self):
        """Test retrieving list of searchable fields."""
        fields = get_searchable_fields()

        assert isinstance(fields, list)
        assert len(fields) == 1
        assert "text" in fields

    def test_schema_version_defined(self):
        """Test schema version constant is defined."""
        assert SCHEMA_VERSION is not None
        assert isinstance(SCHEMA_VERSION, str)
        assert SCHEMA_VERSION == "2.3.2"
