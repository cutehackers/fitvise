"""Quick validation script for Weaviate schema (Task 2.3.2).

This script validates the schema implementation without requiring pytest.
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.infrastructure.external_services.vector_stores.weaviate_schema import (
    create_chunk_class_schema,
    validate_chunk_metadata,
    create_chunk_metadata_template,
    get_filterable_fields,
    get_searchable_fields,
    SCHEMA_VERSION,
)


def validate_schema_structure():
    """Validate schema structure."""
    print("✓ Testing schema structure...")
    schema = create_chunk_class_schema()

    assert schema["class"] == "Chunk", "Schema class should be 'Chunk'"
    assert schema["vectorizer"] == "none", "Vectorizer should be 'none'"
    assert len(schema["properties"]) == 21, f"Should have 21 properties, got {len(schema['properties'])}"

    property_names = [prop["name"] for prop in schema["properties"]]

    # Core fields
    assert "text" in property_names
    assert "chunk_id" in property_names
    assert "document_id" in property_names

    # Enhanced metadata
    assert "department" in property_names
    assert "category" in property_names
    assert "tags" in property_names
    assert "quality_score" in property_names

    print(f"  ✓ Schema has {len(schema['properties'])} properties")
    print(f"  ✓ HNSW distance: {schema['vectorIndexConfig']['distance']}")


def validate_metadata_functions():
    """Validate metadata helper functions."""
    print("\n✓ Testing metadata validation...")

    # Valid metadata
    valid_meta = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "text": "Sample chunk",
        "sequence": 0,
    }
    validate_chunk_metadata(valid_meta)
    print("  ✓ Valid metadata passes")

    # Invalid metadata (missing text)
    invalid_meta = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
    }
    try:
        validate_chunk_metadata(invalid_meta)
        raise AssertionError("Should have failed validation")
    except ValueError:
        print("  ✓ Invalid metadata fails correctly")


def validate_template_creation():
    """Validate template creation."""
    print("\n✓ Testing template creation...")

    chunk_id = uuid4()
    doc_id = uuid4()

    template = create_chunk_metadata_template(
        chunk_id=chunk_id,
        document_id=doc_id,
        text="Sample text",
        department="engineering",
        tags=["test", "example"]
    )

    assert template["chunk_id"] == chunk_id
    assert template["document_id"] == doc_id
    assert template["department"] == "engineering"
    assert template["tags"] == ["test", "example"]
    assert "created_at" in template
    print(f"  ✓ Template created with {len(template)} fields")


def validate_utility_functions():
    """Validate utility functions."""
    print("\n✓ Testing utility functions...")

    filterable = get_filterable_fields()
    searchable = get_searchable_fields()

    assert len(filterable) == 20, f"Should have 20 filterable fields, got {len(filterable)}"
    assert len(searchable) == 1, f"Should have 1 searchable field, got {len(searchable)}"
    assert "text" in searchable
    assert "department" in filterable

    print(f"  ✓ {len(filterable)} filterable fields")
    print(f"  ✓ {len(searchable)} searchable fields")


def main():
    """Run all validations."""
    print("=" * 60)
    print("Weaviate Schema Validation (Task 2.3.2)")
    print(f"Schema Version: {SCHEMA_VERSION}")
    print("=" * 60)

    try:
        validate_schema_structure()
        validate_metadata_functions()
        validate_template_creation()
        validate_utility_functions()

        print("\n" + "=" * 60)
        print("✅ All validations passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
