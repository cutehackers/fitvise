"""Manual test script for table serialization (Task 2.1.2).

Run this script to verify table serialization works across all domains.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend directory to path so we can import app
backend_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.application.use_cases.table_serialization import (
    MarkdownSerializer,
    JSONSerializer
)
from tests.unit.table_serialization.fixtures.sample_tables import SAMPLE_TABLES


def test_markdown_serialization():
    """Test markdown serialization across all sample tables."""
    print("=" * 80)
    print("MARKDOWN SERIALIZATION TESTS")
    print("=" * 80)

    serializer = MarkdownSerializer()

    for table_name, table_func in SAMPLE_TABLES.items():
        print(f"\n--- Testing: {table_name} ---")
        df, metadata = table_func()

        try:
            result = serializer.serialize(df, metadata)
            print(f"✅ SUCCESS: {len(result)} characters")
            print(f"\nPreview (first 500 chars):")
            print(result[:500])
            print("...")
        except Exception as exc:
            print(f"❌ FAILED: {exc}")


def test_json_serialization():
    """Test JSON serialization across all sample tables."""
    print("\n" + "=" * 80)
    print("JSON SERIALIZATION TESTS")
    print("=" * 80)

    serializer = JSONSerializer()

    for table_name, table_func in SAMPLE_TABLES.items():
        print(f"\n--- Testing: {table_name} ---")
        df, metadata = table_func()

        try:
            result = serializer.serialize(df, metadata)
            print(f"✅ SUCCESS: {len(result)} keys in result")
            print(f"Structure keys: {list(result.keys())}")
            print(f"Data rows: {len(result.get('data', []))}")
            print(f"Columns: {len(result.get('structure', {}).get('columns', []))}")
        except Exception as exc:
            print(f"❌ FAILED: {exc}")


def test_end_to_end():
    """Test complete workflow: DataFrame -> Markdown + JSON."""
    print("\n" + "=" * 80)
    print("END-TO-END WORKFLOW TEST")
    print("=" * 80)

    # Use financial table as comprehensive example
    df, metadata = SAMPLE_TABLES["financial"]()

    print("\nOriginal DataFrame:")
    print(df)

    # Serialize to markdown
    md_serializer = MarkdownSerializer()
    markdown = md_serializer.serialize(df, metadata)

    print("\n--- Markdown Output ---")
    print(markdown)

    # Serialize to JSON
    json_serializer = JSONSerializer()
    json_data = json_serializer.serialize(df, metadata)

    print("\n--- JSON Output (formatted) ---")
    import json
    print(json.dumps(json_data, indent=2))

    print("\n✅ END-TO-END TEST COMPLETE")


if __name__ == "__main__":
    test_markdown_serialization()
    test_json_serialization()
    test_end_to_end()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
