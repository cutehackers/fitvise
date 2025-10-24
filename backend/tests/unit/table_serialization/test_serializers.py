"""Comprehensive tests for table serializers (Task 2.1.2).

Tests markdown and JSON serialization across multiple domains:
- Financial reports (complex multi-level headers)
- Medical data (scientific measurements)
- API documentation (technical specifications)
- Chemistry data (scientific notation)
- Business metrics (general use case)
- Edge cases (nulls, merged cells)
"""
from __future__ import annotations

import pytest
import pandas as pd
import json

from app.application.use_cases.table_serialization import (
    MarkdownSerializer,
    JSONSerializer,
    ValidationError
)
from tests.unit.table_serialization.fixtures.sample_tables import SAMPLE_TABLES


class TestMarkdownSerializer:
    """Test suite for MarkdownSerializer."""

    def test_financial_table_serialization(self):
        """Test markdown serialization of complex financial table."""
        df, metadata = SAMPLE_TABLES["financial"]()
        serializer = MarkdownSerializer()

        result = serializer.serialize(df, metadata)

        # Verify markdown structure
        assert "| Metric |" in result
        assert "Q1 2024 - Actual" in result
        assert "Q2 2024 - Budget" in result
        assert "Revenue ($M)" in result
        assert "125.5" in result

        # Verify metadata inclusion
        assert "Source: financial_report_q2_2024.pdf" in result
        assert "Page 15" in result

        # Verify table caption
        assert "Quarterly Financial Performance Summary" in result

    def test_medical_table_serialization(self):
        """Test markdown serialization of medical lab results."""
        df, metadata = SAMPLE_TABLES["medical"]()
        serializer = MarkdownSerializer()

        result = serializer.serialize(df, metadata)

        # Verify structure
        assert "| Test Name |" in result
        assert "Hemoglobin" in result
        assert "13.5-17.5 g/dL" in result
        assert "Normal" in result

        # Verify metadata
        assert "lab_results_2024_10_24.pdf" in result

    def test_api_documentation_serialization(self):
        """Test markdown serialization of API docs table."""
        df, metadata = SAMPLE_TABLES["api_docs"]()
        serializer = MarkdownSerializer()

        result = serializer.serialize(df, metadata)

        # Verify endpoints
        assert "/api/users" in result
        assert "POST" in result
        assert "Rate Limit" in result
        assert "100/min" in result

    def test_chemistry_table_serialization(self):
        """Test markdown serialization of chemistry data."""
        df, metadata = SAMPLE_TABLES["chemistry"]()
        serializer = MarkdownSerializer()

        result = serializer.serialize(df, metadata)

        # Verify compounds and properties
        assert "Ethanol" in result
        assert "78.4" in result  # Boiling point
        assert "0.789" in result  # Density
        assert "Miscible" in result

    def test_null_handling(self):
        """Test markdown serialization handles null values correctly."""
        df, metadata = SAMPLE_TABLES["with_nulls"]()
        serializer = MarkdownSerializer()

        result = serializer.serialize(df, metadata)

        # Verify table structure is maintained
        assert "| Product |" in result
        assert "Widget A" in result

        # Null cells should be empty in markdown
        lines = result.split("\n")
        data_lines = [line for line in lines if line.startswith("|") and "---" not in line]
        assert len(data_lines) >= 4  # Header + 4 data rows

    def test_merged_cells_context(self):
        """Test markdown serialization adds context for merged cells."""
        df, metadata = SAMPLE_TABLES["with_merged_cells"]()
        serializer = MarkdownSerializer()

        result = serializer.serialize(df, metadata)

        # Verify hierarchical structure
        assert "Electronics" in result
        assert "Furniture" in result
        assert "Clothing" in result

        # Check for merge context markers (optional based on implementation)
        # This validates the table is properly formed
        assert "iPhone" in result
        assert "Desk" in result

    def test_empty_dataframe_validation(self):
        """Test that empty DataFrames raise ValidationError."""
        serializer = MarkdownSerializer()
        df = pd.DataFrame()

        with pytest.raises(ValidationError):
            serializer.serialize(df)

    def test_invalid_input_validation(self):
        """Test that non-DataFrame inputs raise ValidationError."""
        serializer = MarkdownSerializer()

        with pytest.raises(ValidationError):
            serializer.serialize(None)

        with pytest.raises(ValidationError):
            serializer.serialize("not a dataframe")


class TestJSONSerializer:
    """Test suite for JSONSerializer."""

    def test_financial_table_json_serialization(self):
        """Test JSON serialization of financial table with structure."""
        df, metadata = SAMPLE_TABLES["financial"]()
        serializer = JSONSerializer()

        result = serializer.serialize(df, metadata)

        # Verify top-level structure
        assert "data" in result
        assert "structure" in result
        assert "metadata" in result
        assert "data_types" in result

        # Verify source metadata
        assert result["source"]["file"] == "financial_report_q2_2024.pdf"
        assert result["source"]["page"] == 15

        # Verify structure information
        assert "columns" in result["structure"]
        assert "Q1 2024 - Actual" in result["structure"]["columns"]

        # Verify data
        assert len(result["data"]) == 5  # 5 rows
        assert result["data"][0]["Metric"] == "Revenue ($M)"

    def test_medical_table_json_serialization(self):
        """Test JSON serialization of medical data."""
        df, metadata = SAMPLE_TABLES["medical"]()
        serializer = JSONSerializer()

        result = serializer.serialize(df, metadata)

        # Verify data types preserved
        assert "data_types" in result
        assert result["data_types"]["Result"] in ["float", "integer"]

        # Verify data content
        assert len(result["data"]) == 5
        assert result["data"][0]["Test Name"] == "Hemoglobin"

    def test_null_handling_in_json(self):
        """Test JSON serialization handles nulls as None."""
        df, metadata = SAMPLE_TABLES["with_nulls"]()
        serializer = JSONSerializer()

        result = serializer.serialize(df, metadata)

        # Verify nulls converted to None (JSON null)
        data = result["data"]
        assert any(record.get("Price") is None for record in data)
        assert any(record.get("Category") is None for record in data)

    def test_json_to_string_method(self):
        """Test convenience method for JSON string output."""
        df, metadata = SAMPLE_TABLES["business"]()
        serializer = JSONSerializer()

        json_str = serializer.to_json_string(df, metadata, indent=2)

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert "data" in parsed
        assert "metadata" in parsed

    def test_data_type_preservation(self):
        """Test that data types are correctly identified and preserved."""
        df, metadata = SAMPLE_TABLES["business"]()
        serializer = JSONSerializer()

        result = serializer.serialize(df, metadata)

        # Check data type mappings
        dtypes = result["data_types"]
        assert dtypes["Sales ($K)"] in ["integer", "float"]
        assert dtypes["Growth (%)"] == "float"
        assert dtypes["Region"] == "string"

    def test_multi_level_header_structure(self):
        """Test JSON captures multi-level header hierarchy."""
        df, metadata = SAMPLE_TABLES["financial"]()
        serializer = JSONSerializer()

        result = serializer.serialize(df, metadata)

        # Verify multi-level structure captured
        assert result["structure"].get("multi_level_headers", False)
        assert result["structure"]["header_levels"] > 1
        assert "header_hierarchy" in result["structure"]

    def test_metadata_extraction(self):
        """Test comprehensive metadata extraction."""
        df, metadata = SAMPLE_TABLES["chemistry"]()
        serializer = JSONSerializer()

        result = serializer.serialize(df, metadata)

        # Verify metadata completeness
        meta = result["metadata"]
        assert "row_count" in meta
        assert meta["row_count"] == 5
        assert "column_count" in meta
        assert meta["column_count"] == 5

    def test_empty_dataframe_validation(self):
        """Test that empty DataFrames raise ValidationError."""
        serializer = JSONSerializer()
        df = pd.DataFrame()

        with pytest.raises(ValidationError):
            serializer.serialize(df)


class TestCrossFormatConsistency:
    """Test consistency between markdown and JSON serialization."""

    def test_same_table_both_formats(self):
        """Verify both formats handle same table consistently."""
        df, metadata = SAMPLE_TABLES["business"]()

        md_serializer = MarkdownSerializer()
        json_serializer = JSONSerializer()

        md_result = md_serializer.serialize(df, metadata)
        json_result = json_serializer.serialize(df, metadata)

        # Both should succeed
        assert md_result is not None
        assert json_result is not None

        # Both should preserve row count
        md_lines = [line for line in md_result.split("\n") if line.startswith("|") and "---" not in line]
        assert len(md_lines) - 1 == len(json_result["data"])  # -1 for header

    def test_all_sample_tables(self):
        """Test that all sample tables serialize successfully."""
        md_serializer = MarkdownSerializer()
        json_serializer = JSONSerializer()

        for table_name, table_func in SAMPLE_TABLES.items():
            df, metadata = table_func()

            # Both formats should work
            md_result = md_serializer.serialize(df, metadata)
            json_result = json_serializer.serialize(df, metadata)

            assert md_result is not None, f"Markdown failed for {table_name}"
            assert json_result is not None, f"JSON failed for {table_name}"
            assert len(md_result) > 0, f"Empty markdown for {table_name}"
            assert len(json_result) > 0, f"Empty JSON for {table_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
