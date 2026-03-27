"""Configuration for table serialization module (Task 2.1.2).

Defines settings for markdown and JSON serialization of pandas DataFrames
extracted from documents by Docling or other processors.
"""
from __future__ import annotations

from typing import Dict, Any


# Markdown serialization settings
MARKDOWN_CONFIG: Dict[str, Any] = {
    "preserve_headers": True,  # Always include headers in output
    "handle_merged_cells": True,  # Process rowspan/colspan
    "include_metadata": True,  # Add source info as comments
    "max_column_width": 50,  # Truncate cells exceeding this width
    "multi_level_separator": " - ",  # Separator for flattened hierarchical headers
    "null_representation": "",  # How to represent null/NaN values
    "include_row_index": False,  # Include DataFrame index as first column
}

# JSON serialization settings
JSON_CONFIG: Dict[str, Any] = {
    "preserve_dtypes": True,  # Maintain pandas data types in output
    "include_structure": True,  # Include header hierarchy info
    "flatten_multi_index": True,  # Convert multi-level headers to flat strings
    "orient": "records",  # Pandas to_dict() orientation ('records', 'split', 'index')
    "date_format": "iso",  # ISO 8601 format for datetime objects
    "include_dimensions": True,  # Add row/column count to metadata
}

# Table structure validation settings
VALIDATION_CONFIG: Dict[str, Any] = {
    "min_rows": 1,  # Minimum rows to be considered a valid table
    "min_columns": 1,  # Minimum columns to be considered a valid table
    "max_cell_length": 10000,  # Maximum characters per cell
    "allow_empty_headers": False,  # Whether to allow unnamed columns
    "check_consistency": True,  # Validate DataFrame structure integrity
}

# Metadata extraction settings
METADATA_CONFIG: Dict[str, Any] = {
    "extract_caption": True,  # Look for table caption/title
    "extract_source": True,  # Include source document info
    "extract_page": True,  # Include page number if available
    "extract_dimensions": True,  # Include row/column counts
    "extract_dtypes": True,  # Include data type information
    "include_hash": True,  # Generate content hash for deduplication
}
