"""Table serialization module for converting DataFrames to markdown and JSON (Task 2.1.2).

This module provides universal table serialization capabilities for any domain:
- Markdown format: For semantic chunking and embedding
- JSON format: For structured queries and metadata storage
- Multi-level header handling
- Merged cell processing
- Comprehensive metadata extraction

Usage:
    from app.application.use_cases.table_serialization import (
        MarkdownSerializer,
        JSONSerializer
    )

    # Markdown serialization
    md_serializer = MarkdownSerializer()
    markdown_table = md_serializer.serialize(df, metadata)

    # JSON serialization
    json_serializer = JSONSerializer()
    json_data = json_serializer.serialize(df, metadata)
"""
from __future__ import annotations

from .base_serializer import BaseSerializer, SerializationError, ValidationError
from .markdown_serializer import MarkdownSerializer
from .json_serializer import JSONSerializer
from .header_parser import HeaderParser
from .cell_processor import CellProcessor
from .metadata_extractor import MetadataExtractor
from .config import (
    MARKDOWN_CONFIG,
    JSON_CONFIG,
    VALIDATION_CONFIG,
    METADATA_CONFIG
)

__all__ = [
    # Main serializers
    "MarkdownSerializer",
    "JSONSerializer",

    # Base classes and exceptions
    "BaseSerializer",
    "SerializationError",
    "ValidationError",

    # Specialized components
    "HeaderParser",
    "CellProcessor",
    "MetadataExtractor",

    # Configuration
    "MARKDOWN_CONFIG",
    "JSON_CONFIG",
    "VALIDATION_CONFIG",
    "METADATA_CONFIG",
]
