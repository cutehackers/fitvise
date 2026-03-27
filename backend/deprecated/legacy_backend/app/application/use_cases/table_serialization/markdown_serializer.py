"""Markdown serializer for table conversion (Task 2.1.2).

Converts pandas DataFrames to markdown format suitable for:
- Semantic chunking and embedding (input to RAG vector stores)
- Human-readable documentation and reports
- LLM context injection with preserving table structure

The MarkdownSerializer produces clean, readable markdown tables that preserve:
- Multi-level/hierarchical column headers
- Metadata context (source file, page number, table caption)
- Data type information through formatting
- Null/missing value indicators

Example:
    Converting a financial report table to markdown:

    >>> import pandas as pd
    >>> from markdown_serializer import MarkdownSerializer
    >>>
    >>> df = pd.DataFrame(
    ...     {'Revenue ($M)': [125.5, 138.7], 'Expenses ($M)': [65.2, 70.1]},
    ...     index=['Q1 2024', 'Q2 2024']
    ... )
    >>>
    >>> serializer = MarkdownSerializer()
    >>> markdown = serializer.serialize(
    ...     df,
    ...     metadata={
    ...         'file_name': 'financial_report.pdf',
    ...         'page': 15,
    ...         'caption': 'Quarterly Revenue and Expenses'
    ...     }
    ... )
    >>> print(markdown)
    # Quarterly Revenue and Expenses

    | Quarter | Revenue ($M) | Expenses ($M) |
    |---------|-------------|---------------|
    | Q1 2024 | 125.5 | 65.2 |
    | Q2 2024 | 138.7 | 70.1 |

    Source: financial_report.pdf, Page 15
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .base_serializer import BaseSerializer, SerializationError, ValidationError
from .config import MARKDOWN_CONFIG, VALIDATION_CONFIG
from .header_parser import HeaderParser
from .cell_processor import CellProcessor
from .metadata_extractor import MetadataExtractor


class MarkdownSerializer(BaseSerializer):
    """Serializer for converting DataFrames to markdown tables.

    Transforms pandas DataFrames into clean, semantic markdown tables optimized for
    embedding and LLM consumption. Handles complex table structures including
    multi-level headers, merged cells, and various data types.

    Features:
    - Multi-level header flattening with configurable separator
    - Merged cell context preservation (detects repeated values)
    - Metadata injection as markdown comments
    - Intelligent column width management
    - Clean table formatting with proper alignment
    - Support for nullable values and special formatting

    Attributes:
        header_parser (HeaderParser): Handles multi-level column indices
        cell_processor (CellProcessor): Processes and formats cell values
        metadata_extractor (MetadataExtractor): Extracts table metadata

    Example:
        >>> import pandas as pd
        >>> from markdown_serializer import MarkdownSerializer
        >>>
        >>> # Complex medical data with nulls
        >>> df = pd.DataFrame({
        ...     'Test Name': ['Hemoglobin', 'Glucose', 'Cholesterol'],
        ...     'Patient Result': [14.2, 95.0, None],
        ...     'Reference Range': ['13.5-17.5 g/dL', '70-100 mg/dL', '<200 mg/dL'],
        ... })
        >>>
        >>> serializer = MarkdownSerializer()
        >>> markdown = serializer.serialize(
        ...     df,
        ...     metadata={
        ...         'file_name': 'lab_report.pdf',
        ...         'caption': 'Complete Blood Count',
        ...         'timestamp': '2024-10-24'
        ...     }
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize markdown serializer with optional configuration.

        Args:
            config (dict, optional): Configuration overrides. Common options:
                - max_column_width: Maximum width for column content (default: 50)
                - multi_level_separator: Separator for flattening headers (default: " - ")
                - null_representation: How to represent None/NaN values (default: "")
                - include_hash: Whether to include content hash (default: True)

        Example:
            >>> config = {
            ...     'max_column_width': 100,
            ...     'multi_level_separator': ' | ',
            ...     'null_representation': '[missing]'
            ... }
            >>> serializer = MarkdownSerializer(config)
        """
        # Merge default config with overrides
        full_config = {**MARKDOWN_CONFIG, **VALIDATION_CONFIG}
        if config:
            full_config.update(config)

        super().__init__(full_config)

        # Initialize specialized components
        separator = self.config.get("multi_level_separator", " - ")
        self.header_parser = HeaderParser(separator=separator)

        null_repr = self.config.get("null_representation", "")
        self.cell_processor = CellProcessor(null_representation=null_repr)

        self.metadata_extractor = MetadataExtractor(
            include_hash=self.config.get("include_hash", True)
        )

    def serialize(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert DataFrame to markdown table format.

        Transforms a pandas DataFrame into a clean markdown table with optional metadata.
        Automatically handles:
        - Multi-level column headers (flattened with separator)
        - Merged cells (detected via repeated values, marked with context arrows)
        - Null/NaN values (rendered as empty strings)
        - Data validation and preprocessing

        Args:
            df (pd.DataFrame): DataFrame to serialize
            metadata (dict, optional): Source metadata with keys:
                - 'file_name': Source file name
                - 'page': Page number in source document
                - 'caption': Title for the table
                - 'source': Full source reference
                - 'timestamp': When table was extracted

        Returns:
            str: Markdown-formatted table as a complete string

        Raises:
            SerializationError: If conversion fails during processing
            ValidationError: If DataFrame doesn't meet validation requirements

        Example:
            >>> import pandas as pd
            >>> from markdown_serializer import MarkdownSerializer
            >>>
            >>> # API documentation table
            >>> df = pd.DataFrame({
            ...     'Endpoint': ['/api/users', '/api/posts'],
            ...     'Method': ['GET', 'POST'],
            ...     'Rate Limit': ['100/min', '10/min']
            ... })
            >>>
            >>> serializer = MarkdownSerializer()
            >>> markdown = serializer.serialize(
            ...     df,
            ...     metadata={
            ...         'caption': 'API Endpoints Reference',
            ...         'file_name': 'api_docs.md'
            ...     }
            ... )
            >>>
            >>> print(markdown)
            # API Endpoints Reference
            | Endpoint | Method | Rate Limit |
            |----------|--------|------------|
            | /api/users | GET | 100/min |
            | /api/posts | POST | 10/min |
            Source: api_docs.md
        """
        try:
            # Validate structure
            self.validate_structure(df)

            # Preprocess DataFrame
            df_processed = self.preprocess_dataframe(df)

            # Handle merged cells if configured
            if self.config.get("handle_merged_cells", True):
                df_processed = self.cell_processor.add_merge_context(
                    df_processed,
                    markdown_mode=True
                )

            # Build markdown output
            markdown_parts: List[str] = []

            # Add metadata as comments if configured
            if self.config.get("include_metadata", True) and metadata:
                markdown_parts.extend(self._format_metadata(metadata))

            # Add table caption if available
            if metadata and metadata.get("caption"):
                markdown_parts.append(f"# {metadata['caption']}\n")

            # Build table
            markdown_parts.append(self._build_table(df_processed))

            # Add source reference if available
            if metadata and self.config.get("include_metadata", True):
                source_ref = self._format_source_reference(metadata)
                if source_ref:
                    markdown_parts.append(f"\n{source_ref}")

            return "\n".join(markdown_parts)

        except ValidationError:
            raise
        except Exception as exc:
            raise self.handle_error(exc, "Markdown serialization failed")

    def _build_table(self, df: pd.DataFrame) -> str:
        """Build markdown table from DataFrame.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Markdown table string
        """
        # Flatten multi-level headers if needed
        df_display = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df_display.columns = self.header_parser.flatten_multi_level_headers(df.columns)

        # Get column headers
        include_index = self.config.get("include_row_index", False)
        headers = self.header_parser.get_column_names(df_display, include_index=include_index)

        # Build header row
        header_row = "| " + " | ".join(headers) + " |"

        # Build separator row
        separator = "|" + "|".join(["-" * max(10, len(h)) for h in headers]) + "|"

        # Build data rows
        data_rows: List[str] = []
        max_width = self.config.get("max_column_width", 50)

        for idx, row in df_display.iterrows():
            cells: List[str] = []

            if include_index:
                cells.append(self._format_cell(str(idx), max_width))

            for col in df_display.columns:
                value = row[col]
                cells.append(self._format_cell(value, max_width))

            data_rows.append("| " + " | ".join(cells) + " |")

        # Combine all parts
        table_lines = [header_row, separator] + data_rows
        return "\n".join(table_lines)

    def _format_cell(self, value: Any, max_width: int) -> str:
        """Format a single cell value for markdown.

        Args:
            value: Cell value
            max_width: Maximum width for cell content

        Returns:
            Formatted cell string
        """
        # Handle None/NaN
        if pd.isna(value):
            return self.config.get("null_representation", "")

        # Convert to string
        cell_str = str(value).strip()

        # Truncate if too long
        if len(cell_str) > max_width:
            cell_str = cell_str[:max_width - 3] + "..."

        # Escape markdown special characters
        cell_str = (
            cell_str
            .replace("|", "\\|")
            .replace("\n", " ")
            .replace("\r", "")
        )

        return cell_str

    def _format_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Format metadata as markdown comments.

        Args:
            metadata: Metadata dictionary

        Returns:
            List of markdown comment lines
        """
        comments: List[str] = []

        if "source" in metadata:
            source = metadata["source"]
            if source.get("file"):
                comments.append(f"<!-- Source: {source['file']} -->")
            if source.get("page"):
                comments.append(f"<!-- Page: {source['page']} -->")

        return comments

    def _format_source_reference(self, metadata: Dict[str, Any]) -> str:
        """Format source reference as markdown italic text.

        Args:
            metadata: Metadata dictionary

        Returns:
            Formatted source reference string
        """
        parts: List[str] = []

        # Handle both nested "source" dict and flat metadata structure
        if "source" in metadata and isinstance(metadata["source"], dict):
            source = metadata["source"]
            if source.get("file"):
                parts.append(f"Source: {source['file']}")
            if source.get("page"):
                parts.append(f"Page {source['page']}")
        else:
            # Handle flat structure
            if metadata.get("file_name"):
                parts.append(f"Source: {metadata['file_name']}")
            if metadata.get("page"):
                parts.append(f"Page {metadata['page']}")

        if parts:
            return f"*{', '.join(parts)}*"

        return ""
