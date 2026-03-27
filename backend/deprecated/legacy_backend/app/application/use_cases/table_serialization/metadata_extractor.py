"""Metadata extractor for table context and provenance (Task 2.1.2).

Specialized component for extracting and organizing comprehensive metadata from
DataFrames and source documents. Essential for maintaining data provenance in
RAG systems and ensuring tables can be traced back to source documents.

Extracts metadata including:
- Source document information (file name, page, location)
- Table dimensions and statistics
- Data type information for each column
- Content hash for deduplication and change detection
- Descriptive statistics (optional)

Example:
    >>> import pandas as pd
    >>> from metadata_extractor import MetadataExtractor
    >>>
    >>> df = pd.DataFrame({
    ...     'Name': ['Alice', 'Bob', 'Charlie'],
    ...     'Score': [95.5, 87.0, 92.3]
    ... })
    >>>
    >>> extractor = MetadataExtractor(include_hash=True)
    >>> metadata = extractor.extract(
    ...     df,
    ...     source_metadata={
    ...         'file_name': 'exam_results.pdf',
    ...         'page': 3,
    ...         'caption': 'Q4 2024 Exam Results'
    ...     }
    ... )
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


class MetadataExtractor:
    """Extractor for comprehensive table metadata and context information.

    Generates rich metadata for tables suitable for:
    - Maintaining source provenance in RAG pipelines
    - Deduplication and change detection via content hashing
    - Statistics and exploratory analysis
    - Structured data catalogs and data lineage tracking

    Collects metadata including:
    - Source provenance (file, page, location in document)
    - Structural information (dimensions, header structure, dtypes)
    - Content summary (hash for deduplication)
    - Descriptive statistics (optional, for numeric columns)
    - Table context (caption, title, processor information)

    Attributes:
        include_hash (bool): Whether to generate content hash
        include_statistics (bool): Whether to include descriptive statistics

    Example:
        >>> import pandas as pd
        >>> from metadata_extractor import MetadataExtractor
        >>>
        >>> # Sales data with rich metadata
        >>> df = pd.DataFrame({
        ...     'Date': pd.date_range('2024-01-01', periods=5),
        ...     'Sales': [1000, 1200, 1100, 1350, 1400],
        ...     'Region': ['US', 'EU', 'US', 'APAC', 'EU']
        ... })
        >>>
        >>> extractor = MetadataExtractor(include_statistics=True)
        >>> metadata = extractor.extract(
        ...     df,
        ...     source_metadata={
        ...         'file_name': 'sales_report_2024.pdf',
        ...         'page': 5,
        ...         'caption': 'Monthly Sales by Region'
        ...     }
        ... )
        >>> print(metadata['structure']['row_count'])  # 5
        >>> print(metadata['statistics']['Sales']['mean'])  # Average sales
    """

    def __init__(
        self,
        include_hash: bool = True,
        include_statistics: bool = False
    ) -> None:
        """Initialize metadata extractor with configuration.

        Args:
            include_hash (bool): Generate SHA256 hash of table content for
                                deduplication and change detection.
                                Default: True
            include_statistics (bool): Include descriptive statistics (mean, std, min, max)
                                      for numeric columns. Useful for exploratory analysis
                                      but increases metadata size.
                                      Default: False

        Example:
            >>> # For RAG systems (minimal metadata for speed)
            >>> extractor_fast = MetadataExtractor(include_hash=True)
            >>>
            >>> # For data catalogs (comprehensive metadata)
            >>> extractor_full = MetadataExtractor(
            ...     include_hash=True,
            ...     include_statistics=True
            ... )
        """
        self.include_hash = include_hash
        self.include_statistics = include_statistics

    def extract(
        self,
        df: pd.DataFrame,
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata from DataFrame.

        Collects structural, statistical, and provenance information from the DataFrame
        and merges with source document metadata.

        Args:
            df (pd.DataFrame): DataFrame to extract metadata from
            source_metadata (dict, optional): Metadata from source document with keys:
                - 'file_name': Source file name (string)
                - 'page': Page number in document (int)
                - 'caption': Table title/caption (string)
                - 'processor': Name of extraction processor (string)
                - Any other custom metadata

        Returns:
            Dict[str, Any]: Comprehensive metadata structure containing:
                - 'source': Provenance information
                - 'structure': Table dimensions and header info
                - 'data_types': Data type for each column
                - 'content_hash': SHA256 hash of content (if enabled)
                - 'statistics': Numeric column statistics (if enabled)

        Example:
            >>> df = pd.DataFrame({
            ...     'Product': ['Widget', 'Gadget', 'Gizmo'],
            ...     'Price': [9.99, 19.99, 29.99]
            ... })
            >>>
            >>> extractor = MetadataExtractor()
            >>> metadata = extractor.extract(
            ...     df,
            ...     source_metadata={
            ...         'file_name': 'catalog.pdf',
            ...         'page': 1,
            ...         'caption': 'Product Prices'
            ...     }
            ... )
            >>>
            >>> # Access metadata
            >>> print(metadata['structure']['row_count'])  # 3
            >>> print(metadata['data_types'])
            # {'Product': 'string', 'Price': 'float'}
        """
        metadata: Dict[str, Any] = {}

        # Source information
        if source_metadata:
            metadata["source"] = {
                "file": source_metadata.get("file_name", "unknown"),
                "page": source_metadata.get("page"),
                "caption": source_metadata.get("caption"),
                "processor": source_metadata.get("processor", "unknown"),
            }

        # Structural information
        metadata["structure"] = self._extract_structure(df)

        # Data type information
        metadata["data_types"] = self._extract_dtypes(df)

        # Content hash
        if self.include_hash:
            metadata["content_hash"] = self._generate_hash(df)

        # Column statistics
        if self.include_statistics:
            metadata["statistics"] = self._extract_statistics(df)

        return metadata

    def _extract_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract structural information about the table.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with structural metadata
        """
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "has_multi_level_headers": isinstance(df.columns, pd.MultiIndex),
            "header_levels": df.columns.nlevels if isinstance(df.columns, pd.MultiIndex) else 1,
            "has_row_index": df.index.name is not None,
            "total_cells": len(df) * len(df.columns),
            "shape": list(df.shape),
        }

    def _extract_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """Extract data type information for each column.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary mapping column names to data types
        """
        dtype_map: Dict[str, str] = {}

        for col in df.columns:
            col_name = str(col)
            dtype = df[col].dtype

            # Map pandas dtypes to human-readable types
            if pd.api.types.is_integer_dtype(dtype):
                dtype_map[col_name] = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                dtype_map[col_name] = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                dtype_map[col_name] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                dtype_map[col_name] = "datetime"
            elif pd.api.types.is_timedelta64_dtype(dtype):
                dtype_map[col_name] = "timedelta"
            elif isinstance(dtype, pd.CategoricalDtype):
                dtype_map[col_name] = "categorical"
            else:
                dtype_map[col_name] = "string"

        return dtype_map

    def _generate_hash(self, df: pd.DataFrame) -> str:
        """Generate content hash for table deduplication.

        Args:
            df: DataFrame to hash

        Returns:
            SHA256 hash of table content
        """
        # Convert DataFrame to string representation
        content = df.to_string()

        # Generate SHA256 hash
        hash_obj = hashlib.sha256(content.encode("utf-8"))
        return hash_obj.hexdigest()

    def _extract_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract statistical information for numeric columns.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with statistics per numeric column
        """
        stats: Dict[str, Any] = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_name = str(col)
            col_data = df[col].dropna()

            if len(col_data) > 0:
                stats[col_name] = {
                    "count": int(col_data.count()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "null_count": int(df[col].isna().sum()),
                }

        return stats

    def extract_caption(
        self,
        markdown: str,
        table_position: int = 0
    ) -> Optional[str]:
        """Extract table caption from markdown document.

        Args:
            markdown: Markdown text containing the table
            table_position: Position/index of table in document

        Returns:
            Caption text if found, None otherwise
        """
        # Look for common caption patterns before table
        # Examples: "Table 1: ...", "# Financial Summary", etc.
        lines = markdown.split("\n")

        # Search backwards from table position for caption
        for i in range(max(0, table_position - 5), table_position):
            if i < len(lines):
                line = lines[i].strip()

                # Check for table caption patterns
                if line.startswith("Table") and ":" in line:
                    return line.split(":", 1)[1].strip()
                elif line.startswith("#"):
                    # Markdown heading
                    return line.lstrip("#").strip()
                elif line and not line.startswith("|"):
                    # Non-table line immediately before table
                    return line

        return None

    def add_provenance(
        self,
        metadata: Dict[str, Any],
        file_path: str,
        page_number: Optional[int] = None,
        table_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add provenance information to metadata.

        Args:
            metadata: Existing metadata dictionary
            file_path: Path to source file
            page_number: Page number where table appears
            table_id: Unique identifier for table

        Returns:
            Updated metadata with provenance
        """
        if "source" not in metadata:
            metadata["source"] = {}

        metadata["source"]["file"] = file_path

        if page_number is not None:
            metadata["source"]["page"] = page_number

        if table_id is not None:
            metadata["table_id"] = table_id

        return metadata
