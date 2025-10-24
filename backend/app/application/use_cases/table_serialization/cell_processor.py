"""Cell processor for handling merged cells and spans (Task 2.1.2).

Specialized component for processing DataFrames to handle merged cells, rowspan/colspan
scenarios, and complex table structures common in financial and scientific reports.

Detects merged cells via repeated values (common in extracted tables) and provides
context markers to preserve table semantics during serialization.

Example:
    >>> import pandas as pd
    >>> from cell_processor import CellProcessor
    >>>
    >>> # Table with hierarchical structure (merged cells)
    >>> df = pd.DataFrame({
    ...     'Category': ['Electronics', 'Electronics', 'Furniture', 'Furniture'],
    ...     'Product': ['Phone', 'Laptop', 'Desk', 'Chair'],
    ...     'Price': [999, 1599, 299, 199]
    ... })
    >>>
    >>> processor = CellProcessor()
    >>> merged_regions = processor.detect_merged_regions(df)
    >>> # Identifies that 'Electronics' appears twice (merged cell)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


class CellProcessor:
    """Processor for handling merged cells and complex table structures.

    Specializes in detecting and formatting merged cells that result from
    hierarchical table structures. Common in:
    - Financial reports (category headers spanning multiple rows)
    - Organizational charts (department headers)
    - Scientific data (measurement category headers)
    - Inventory tables (product category groupings)

    Handles:
    - Repeated values indicating merged cells (hierarchical grouping)
    - Empty cells in structured tables
    - Cell spanning across rows/columns
    - Special formatting preservation for merged content
    - Context markers for markdown output

    Attributes:
        merge_threshold (int): Minimum repeated values to classify as merged
        null_representation (str): How to render None/NaN values

    Example:
        >>> import pandas as pd
        >>> from cell_processor import CellProcessor
        >>>
        >>> # Sales data with category hierarchy
        >>> df = pd.DataFrame({
        ...     'Region': ['US', 'US', 'EU', 'EU'],
        ...     'Country': ['USA', 'Canada', 'Germany', 'France'],
        ...     'Sales': [1000, 800, 1200, 950]
        ... })
        >>>
        >>> processor = CellProcessor(merge_threshold=2)
        >>> # Detects 'US' and 'EU' as merged cells
    """

    def __init__(
        self,
        merge_threshold: int = 2,
        null_representation: str = ""
    ) -> None:
        """Initialize cell processor with configuration.

        Args:
            merge_threshold (int): Number of consecutive repeated values to classify
                                  as merged cell. Default: 2 (appears 2+ times).
            null_representation (str): How to represent None/NaN values in output.
                                      Default: "" (empty string for markdown).
                                      Use "null" for JSON or "[NA]" for clarity.

        Example:
            >>> # Markdown output with empty nulls
            >>> processor_md = CellProcessor(null_representation='')
            >>>
            >>> # JSON output with null values
            >>> processor_json = CellProcessor(null_representation='null')
            >>>
            >>> # Medical data with [NA] markers
            >>> processor_medical = CellProcessor(null_representation='[NA]')
        """
        self.merge_threshold = merge_threshold
        self.null_representation = null_representation

    def process_dataframe(
        self,
        df: pd.DataFrame,
        preserve_merges: bool = True
    ) -> pd.DataFrame:
        """Process DataFrame to handle merged cells and formatting.

        Args:
            df: Original DataFrame
            preserve_merges: If False, replicate merged values to fill cells

        Returns:
            Processed DataFrame
        """
        df_copy = df.copy()

        # Handle null values
        df_copy = self._handle_nulls(df_copy)

        # Process merged cells if needed
        if not preserve_merges:
            df_copy = self._replicate_merged_cells(df_copy)

        # Clean cell content
        df_copy = self._clean_cells(df_copy)

        return df_copy

    def detect_merged_regions(
        self,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect regions with merged cells.

        Args:
            df: DataFrame to analyze

        Returns:
            List of dictionaries describing merged regions:
                {
                    'start_row': int,
                    'end_row': int,
                    'column': str,
                    'value': str,
                    'span_size': int
                }
        """
        merged_regions: List[Dict[str, Any]] = []

        for col_name in df.columns:
            current_value: Optional[str] = None
            start_row: Optional[int] = None
            span_count = 0

            for row_idx, value in enumerate(df[col_name]):
                # Convert to string and handle nulls
                if pd.isna(value):
                    str_value = ""
                else:
                    str_value = str(value).strip()

                # Check if value continues from previous row
                if str_value == current_value and str_value != "":
                    span_count += 1
                else:
                    # Save previous merged region if it meets threshold
                    if span_count >= self.merge_threshold and start_row is not None:
                        merged_regions.append({
                            "start_row": start_row,
                            "end_row": start_row + span_count,
                            "column": col_name,
                            "value": current_value or "",
                            "span_size": span_count + 1
                        })

                    # Start new potential merge
                    current_value = str_value
                    start_row = row_idx
                    span_count = 0

            # Check final region
            if span_count >= self.merge_threshold and start_row is not None:
                merged_regions.append({
                    "start_row": start_row,
                    "end_row": start_row + span_count,
                    "column": col_name,
                    "value": current_value or "",
                    "span_size": span_count + 1
                })

        return merged_regions

    def add_merge_context(
        self,
        df: pd.DataFrame,
        markdown_mode: bool = True
    ) -> pd.DataFrame:
        """Add context to cells in merged regions for better understanding.

        Args:
            df: Original DataFrame
            markdown_mode: If True, add markdown annotations

        Returns:
            DataFrame with context annotations
        """
        df_copy = df.copy()
        merged_regions = self.detect_merged_regions(df)

        for region in merged_regions:
            col = region["column"]
            start = region["start_row"]
            end = region["end_row"]
            value = region["value"]

            if markdown_mode:
                # Add row span notation in markdown
                for row_idx in range(start + 1, end + 1):
                    df_copy.at[row_idx, col] = f"↑ ({value})"
            else:
                # Replicate value to all cells in merged region
                for row_idx in range(start, end + 1):
                    df_copy.at[row_idx, col] = value

        return df_copy

    def handle_empty_cells(
        self,
        df: pd.DataFrame,
        strategy: str = "keep"
    ) -> pd.DataFrame:
        """Handle empty cells in DataFrame.

        Args:
            df: DataFrame to process
            strategy: How to handle empties:
                - 'keep': Keep as empty strings
                - 'null': Convert to NaN
                - 'forward_fill': Fill with previous non-empty value
                - 'n/a': Replace with 'N/A' string

        Returns:
            Processed DataFrame
        """
        df_copy = df.copy()

        if strategy == "keep":
            return df_copy
        elif strategy == "null":
            df_copy = df_copy.replace("", np.nan)
        elif strategy == "forward_fill":
            df_copy = df_copy.replace("", np.nan).fillna(method="ffill")
        elif strategy == "n/a":
            df_copy = df_copy.replace("", "N/A")

        return df_copy

    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert null values to configured representation.

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with nulls handled
        """
        df_copy = df.copy()

        # Replace NaN with configured representation
        for col in df_copy.columns:
            if df_copy[col].dtype in ["float64", "int64"]:
                # Keep numeric NaN as-is (will be handled during serialization)
                pass
            else:
                df_copy[col] = df_copy[col].fillna(self.null_representation)

        return df_copy

    def _replicate_merged_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill merged cell regions by replicating values.

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with merged regions filled
        """
        df_copy = df.copy()
        merged_regions = self.detect_merged_regions(df)

        for region in merged_regions:
            col = region["column"]
            start = region["start_row"]
            end = region["end_row"]
            value = region["value"]

            # Replicate value to all rows in merged region
            for row_idx in range(start, end + 1):
                df_copy.at[row_idx, col] = value

        return df_copy

    def _clean_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean cell content (whitespace, special characters).

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df_copy = df.copy()

        for col in df_copy.select_dtypes(include=["object"]).columns:
            # Strip whitespace
            df_copy[col] = df_copy[col].astype(str).str.strip()

            # Replace multiple spaces with single space
            df_copy[col] = df_copy[col].str.replace(r'\s+', ' ', regex=True)

            # Remove zero-width characters and other invisible unicode
            df_copy[col] = df_copy[col].str.replace(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', regex=True)

        return df_copy
