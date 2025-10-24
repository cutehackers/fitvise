"""Header parser for multi-level table headers (Task 2.1.2).

Specialized component for detecting and flattening hierarchical column/row headers
from pandas DataFrames. Essential for handling complex tables like financial reports
with quarterly/monthly subdivisions or product categories.

Example:
    >>> import pandas as pd
    >>> from header_parser import HeaderParser
    >>>
    >>> # Multi-level financial headers
    >>> df = pd.DataFrame(
    ...     [[100, 95, 110, 105]],
    ...     columns=pd.MultiIndex.from_tuples([
    ...         ('Q1 2024', 'Actual'),
    ...         ('Q1 2024', 'Budget'),
    ...         ('Q2 2024', 'Actual'),
    ...         ('Q2 2024', 'Budget')
    ...     ])
    ... )
    >>>
    >>> parser = HeaderParser(separator=' | ')
    >>> headers = parser.flatten_multi_level_headers(df.columns)
    >>> print(headers)
    ['Q1 2024 | Actual', 'Q1 2024 | Budget', 'Q2 2024 | Actual', 'Q2 2024 | Budget']
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


class HeaderParser:
    """Parser for detecting and flattening multi-level table headers.

    Specializes in extracting and transforming hierarchical column structures
    commonly found in financial, scientific, and business documents.

    Handles DataFrames with:
    - Simple column headers (single level)
    - Multi-level column headers (pandas MultiIndex)
    - Row headers (DataFrame index)
    - Hierarchical structures (nested categories)

    Attributes:
        separator (str): String used to join multi-level header parts

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     [[1, 2, 3, 4]],
        ...     columns=pd.MultiIndex.from_product(
        ...         [['Q1', 'Q2'], ['Actual', 'Budget']]
        ...     )
        ... )
        >>>
        >>> parser = HeaderParser()
        >>> info = parser.parse(df)
        >>> print(info['has_multi_level'])  # True
        >>> print(info['header_levels'])  # 2
        >>> print(info['column_headers'])
        # ['Q1 - Actual', 'Q1 - Budget', 'Q2 - Actual', 'Q2 - Budget']
    """

    def __init__(self, separator: str = " - ") -> None:
        """Initialize header parser with configurable separator.

        Args:
            separator (str): String to use when joining multi-level header parts.
                            Default: " - " produces "Q1 - Actual" style headers.
                            Use " | " for "Q1 | Actual" or "/" for "Q1/Actual".

        Example:
            >>> parser_dash = HeaderParser(separator=' - ')
            >>> parser_pipe = HeaderParser(separator=' | ')
            >>> parser_slash = HeaderParser(separator='/')
        """
        self.separator = separator

    def parse(self, df: pd.DataFrame) -> Dict[str, any]:
        """Parse comprehensive header information from DataFrame.

        Extracts all column and index header metadata including level count,
        hierarchy structure, and flattened representations.

        Args:
            df (pd.DataFrame): DataFrame to analyze

        Returns:
            Dict[str, any]: Comprehensive header information:
                - 'has_multi_level': bool - Whether columns are MultiIndex
                - 'column_headers': List[str] - Flattened column names
                - 'header_levels': int - Number of header levels (1 or more)
                - 'row_index_name': Optional[str] - Name of DataFrame index
                - 'header_structure': List[Dict] - Hierarchy info (if multi-level)

        Example:
            >>> df = pd.DataFrame({
            ...     'A': [1, 2],
            ...     'B': [3, 4]
            ... }, index=pd.Index(['Row1', 'Row2'], name='Rows'))
            >>>
            >>> parser = HeaderParser()
            >>> info = parser.parse(df)
            >>> print(info['column_headers'])  # ['A', 'B']
            >>> print(info['row_index_name'])  # 'Rows'
            >>> print(info['has_multi_level'])  # False
        """
        result: Dict[str, any] = {
            "has_multi_level": isinstance(df.columns, pd.MultiIndex),
            "column_headers": [],
            "header_levels": 1,
            "row_index_name": None,
            "header_structure": [],
        }

        if isinstance(df.columns, pd.MultiIndex):
            result["header_levels"] = df.columns.nlevels
            result["column_headers"] = self.flatten_multi_level_headers(df.columns)
            result["header_structure"] = self.extract_header_hierarchy(df.columns)
        else:
            result["column_headers"] = [str(col) for col in df.columns]

        # Check for row index
        if df.index.name:
            result["row_index_name"] = str(df.index.name)

        return result

    def flatten_multi_level_headers(
        self,
        columns: pd.MultiIndex
    ) -> List[str]:
        """Flatten multi-level column headers to single-level strings.

        Args:
            columns: MultiIndex columns from DataFrame

        Returns:
            List of flattened header strings

        Example:
            Input: [('Q1 2024', 'Actual'), ('Q1 2024', 'Budget')]
            Output: ['Q1 2024 - Actual', 'Q1 2024 - Budget']
        """
        flattened: List[str] = []

        for col_tuple in columns:
            # Filter out empty/None parts
            parts = [str(part).strip() for part in col_tuple if str(part).strip() and str(part) != ""]

            if parts:
                flattened.append(self.separator.join(parts))
            else:
                flattened.append("Unnamed")

        return flattened

    def extract_header_hierarchy(
        self,
        columns: pd.MultiIndex
    ) -> List[Dict[str, any]]:
        """Extract hierarchical structure from multi-level headers.

        Args:
            columns: MultiIndex columns from DataFrame

        Returns:
            List of dictionaries describing header structure

        Example:
            [
                {"level": 0, "name": "Q1 2024", "children": ["Actual", "Budget"]},
                {"level": 0, "name": "Q2 2024", "children": ["Actual", "Budget"]}
            ]
        """
        if not isinstance(columns, pd.MultiIndex):
            return []

        hierarchy: List[Dict[str, any]] = []
        seen_parents: Dict[str, Dict[str, any]] = {}

        for col_tuple in columns:
            # Get top-level parent
            parent_name = str(col_tuple[0]).strip()

            if parent_name not in seen_parents:
                parent_entry = {
                    "level": 0,
                    "name": parent_name,
                    "children": []
                }
                seen_parents[parent_name] = parent_entry
                hierarchy.append(parent_entry)

            # Add children if multi-level
            if len(col_tuple) > 1:
                for child_part in col_tuple[1:]:
                    child_name = str(child_part).strip()
                    if child_name and child_name not in seen_parents[parent_name]["children"]:
                        seen_parents[parent_name]["children"].append(child_name)

        return hierarchy

    def detect_merged_cells(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[int, int, str]]:
        """Detect potential merged cells by analyzing repeated values.

        Args:
            df: DataFrame to analyze

        Returns:
            List of (row, col, value) tuples indicating potential merges
        """
        merged_cells: List[Tuple[int, int, str]] = []

        for col_idx, col_name in enumerate(df.columns):
            prev_value: Optional[str] = None
            repeat_count = 0

            for row_idx, value in enumerate(df[col_name]):
                str_value = str(value).strip()

                if str_value == prev_value and str_value:
                    repeat_count += 1
                    if repeat_count >= 2:  # Threshold for considering as merged
                        merged_cells.append((row_idx, col_idx, str_value))
                else:
                    prev_value = str_value
                    repeat_count = 0

        return merged_cells

    def get_column_names(
        self,
        df: pd.DataFrame,
        include_index: bool = False
    ) -> List[str]:
        """Get all column names including optional index.

        Args:
            df: DataFrame to get columns from
            include_index: Whether to include index name as first column

        Returns:
            List of column names
        """
        columns: List[str] = []

        if include_index and df.index.name:
            columns.append(str(df.index.name))

        if isinstance(df.columns, pd.MultiIndex):
            columns.extend(self.flatten_multi_level_headers(df.columns))
        else:
            columns.extend([str(col) for col in df.columns])

        return columns
