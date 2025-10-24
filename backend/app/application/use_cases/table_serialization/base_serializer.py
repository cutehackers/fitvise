"""Base serializer abstract class for table conversion (Task 2.1.2).

Provides abstract interface and shared functionality for converting pandas DataFrames
to different output formats (Markdown, JSON, etc.). Implements common validation,
preprocessing, and header extraction logic used by all concrete serializers.

This module defines:
- SerializationError: Exception raised during serialization failures
- ValidationError: Exception raised when DataFrame structure is invalid
- BaseSerializer: Abstract base class defining the serialization interface

Example:
    To create a custom serializer, inherit from BaseSerializer and implement serialize():

    >>> from base_serializer import BaseSerializer
    >>> import pandas as pd
    >>>
    >>> class CustomSerializer(BaseSerializer):
    ...     def serialize(self, df, metadata=None):
    ...         # Custom implementation
    ...         self.validate_structure(df)
    ...         processed = self.preprocess_dataframe(df)
    ...         return "custom output"
    >>>
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> serializer = CustomSerializer()
    >>> result = serializer.serialize(df)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class SerializationError(Exception):
    """Raised when table serialization fails.

    Indicates an error during conversion of a DataFrame to target format.
    Includes context about where the error occurred in the serialization process.

    Example:
        >>> try:
        ...     serializer.serialize(df)
        ... except SerializationError as e:
        ...     print(f"Serialization failed: {e}")
    """
    pass


class ValidationError(Exception):
    """Raised when table structure validation fails.

    Indicates that a DataFrame doesn't meet minimum requirements for serialization,
    such as:
    - Empty DataFrame (no rows or columns)
    - Invalid data types
    - Cells exceeding maximum length
    - Invalid metadata

    Example:
        >>> try:
        ...     serializer.validate_structure(df)
        ... except ValidationError as e:
        ...     print(f"Invalid DataFrame: {e}")
    """
    pass


class BaseSerializer(ABC):
    """Abstract base class for table serializers.

    Defines the interface that all concrete serializers must implement and provides
    common functionality for:
    - Validating DataFrame structure
    - Preprocessing data (whitespace cleaning, type conversion, etc.)
    - Extracting headers (including multi-level/hierarchical columns)
    - Error handling with context

    All concrete serializers (MarkdownSerializer, JSONSerializer, etc.) must inherit
    from this class and implement the serialize() method.

    Attributes:
        config (dict): Configuration dictionary for serializer behavior

    Example:
        >>> import pandas as pd
        >>> from markdown_serializer import MarkdownSerializer
        >>>
        >>> # Create a sample DataFrame with multi-level headers
        >>> df = pd.DataFrame(
        ...     [[100, 95], [120, 110]],
        ...     columns=pd.MultiIndex.from_tuples(
        ...         [('Q1 2024', 'Actual'), ('Q1 2024', 'Budget')],
        ...         names=['Quarter', 'Type']
        ...     ),
        ...     index=['Revenue', 'Expenses']
        ... )
        >>>
        >>> # Serialize to Markdown
        >>> serializer = MarkdownSerializer()
        >>> markdown = serializer.serialize(df, metadata={'source': 'financial_report.pdf'})
        >>> print(markdown)
        # Financial Table
        | Metric | Q1 2024 - Actual | Q1 2024 - Budget |
        |--------|------------------|-----------------|
        | Revenue | 100 | 95 |
        | Expenses | 120 | 110 |
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize serializer with optional configuration.

        Args:
            config: Serializer-specific configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def serialize(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """Convert DataFrame to target format.

        Args:
            df: pandas DataFrame to serialize
            metadata: Optional metadata (source file, page, caption, etc.)

        Returns:
            Serialized representation (string for Markdown, dict for JSON)

        Raises:
            SerializationError: If conversion fails
            ValidationError: If DataFrame structure is invalid
        """
        pass

    def validate_structure(self, df: pd.DataFrame) -> None:
        """Validate DataFrame structure before serialization.

        Checks that the DataFrame meets minimum requirements:
        - Is a valid pandas DataFrame (not None, not another type)
        - Has at least min_rows and min_columns (configurable)
        - No cells exceed max_cell_length (configurable)

        Args:
            df (pd.DataFrame): DataFrame to validate

        Raises:
            ValidationError: If structure is invalid with descriptive message

        Example:
            >>> import pandas as pd
            >>> from base_serializer import BaseSerializer
            >>>
            >>> serializer = BaseSerializer()
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> serializer.validate_structure(df)  # Passes validation
            >>>
            >>> empty_df = pd.DataFrame()
            >>> serializer.validate_structure(empty_df)  # Raises ValidationError
            ValidationError: DataFrame is empty
        """
        if df is None:
            raise ValidationError("DataFrame cannot be None")

        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"Expected pandas DataFrame, got {type(df)}")

        if df.empty:
            raise ValidationError("DataFrame is empty")

        # Check minimum dimensions
        min_rows = self.config.get("min_rows", 1)
        min_cols = self.config.get("min_columns", 1)

        if len(df) < min_rows:
            raise ValidationError(f"DataFrame has {len(df)} rows, minimum is {min_rows}")

        if len(df.columns) < min_cols:
            raise ValidationError(f"DataFrame has {len(df.columns)} columns, minimum is {min_cols}")

        # Check for extremely large cells
        max_cell_length = self.config.get("max_cell_length", 10000)
        for col in df.columns:
            if df[col].dtype == "object":  # Text columns
                max_len = df[col].astype(str).str.len().max()
                if max_len > max_cell_length:
                    raise ValidationError(
                        f"Column '{col}' has cell exceeding {max_cell_length} characters"
                    )

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame before serialization.

        Applies common preprocessing steps:
        - Cleans column names (removes leading/trailing whitespace)
        - Preserves multi-level column indices
        - Strips whitespace from string columns while preserving NaN values
        - Does NOT modify NaN values (handled appropriately by each serializer)

        Args:
            df (pd.DataFrame): Original DataFrame

        Returns:
            pd.DataFrame: Preprocessed DataFrame (original unchanged)

        Note:
            Creates a copy of the DataFrame to avoid modifying the original.

        Example:
            >>> import pandas as pd
            >>>
            >>> df = pd.DataFrame({
            ...     'Name  ': ['  Alice', 'Bob  '],  # Extra whitespace
            ...     'Score': [95.5, None],  # Includes NaN
            ... })
            >>>
            >>> preprocessed = serializer.preprocess_dataframe(df)
            >>> # Column name 'Name  ' becomes 'Name'
            >>> # '  Alice' becomes 'Alice'
            >>> # None (NaN) is preserved as-is
        """
        # Create a copy to avoid modifying original
        df_copy = df.copy()

        # Clean column names (remove leading/trailing whitespace)
        # Preserve MultiIndex structure - only clean simple column names
        if not isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = df_copy.columns.map(lambda x: str(x).strip() if x else "")

        # Note: We keep NaN values as-is during preprocessing
        # They will be handled appropriately by each serializer
        # (empty string for markdown, null for JSON)

        # Strip whitespace from string columns (but preserve NaN)
        for col in df_copy.select_dtypes(include=["object"]).columns:
            # Only strip non-null values
            df_copy[col] = df_copy[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

        return df_copy

    def extract_headers(self, df: pd.DataFrame) -> List[str]:
        """Extract column headers from DataFrame.

        Handles both simple and multi-level (MultiIndex) column indices.
        Multi-level headers are flattened using a configurable separator.

        Args:
            df (pd.DataFrame): DataFrame to extract headers from

        Returns:
            List[str]: Flattened header strings

        Example:
            >>> import pandas as pd
            >>>
            >>> # Simple headers
            >>> df_simple = pd.DataFrame([[1, 2]], columns=['A', 'B'])
            >>> headers = serializer.extract_headers(df_simple)
            >>> print(headers)
            ['A', 'B']
            >>>
            >>> # Multi-level headers (common in financial data)
            >>> df_multi = pd.DataFrame(
            ...     [[100, 95]],
            ...     columns=pd.MultiIndex.from_tuples(
            ...         [('Q1 2024', 'Actual'), ('Q1 2024', 'Budget')]
            ...     )
            ... )
            >>> headers = serializer.extract_headers(df_multi)
            >>> print(headers)
            ['Q1 2024 - Actual', 'Q1 2024 - Budget']
        """
        if isinstance(df.columns, pd.MultiIndex):
            # Multi-level headers: flatten with separator
            separator = self.config.get("multi_level_separator", " - ")
            headers = [
                separator.join(str(part) for part in col_tuple if str(part).strip())
                for col_tuple in df.columns
            ]
        else:
            # Simple headers
            headers = [str(col) for col in df.columns]

        return headers

    def handle_error(self, error: Exception, context: str) -> SerializationError:
        """Convert generic exceptions to SerializationError with context.

        Args:
            error: Original exception
            context: Descriptive context of where error occurred

        Returns:
            SerializationError with full context
        """
        return SerializationError(f"{context}: {type(error).__name__}: {error}")
