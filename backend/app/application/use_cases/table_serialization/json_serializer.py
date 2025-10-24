"""JSON serializer for table conversion (Task 2.1.2).

Converts pandas DataFrames to structured JSON format suitable for:
- Precise queries and filtering in RAG systems
- Structured data storage and archival
- API responses with rich metadata
- Metadata-rich representations preserving table semantics

The JSONSerializer produces fully-validated JSON with:
- Data types preserved for each column
- Multi-level header hierarchy for complex tables
- Complete metadata (source, statistics, data hashing)
- Nullable value handling (NaN → JSON null)
- JSON-compatible numeric type conversion

Example:
    >>> import pandas as pd
    >>> from json_serializer import JSONSerializer
    >>>
    >>> df = pd.DataFrame({
    ...     'Product': ['Widget A', 'Widget B', 'Gadget X'],
    ...     'Sales ($K)': [100.5, 250.0, None],
    ...     'Region': ['US', 'EU', 'APAC']
    ... })
    >>>
    >>> serializer = JSONSerializer()
    >>> json_data = serializer.serialize(
    ...     df,
    ...     metadata={
    ...         'file_name': 'sales_report.pdf',
    ...         'caption': 'Q4 Product Sales'
    ...     }
    ... )
    >>>
    >>> import json
    >>> print(json.dumps(json_data, indent=2))
    {
      "data": [
        {"Product": "Widget A", "Sales ($K)": 100.5, "Region": "US"},
        {"Product": "Widget B", "Sales ($K)": 250.0, "Region": "EU"},
        {"Product": "Gadget X", "Sales ($K)": null, "Region": "APAC"}
      ],
      "structure": {
        "columns": ["Product", "Sales ($K)", "Region"],
        "multi_level_headers": false
      },
      "metadata": {
        "row_count": 3,
        "column_count": 3,
        "file_name": "sales_report.pdf"
      },
      "data_types": {
        "Product": "string",
        "Sales ($K)": "float",
        "Region": "string"
      }
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

import pandas as pd
import numpy as np

from .base_serializer import BaseSerializer, SerializationError, ValidationError
from .config import JSON_CONFIG, VALIDATION_CONFIG
from .header_parser import HeaderParser
from .cell_processor import CellProcessor
from .metadata_extractor import MetadataExtractor


class JSONSerializer(BaseSerializer):
    """Serializer for converting DataFrames to structured JSON.

    Transforms pandas DataFrames into fully-typed JSON representations optimized for
    querying, filtering, and programmatic access. Preserves table structure and semantics
    while converting to JSON-compatible formats.

    Features:
    - Automatic data type detection and preservation
    - Multi-level header structure capture
    - Complete metadata inclusion
    - Queryable format with proper JSON typing
    - Smart conversion of pandas types (NaN→null, int64→int, etc.)
    - Supports both row-oriented and structured formats

    Attributes:
        header_parser (HeaderParser): Extracts and analyzes column structure
        cell_processor (CellProcessor): Processes cell values for JSON
        metadata_extractor (MetadataExtractor): Collects table statistics

    Example:
        >>> import pandas as pd
        >>> from json_serializer import JSONSerializer
        >>>
        >>> # Multi-level financial data
        >>> df = pd.DataFrame(
        ...     [[125.5, 120.0, 138.7], [65.2, 62.0, 70.1]],
        ...     columns=pd.MultiIndex.from_tuples(
        ...         [('Q1', 'Actual'), ('Q1', 'Budget'), ('Q2', 'Actual')],
        ...         names=['Quarter', 'Type']
        ...     ),
        ...     index=['Revenue ($M)', 'Expenses ($M)']
        ... )
        >>>
        >>> serializer = JSONSerializer()
        >>> json_data = serializer.serialize(df)
        >>> # Returns dict with 'data', 'structure', 'metadata', 'data_types'
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize JSON serializer with optional configuration.

        Args:
            config (dict, optional): Configuration overrides. Common options:
                - data_orientation: 'records', 'split', 'index' (default: 'records')
                - include_hash: Include content hash (default: True)
                - include_statistics: Include descriptive statistics (default: False)
                - multi_level_separator: Separator for flattening headers (default: " - ")
                - max_items_in_statistics: Max items for statistical analysis (default: 1000)

        Example:
            >>> config = {
            ...     'data_orientation': 'split',
            ...     'include_statistics': True,
            ...     'multi_level_separator': ' | '
            ... }
            >>> serializer = JSONSerializer(config)
        """
        # Merge default config with overrides
        full_config = {**JSON_CONFIG, **VALIDATION_CONFIG}
        if config:
            full_config.update(config)

        super().__init__(full_config)

        # Initialize specialized components
        separator = full_config.get("multi_level_separator", " - ")
        self.header_parser = HeaderParser(separator=separator)

        self.cell_processor = CellProcessor()

        self.metadata_extractor = MetadataExtractor(
            include_hash=full_config.get("include_hash", True),
            include_statistics=full_config.get("include_statistics", False)
        )

    def serialize(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convert DataFrame to structured JSON format.

        Produces a JSON-compatible dictionary with complete table information including
        data, structure, metadata, and type information. Suitable for:
        - Storage in databases and file systems
        - API responses
        - RAG query execution and filtering
        - Cross-system data exchange

        Args:
            df (pd.DataFrame): DataFrame to serialize
            metadata (dict, optional): Source metadata with keys:
                - 'file_name': Source file name
                - 'page': Page number in source document
                - 'caption': Table title
                - 'source': Full source reference
                - 'timestamp': Extraction timestamp

        Returns:
            Dict[str, Any]: Dictionary structure:
                {
                    'data': [rows as dicts],
                    'structure': {
                        'columns': [col names],
                        'multi_level_headers': bool,
                        'header_hierarchy': [hierarchy info if multi-level],
                        'header_levels': int
                    },
                    'metadata': {
                        'row_count': int,
                        'column_count': int,
                        'source': str,
                        ...
                    },
                    'data_types': {
                        'column_name': 'type',
                        ...
                    }
                }

        Raises:
            SerializationError: If conversion fails during processing
            ValidationError: If DataFrame doesn't meet validation requirements

        Example:
            >>> import pandas as pd
            >>> from json_serializer import JSONSerializer
            >>>
            >>> # Chemistry lab results with multi-level headers
            >>> df = pd.DataFrame({
            ...     'Compound': ['Ethanol', 'Acetone', 'Benzene'],
            ...     'Mol. Weight': [46.07, 58.08, 78.11],
            ...     'BP (°C)': [78.4, 56.0, 80.1],
            ...     'Density (g/cm³)': [0.789, 0.784, 0.876]
            ... })
            >>>
            >>> serializer = JSONSerializer()
            >>> json_data = serializer.serialize(
            ...     df,
            ...     metadata={
            ...         'file_name': 'organic_compounds.pdf',
            ...         'caption': 'Physical Properties of Organic Solvents',
            ...         'page': 42
            ...     }
            ... )
            >>>
            >>> # Access results
            >>> print(f"Rows: {json_data['metadata']['row_count']}")
            >>> print(f"Data types: {json_data['data_types']}")
            >>> for row in json_data['data']:
            ...     print(f"{row['Compound']}: BP={row['BP (°C)']}°C")
        """
        try:
            # Validate structure
            self.validate_structure(df)

            # Parse headers BEFORE preprocessing (to preserve MultiIndex)
            header_info = self.header_parser.parse(df)

            # Preprocess DataFrame
            df_processed = self.preprocess_dataframe(df)

            # Extract metadata
            extracted_metadata = self.metadata_extractor.extract(df_processed, metadata)

            # Build JSON structure
            result: Dict[str, Any] = {}

            # Add table identifier if available
            if metadata and "table_id" in metadata:
                result["table_id"] = metadata["table_id"]

            # Add source metadata
            if metadata:
                result["source"] = self._format_source_metadata(metadata)

            # Add structure information
            if self.config.get("include_structure", True):
                result["structure"] = self._build_structure_info(header_info)

            # Add data
            result["data"] = self._convert_data(df_processed, header_info)

            # Add metadata
            result["metadata"] = extracted_metadata.get("structure", {})

            # Add data types
            if self.config.get("preserve_dtypes", True):
                result["data_types"] = extracted_metadata.get("data_types", {})

            return result

        except ValidationError:
            raise
        except Exception as exc:
            raise self.handle_error(exc, "JSON serialization failed")

    def _convert_data(
        self,
        df: pd.DataFrame,
        header_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert DataFrame to JSON-compatible data structure.

        Args:
            df: Preprocessed DataFrame
            header_info: Header information from parser

        Returns:
            List of row dictionaries
        """
        # Flatten multi-level headers if needed
        if header_info["has_multi_level"] and self.config.get("flatten_multi_index", True):
            df_flat = df.copy()
            df_flat.columns = header_info["column_headers"]
        else:
            df_flat = df

        # Convert to specified orientation
        orient = self.config.get("orient", "records")
        date_format = self.config.get("date_format", "iso")

        # Convert DataFrame to dict
        data_dict = df_flat.to_dict(orient=orient)

        # Handle special data types
        if orient == "records":
            return [self._clean_record(record) for record in data_dict]
        elif orient == "split":
            # Clean values in split format
            data_dict["data"] = [
                [self._clean_value(val) for val in row]
                for row in data_dict["data"]
            ]
            return data_dict
        else:
            return data_dict

    def _clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single record for JSON compatibility.

        Args:
            record: Dictionary representing a table row

        Returns:
            Cleaned record with JSON-compatible types
        """
        cleaned: Dict[str, Any] = {}

        for key, value in record.items():
            cleaned[str(key)] = self._clean_value(value)

        return cleaned

    def _clean_value(self, value: Any) -> Any:
        """Convert value to JSON-compatible type.

        Args:
            value: Original value

        Returns:
            JSON-compatible value
        """
        # Handle pandas NA/NaN
        if pd.isna(value):
            return None

        # Handle numpy types
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            # Check for inf/-inf
            if np.isinf(value):
                return None
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()

        # Handle pandas timestamps
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        # Handle datetime objects
        if hasattr(value, 'isoformat'):
            return value.isoformat()

        # Return as-is for strings, numbers, bools, None
        if isinstance(value, (str, int, float, bool, type(None))):
            return value

        # Convert everything else to string
        return str(value)

    def _build_structure_info(
        self,
        header_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build structure information section.

        Args:
            header_info: Header information from parser

        Returns:
            Dictionary with structural metadata
        """
        structure: Dict[str, Any] = {
            "columns": header_info["column_headers"],
            "header_levels": header_info["header_levels"],
        }

        # Add multi-level header details if present
        if header_info["has_multi_level"]:
            structure["multi_level_headers"] = True
            structure["header_hierarchy"] = header_info["header_structure"]

        # Add row index info if present
        if header_info["row_index_name"]:
            structure["row_index_name"] = header_info["row_index_name"]

        return structure

    def _format_source_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format source metadata for JSON output.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Formatted source metadata
        """
        source: Dict[str, Any] = {}

        if "file_name" in metadata:
            source["file"] = metadata["file_name"]

        if "page" in metadata:
            source["page"] = metadata["page"]

        if "caption" in metadata:
            source["caption"] = metadata["caption"]

        if "processor" in metadata:
            source["processor"] = metadata["processor"]

        # Handle nested source dict
        if "source" in metadata and isinstance(metadata["source"], dict):
            source.update(metadata["source"])

        return source

    def to_json_string(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        indent: int = 2
    ) -> str:
        """Serialize to JSON string (convenience method).

        Args:
            df: DataFrame to serialize
            metadata: Optional source metadata
            indent: JSON indentation level

        Returns:
            JSON string
        """
        result = self.serialize(df, metadata)
        return json.dumps(result, indent=indent, ensure_ascii=False)
