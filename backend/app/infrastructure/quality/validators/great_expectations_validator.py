"""Great Expectations adapter utilities.

These helpers are intentionally lightweight and optional. They return
None if GE isn't installed or issues occur.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def try_run_basic_suite(df: pd.DataFrame, min_words: int, min_chars: int) -> Optional[Dict[str, Any]]:
    try:
        import great_expectations as ge  # type: ignore

        ge_df = ge.from_pandas(df)  # type: ignore[attr-defined]
        results = []
        results.append(
            ge_df.expect_column_min_to_be_between(column="total_words", min_value=min_words).to_json_dict()
        )
        results.append(
            ge_df.expect_column_min_to_be_between(column="total_characters", min_value=min_chars).to_json_dict()
        )
        return {
            "success": all(r.get("success", False) for r in results if isinstance(r, dict)),
            "results": results,
        }
    except Exception:
        return None

