from __future__ import annotations

from typing import Any

import pandas as pd


def build_qc_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a compact QC summary for a processed dataset.
    """
    required_columns = {
        "qc_passed",
        "unit_ok",
        "row_complete_ok",
        "date_ok",
        "value_in_range_ok",
    }
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for summary: {missing_str}")

    total_rows = len(df)
    passed_rows = int((df["qc_passed"] == True).sum())  # noqa: E712
    failed_rows = int((df["qc_passed"] != True).sum())  # noqa: E712

    summary = {
        "total_rows": total_rows,
        "qc_passed_rows": passed_rows,
        "qc_failed_rows": failed_rows,
        "pass_rate": round(passed_rows / total_rows, 4) if total_rows > 0 else 0.0,
        "fail_rate": round(failed_rows / total_rows, 4) if total_rows > 0 else 0.0,
        "failures_by_rule": {
            "unit_failures": int((df["unit_ok"] == False).sum()),  # noqa: E712
            "completeness_failures": int((df["row_complete_ok"] == False).sum()),  # noqa: E712
            "date_failures": int((df["date_ok"] == False).sum()),  # noqa: E712
            "range_failures": int((df["value_in_range_ok"] == False).sum()),  # noqa: E712
        },
    }

    return summary