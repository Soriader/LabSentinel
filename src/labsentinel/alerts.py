from __future__ import annotations

import pandas as pd


def build_qc_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows that failed at least one QC check.
    """
    if "qc_passed" not in df.columns:
        raise ValueError("Column 'qc_passed' not found. Run build_qc_flags() first.")

    alerts = df[df["qc_passed"] != True].copy()

    if alerts.empty:
        return alerts

    preferred_columns = [
        "sample_id",
        "product",
        "parameter",
        "value",
        "value_num",
        "unit",
        "expected_unit",
        "date",
        "date_dt",
        "unit_ok",
        "row_complete_ok",
        "date_ok",
        "value_in_range_ok",
        "qc_status",
        "qc_passed",
    ]

    existing_columns = [col for col in preferred_columns if col in alerts.columns]
    remaining_columns = [col for col in alerts.columns if col not in existing_columns]

    return alerts[existing_columns + remaining_columns]