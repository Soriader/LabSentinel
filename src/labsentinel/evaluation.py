from __future__ import annotations

from typing import Any

import pandas as pd


def build_qc_evaluation(df: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate baseline QC against injected errors from the synthetic generator.
    """
    required_columns = {
        "is_injected_error",
        "error_type",
        "qc_passed",
    }
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for QC evaluation: {missing_str}")

    injected_df = df[df["is_injected_error"] == True].copy()  # noqa: E712

    total_injected = len(injected_df)
    detected_injected = int((injected_df["qc_passed"] == False).sum())  # noqa: E712
    missed_injected = int((injected_df["qc_passed"] == True).sum())  # noqa: E712

    evaluation: dict[str, Any] = {
        "total_injected_errors": total_injected,
        "detected_injected_errors": detected_injected,
        "missed_injected_errors": missed_injected,
        "qc_recall_on_injected_errors": round(
            detected_injected / total_injected, 4
        ) if total_injected > 0 else 0.0,
        "by_error_type": {},
    }

    for error_type, group in injected_df.groupby("error_type"):
        total = len(group)
        detected = int((group["qc_passed"] == False).sum())  # noqa: E712
        missed = int((group["qc_passed"] == True).sum())  # noqa: E712

        evaluation["by_error_type"][error_type] = {
            "total": total,
            "detected": detected,
            "missed": missed,
            "recall": round(detected / total, 4) if total > 0 else 0.0,
        }

    return evaluation