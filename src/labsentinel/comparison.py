from __future__ import annotations

from typing import Any

import pandas as pd


ALL_ERROR_TYPES = [
    "bad_date",
    "missing_value",
    "out_of_range",
    "unit_mismatch",
    "near_boundary_anomaly",
    "contextual_shift",
]


def build_comparison_summary(
    full_df: pd.DataFrame,
    qc_summary: dict[str, Any],
    qc_evaluation: dict[str, Any],
    ml_evaluation: dict[str, Any],
    hybrid_alerts_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a final comparison summary for QC vs ML vs Hybrid.
    """
    required_columns = {
        "is_injected_error",
        "error_type",
    }
    missing = required_columns - set(full_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in full_df: {missing_str}")

    injected_df = full_df[full_df["is_injected_error"] == True].copy()  # noqa: E712
    total_injected_errors = len(injected_df)

    hybrid_detected_injected = int((hybrid_alerts_df["is_injected_error"] == True).sum())  # noqa: E712
    hybrid_missed_injected = total_injected_errors - hybrid_detected_injected

    hybrid_recall = (
        round(hybrid_detected_injected / total_injected_errors, 4)
        if total_injected_errors > 0 else 0.0
    )

    hybrid_precision = (
        round(hybrid_detected_injected / len(hybrid_alerts_df), 4)
        if len(hybrid_alerts_df) > 0 else 0.0
    )

    qc_precision = (
        round(
            qc_evaluation["detected_injected_errors"] / qc_summary["qc_failed_rows"],
            4,
        )
        if qc_summary["qc_failed_rows"] > 0 else 0.0
    )

    summary: dict[str, Any] = {
        "total_injected_errors": total_injected_errors,
        "qc": {
            "detected_injected_errors": qc_evaluation["detected_injected_errors"],
            "missed_injected_errors": qc_evaluation["missed_injected_errors"],
            "recall": qc_evaluation["qc_recall_on_injected_errors"],
            "precision": qc_precision,
            "total_alerts": qc_summary["qc_failed_rows"],
        },
        "ml": {
            "detected_injected_errors_in_scored_rows": ml_evaluation["detected_injected_errors"],
            "missed_injected_errors_in_scored_rows": ml_evaluation["missed_injected_errors"],
            "recall_on_scored_rows": ml_evaluation["ml_recall_on_injected_scored_rows"],
            "precision_on_alerts": ml_evaluation["ml_precision_on_alerts"],
            "total_alerts": ml_evaluation["total_ml_alerts"],
        },
        "hybrid": {
            "detected_injected_errors": hybrid_detected_injected,
            "missed_injected_errors": hybrid_missed_injected,
            "recall": hybrid_recall,
            "precision": hybrid_precision,
            "total_alerts": len(hybrid_alerts_df),
        },
        "by_error_type": {},
    }

    hybrid_injected_df = hybrid_alerts_df[hybrid_alerts_df["is_injected_error"] == True].copy()  # noqa: E712

    for error_type in ALL_ERROR_TYPES:
        total = int((injected_df["error_type"] == error_type).sum())

        qc_recall = None
        if error_type in qc_evaluation["by_error_type"]:
            qc_recall = qc_evaluation["by_error_type"][error_type]["recall"]

        ml_recall = None
        if error_type in ml_evaluation["by_error_type"]:
            ml_recall = ml_evaluation["by_error_type"][error_type]["recall"]

        hybrid_detected = int((hybrid_injected_df["error_type"] == error_type).sum())
        hybrid_error_recall = round(hybrid_detected / total, 4) if total > 0 else None

        summary["by_error_type"][error_type] = {
            "total": total,
            "qc_recall": qc_recall,
            "ml_recall": ml_recall,
            "hybrid_recall": hybrid_error_recall,
        }

    return summary