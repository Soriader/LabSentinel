from __future__ import annotations

from typing import Any

import pandas as pd


SOFT_ERROR_TYPES = {"contextual_shift", "near_boundary_anomaly"}


def build_ml_evaluation(ml_scored_df: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate ML anomaly detection performance on QC-passed records.

    Assumes ml_scored_df contains only rows that passed QC and were scored by ML.
    """
    required_columns = {
        "is_ml_anomaly",
        "is_injected_error",
        "error_type",
    }
    missing = required_columns - set(ml_scored_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for ML evaluation: {missing_str}")

    total_scored_rows = len(ml_scored_df)
    total_ml_alerts = int((ml_scored_df["is_ml_anomaly"] == True).sum())  # noqa: E712

    injected_scored_df = ml_scored_df[ml_scored_df["is_injected_error"] == True].copy()  # noqa: E712
    total_injected_in_scored = len(injected_scored_df)

    detected_injected = int((injected_scored_df["is_ml_anomaly"] == True).sum())  # noqa: E712
    missed_injected = int((injected_scored_df["is_ml_anomaly"] == False).sum())  # noqa: E712

    ml_alerts_df = ml_scored_df[ml_scored_df["is_ml_anomaly"] == True].copy()  # noqa: E712
    injected_in_alerts = int((ml_alerts_df["is_injected_error"] == True).sum())  # noqa: E712

    soft_df = injected_scored_df[injected_scored_df["error_type"].isin(SOFT_ERROR_TYPES)].copy()
    total_soft = len(soft_df)
    detected_soft = int((soft_df["is_ml_anomaly"] == True).sum())  # noqa: E712
    missed_soft = int((soft_df["is_ml_anomaly"] == False).sum())  # noqa: E712

    evaluation: dict[str, Any] = {
        "total_scored_rows": total_scored_rows,
        "total_ml_alerts": total_ml_alerts,
        "total_injected_errors_in_scored_rows": total_injected_in_scored,
        "detected_injected_errors": detected_injected,
        "missed_injected_errors": missed_injected,
        "ml_recall_on_injected_scored_rows": round(
            detected_injected / total_injected_in_scored, 4
        ) if total_injected_in_scored > 0 else 0.0,
        "ml_precision_on_alerts": round(
            injected_in_alerts / total_ml_alerts, 4
        ) if total_ml_alerts > 0 else 0.0,
        "soft_anomalies": {
            "total": total_soft,
            "detected": detected_soft,
            "missed": missed_soft,
            "recall": round(detected_soft / total_soft, 4) if total_soft > 0 else 0.0,
        },
        "by_error_type": {},
    }

    for error_type, group in injected_scored_df.groupby("error_type"):
        total = len(group)
        detected = int((group["is_ml_anomaly"] == True).sum())  # noqa: E712
        missed = int((group["is_ml_anomaly"] == False).sum())  # noqa: E712

        evaluation["by_error_type"][error_type] = {
            "total": total,
            "detected": detected,
            "missed": missed,
            "recall": round(detected / total, 4) if total > 0 else 0.0,
        }

    return evaluation