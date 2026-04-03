from __future__ import annotations

from typing import Any

import pandas as pd


DEDUP_COLUMNS = ["sample_id", "product", "parameter", "date"]


def build_hybrid_alerts(
    qc_alerts_df: pd.DataFrame,
    ml_alerts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a hybrid alert dataset by combining QC alerts and ML alerts.

    Rules:
    - all QC alerts are included
    - ML alerts are added only if they are not already present in QC alerts
    - output contains alert_source column: qc / ml / qc+ml
    """
    qc_df = qc_alerts_df.copy()
    ml_df = ml_alerts_df.copy()

    for col in DEDUP_COLUMNS:
        if col not in qc_df.columns:
            raise ValueError(f"Missing dedup column in qc_alerts_df: {col}")
        if col not in ml_df.columns:
            raise ValueError(f"Missing dedup column in ml_alerts_df: {col}")

    qc_df["alert_source"] = "qc"
    ml_df["alert_source"] = "ml"

    qc_keys = set(tuple(row) for row in qc_df[DEDUP_COLUMNS].itertuples(index=False, name=None))

    ml_df["is_already_in_qc"] = ml_df[DEDUP_COLUMNS].apply(tuple, axis=1).isin(qc_keys)

    ml_only_df = ml_df[ml_df["is_already_in_qc"] == False].copy()  # noqa: E712
    ml_only_df = ml_only_df.drop(columns=["is_already_in_qc"])

    hybrid_df = pd.concat([qc_df, ml_only_df], ignore_index=True, sort=False)

    preferred_columns = [
        "sample_id",
        "product",
        "parameter",
        "value",
        "value_num",
        "unit",
        "date",
        "alert_source",
        "qc_status",
        "qc_passed",
        "anomaly_score",
        "is_ml_anomaly",
        "is_injected_error",
        "error_type",
    ]

    existing_columns = [col for col in preferred_columns if col in hybrid_df.columns]
    remaining_columns = [col for col in hybrid_df.columns if col not in existing_columns]

    hybrid_df = hybrid_df[existing_columns + remaining_columns].copy()
    hybrid_df = hybrid_df.sort_values(
        by=["sample_id", "parameter", "date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return hybrid_df


def build_hybrid_evaluation(hybrid_df: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate the hybrid alert layer on injected synthetic errors.
    """
    required_columns = {
        "is_injected_error",
        "error_type",
    }
    missing = required_columns - set(hybrid_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for hybrid evaluation: {missing_str}")

    total_hybrid_alerts = len(hybrid_df)
    injected_detected = int((hybrid_df["is_injected_error"] == True).sum())  # noqa: E712

    evaluation: dict[str, Any] = {
        "total_hybrid_alerts": total_hybrid_alerts,
        "detected_injected_errors": injected_detected,
        "precision_on_hybrid_alerts": round(
            injected_detected / total_hybrid_alerts, 4
        ) if total_hybrid_alerts > 0 else 0.0,
        "by_error_type": {},
    }

    injected_df = hybrid_df[hybrid_df["is_injected_error"] == True].copy()  # noqa: E712

    for error_type, group in injected_df.groupby("error_type"):
        evaluation["by_error_type"][error_type] = {
            "detected": len(group),
        }

    return evaluation