from __future__ import annotations

import pandas as pd


def build_manual_labels_template(
    hybrid_alerts_df: pd.DataFrame,
    run_id: str,
    k: int,
) -> pd.DataFrame:
    """
    Build a manual review template from hybrid alerts.

    The template contains top-K hybrid alerts and empty columns
    for human validation.
    """
    if hybrid_alerts_df.empty:
        columns = [
            "run",
            "sample_id",
            "product",
            "parameter",
            "unit",
            "expected_unit",
            "value_num",
            "alert_source",
            "anomaly_score",
            "qc_status",
            "is_injected_error",
            "error_type",
            "validator_label",
            "validator_notes",
        ]
        return pd.DataFrame(columns=columns)

    template_df = hybrid_alerts_df.copy()

    if "anomaly_score" in template_df.columns:
        template_df["manual_rank_score"] = template_df["anomaly_score"].fillna(float("inf"))
        template_df = template_df.sort_values(by="manual_rank_score", ascending=True).head(k).copy()
        template_df = template_df.drop(columns=["manual_rank_score"])
    else:
        template_df = template_df.head(k).copy()

    template_df.insert(0, "run", run_id)

    if "expected_unit" not in template_df.columns:
        template_df["expected_unit"] = pd.NA

    if "anomaly_score" not in template_df.columns:
        template_df["anomaly_score"] = pd.NA

    if "qc_status" not in template_df.columns:
        template_df["qc_status"] = pd.NA

    template_df["validator_label"] = ""
    template_df["validator_notes"] = ""

    preferred_columns = [
        "run",
        "sample_id",
        "product",
        "parameter",
        "unit",
        "expected_unit",
        "value_num",
        "alert_source",
        "anomaly_score",
        "qc_status",
        "is_injected_error",
        "error_type",
        "validator_label",
        "validator_notes",
    ]

    existing_columns = [col for col in preferred_columns if col in template_df.columns]
    remaining_columns = [col for col in template_df.columns if col not in existing_columns]

    return template_df[existing_columns + remaining_columns].reset_index(drop=True)