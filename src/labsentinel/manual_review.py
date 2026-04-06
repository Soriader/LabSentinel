from __future__ import annotations

import pandas as pd


FINAL_TEMPLATE_COLUMNS = [
    "run",
    "sample_id",
    "product",
    "parameter",
    "unit",
    "expected_unit",
    "value_num",
    "date",
    "alert_source",
    "anomaly_score",
    "qc_status",
    "is_injected_error",
    "error_type",
    "validator_label",
    "validator_notes",
]


def build_manual_labels_template(
    hybrid_alerts_df: pd.DataFrame,
    run_id: str,
    k: int,
) -> pd.DataFrame:
    """
    Build a compact manual review template from hybrid alerts.

    The template is intended for human validation, so it keeps only
    the most relevant columns and appends empty annotation fields.
    """
    if hybrid_alerts_df.empty:
        return pd.DataFrame(columns=FINAL_TEMPLATE_COLUMNS)

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

    if "date" not in template_df.columns:
        template_df["date"] = pd.NA

    if "is_injected_error" not in template_df.columns:
        template_df["is_injected_error"] = pd.NA

    if "error_type" not in template_df.columns:
        template_df["error_type"] = pd.NA

    template_df["validator_label"] = ""
    template_df["validator_notes"] = ""

    for col in FINAL_TEMPLATE_COLUMNS:
        if col not in template_df.columns:
            template_df[col] = pd.NA

    return template_df[FINAL_TEMPLATE_COLUMNS].reset_index(drop=True)