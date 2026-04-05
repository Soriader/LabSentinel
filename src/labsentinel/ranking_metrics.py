from __future__ import annotations

from typing import Any

import pandas as pd


def compute_precision_at_k(
    df: pd.DataFrame,
    k: int,
    score_column: str,
    label_column: str = "is_injected_error",
    ascending: bool = True,
) -> dict[str, Any]:
    """
    Compute precision@k for a ranked alerts DataFrame.

    Parameters:
    - df: input DataFrame
    - k: number of top rows to evaluate
    - score_column: column used for sorting alerts
    - label_column: ground-truth label column
    - ascending: whether lower score means more anomalous
    """
    required_columns = {score_column, label_column}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for precision@k: {missing_str}")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if df.empty:
        return {
            "k": k,
            "evaluated_rows": 0,
            "true_positives_in_top_k": 0,
            "precision_at_k": 0.0,
        }

    ranked_df = df.sort_values(by=score_column, ascending=ascending).head(k).copy()

    evaluated_rows = len(ranked_df)
    true_positives = int((ranked_df[label_column] == True).sum())  # noqa: E712

    precision = round(true_positives / evaluated_rows, 4) if evaluated_rows > 0 else 0.0

    return {
        "k": k,
        "evaluated_rows": evaluated_rows,
        "true_positives_in_top_k": true_positives,
        "precision_at_k": precision,
    }


def build_ranking_metrics(
    ml_alerts_df: pd.DataFrame,
    hybrid_alerts_df: pd.DataFrame,
    k: int,
) -> dict[str, Any]:
    """
    Build ranking-based metrics for ML and Hybrid alerts.

    Assumptions:
    - ML alerts are ranked by anomaly_score ascending
    - Hybrid alerts use anomaly_score when available; QC-only alerts may not have score
    """
    metrics: dict[str, Any] = {
        "k": k,
        "ml": None,
        "hybrid": None,
    }

    if not ml_alerts_df.empty and "anomaly_score" in ml_alerts_df.columns:
        metrics["ml"] = compute_precision_at_k(
            df=ml_alerts_df,
            k=k,
            score_column="anomaly_score",
            label_column="is_injected_error",
            ascending=True,
        )
    else:
        metrics["ml"] = {
            "k": k,
            "evaluated_rows": 0,
            "true_positives_in_top_k": 0,
            "precision_at_k": 0.0,
        }

    hybrid_rankable_df = hybrid_alerts_df.copy()

    if "anomaly_score" in hybrid_rankable_df.columns:
        # Push QC-only alerts (missing anomaly_score) to the end of the ranking
        hybrid_rankable_df["ranking_score"] = hybrid_rankable_df["anomaly_score"].fillna(float("inf"))

        metrics["hybrid"] = compute_precision_at_k(
            df=hybrid_rankable_df,
            k=k,
            score_column="ranking_score",
            label_column="is_injected_error",
            ascending=True,
        )
    else:
        metrics["hybrid"] = {
            "k": k,
            "evaluated_rows": 0,
            "true_positives_in_top_k": 0,
            "precision_at_k": 0.0,
        }

    return metrics