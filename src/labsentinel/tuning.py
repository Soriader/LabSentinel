from __future__ import annotations

from typing import Any

import pandas as pd

from labsentinel.features import prepare_ml_dataset
from labsentinel.ml_evaluation import build_ml_evaluation
from labsentinel.ml_model import build_ml_alerts, run_isolation_forest
from labsentinel.ranking_metrics import build_ranking_metrics


def run_contamination_tuning(
    df: pd.DataFrame,
    k: int = 50,
    random_state: int = 42,
    contamination_values: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run Isolation Forest tuning for multiple contamination values.

    Returns a DataFrame with comparison metrics for each configuration.
    """
    if contamination_values is None:
        contamination_values = [0.03, 0.05, 0.08, 0.10]

    ml_source_df, ml_features_df = prepare_ml_dataset(df)

    results: list[dict[str, Any]] = []

    for contamination in contamination_values:
        ml_scored_df = run_isolation_forest(
            source_df=ml_source_df,
            features_df=ml_features_df,
            contamination=contamination,
            random_state=random_state,
        )

        ml_alerts_df = build_ml_alerts(ml_scored_df)

        if "anomaly_score" in ml_alerts_df.columns:
            ml_alerts_topk_df = (
                ml_alerts_df.sort_values(by="anomaly_score", ascending=True)
                .head(k)
                .copy()
            )
        else:
            ml_alerts_topk_df = ml_alerts_df.head(k).copy()

        ml_evaluation = build_ml_evaluation(ml_scored_df)
        ranking_metrics = build_ranking_metrics(
            ml_alerts_df=ml_alerts_topk_df,
            hybrid_alerts_df=ml_alerts_topk_df,
            k=k,
        )

        results.append(
            {
                "contamination": contamination,
                "total_ml_alerts": ml_evaluation["total_ml_alerts"],
                "detected_injected_errors": ml_evaluation["detected_injected_errors"],
                "missed_injected_errors": ml_evaluation["missed_injected_errors"],
                "ml_recall_on_injected_scored_rows": ml_evaluation["ml_recall_on_injected_scored_rows"],
                "ml_precision_on_alerts": ml_evaluation["ml_precision_on_alerts"],
                "soft_anomalies_recall": ml_evaluation["soft_anomalies"]["recall"],
                "precision_at_k": ranking_metrics["ml"]["precision_at_k"],
            }
        )

    return pd.DataFrame(results).sort_values(by="contamination").reset_index(drop=True)