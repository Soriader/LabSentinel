from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from labsentinel.features import prepare_ml_dataset
from labsentinel.ml_model import build_ml_alerts, run_isolation_forest


ALERT_KEY_COLUMNS = ["sample_id", "parameter", "date"]


def _build_alert_key_set(alerts_df: pd.DataFrame, k: int) -> set[tuple[str, str, str]]:
    if alerts_df.empty:
        return set()

    ranked_df = alerts_df.copy()

    if "anomaly_score" in ranked_df.columns:
        ranked_df = ranked_df.sort_values(by="anomaly_score", ascending=True).head(k).copy()
    else:
        ranked_df = ranked_df.head(k).copy()

    key_set: set[tuple[str, str, str]] = set()

    for _, row in ranked_df.iterrows():
        key = tuple(str(row[col]) for col in ALERT_KEY_COLUMNS)
        key_set.add(key)

    return key_set


def _jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    if not set_a and not set_b:
        return 1.0

    union = set_a | set_b
    intersection = set_a & set_b

    if not union:
        return 0.0

    return round(len(intersection) / len(union), 4)


def run_stability_analysis(
    df: pd.DataFrame,
    contamination: float = 0.05,
    k: int = 50,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ML stability analysis across multiple random seeds.

    Returns:
    - pairwise Jaccard similarity DataFrame
    - per-seed summary DataFrame
    """
    if seeds is None:
        seeds = [42, 123, 999]

    ml_source_df, ml_features_df = prepare_ml_dataset(df)

    seed_to_keyset: dict[int, set[tuple[str, str, str]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for seed in seeds:
        ml_scored_df = run_isolation_forest(
            source_df=ml_source_df,
            features_df=ml_features_df,
            contamination=contamination,
            random_state=seed,
        )

        ml_alerts_df = build_ml_alerts(ml_scored_df)
        key_set = _build_alert_key_set(ml_alerts_df, k=k)

        seed_to_keyset[seed] = key_set

        summary_rows.append(
            {
                "seed": seed,
                "top_k": k,
                "top_k_unique_alerts": len(key_set),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []

    for seed_a, seed_b in combinations(seeds, 2):
        set_a = seed_to_keyset[seed_a]
        set_b = seed_to_keyset[seed_b]

        pairwise_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "top_k": k,
                "intersection_size": len(set_a & set_b),
                "union_size": len(set_a | set_b),
                "jaccard_similarity": _jaccard_similarity(set_a, set_b),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(
        by=["seed_a", "seed_b"]
    ).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="seed").reset_index(drop=True)

    return pairwise_df, summary_df