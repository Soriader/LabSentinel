from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def run_isolation_forest(
    source_df: pd.DataFrame,
    features_df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train Isolation Forest and score the provided records.

    Returns a copy of source_df with:
    - anomaly_score
    - is_ml_anomaly
    """
    if source_df.empty:
        raise ValueError("source_df is empty.")
    if features_df.empty:
        raise ValueError("features_df is empty.")
    if len(source_df) != len(features_df):
        raise ValueError("source_df and features_df must have the same number of rows.")

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(features_df)

    scored_df = source_df.copy()
    scored_df["anomaly_score"] = model.score_samples(features_df)
    predictions = model.predict(features_df)

    scored_df["is_ml_anomaly"] = (predictions == -1)
    scored_df["is_ml_anomaly"] = scored_df["is_ml_anomaly"].astype("boolean")

    return scored_df


def build_ml_alerts(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only records flagged as anomalies by the ML model.
    """
    required_columns = {"anomaly_score", "is_ml_anomaly"}
    missing = required_columns - set(scored_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for ML alerts: {missing_str}")

    alerts_df = scored_df[scored_df["is_ml_anomaly"] == True].copy()  # noqa: E712

    if alerts_df.empty:
        return alerts_df

    preferred_columns = [
        "sample_id",
        "product",
        "parameter",
        "value",
        "value_num",
        "unit",
        "date",
        "anomaly_score",
        "is_ml_anomaly",
        "is_injected_error",
        "error_type",
    ]

    existing_columns = [col for col in preferred_columns if col in alerts_df.columns]
    remaining_columns = [col for col in alerts_df.columns if col not in existing_columns]

    alerts_df = alerts_df[existing_columns + remaining_columns]
    alerts_df = alerts_df.sort_values(by="anomaly_score", ascending=True).reset_index(drop=True)

    return alerts_df