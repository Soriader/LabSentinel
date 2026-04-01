from __future__ import annotations

import pandas as pd


def prepare_ml_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a feature matrix for anomaly detection.

    Returns:
    - filtered_df: original rows used for ML
    - features_df: numeric feature matrix ready for modeling
    """
    required_columns = {
        "sample_id",
        "product",
        "parameter",
        "value_num",
        "qc_passed",
    }
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for ML features: {missing_str}")

    filtered_df = df[
        (df["qc_passed"] == True) &  # noqa: E712
        (df["value_num"].notna())
    ].copy()

    if filtered_df.empty:
        raise ValueError("No valid rows available for ML after QC filtering.")

    base_features = filtered_df[["value_num", "product", "parameter"]].copy()

    encoded_features = pd.get_dummies(
        base_features,
        columns=["product", "parameter"],
        dtype=float,
    )

    encoded_features["value_num"] = encoded_features["value_num"].astype(float)

    return filtered_df, encoded_features