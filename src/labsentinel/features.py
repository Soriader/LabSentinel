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
        "range_lower",
        "range_upper",
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

    # Relative position of the value within the allowed QC range
    range_width = filtered_df["range_upper"] - filtered_df["range_lower"]
    filtered_df["relative_position_in_range"] = (
        (filtered_df["value_num"] - filtered_df["range_lower"]) / range_width
    )

    # Avoid division issues in case of zero-width ranges
    filtered_df.loc[range_width == 0, "relative_position_in_range"] = 0.0

    # Parameter-level z-score to show how unusual a value is within its parameter group
    grouped_mean = filtered_df.groupby("parameter")["value_num"].transform("mean")
    grouped_std = filtered_df.groupby("parameter")["value_num"].transform("std")

    filtered_df["parameter_zscore"] = (
        (filtered_df["value_num"] - grouped_mean) / grouped_std
    )

    grouped_pp_mean = filtered_df.groupby(["product", "parameter"])["value_num"].transform("mean")
    grouped_pp_std = filtered_df.groupby(["product", "parameter"])["value_num"].transform("std")

    filtered_df["product_parameter_zscore"] = (
        (filtered_df["value_num"] - grouped_pp_mean) / grouped_pp_std
    )

    filtered_df["product_parameter_zscore"] = (
        filtered_df["product_parameter_zscore"]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )


    # If std == 0 or NaN, fallback to 0.0
    filtered_df["parameter_zscore"] = (
        filtered_df["parameter_zscore"]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )

    base_features = filtered_df[
        [
            "value_num",
            "relative_position_in_range",
            "parameter_zscore",
            "product",
            "parameter",
            "product_parameter_zscore",
        ]
    ].copy()

    encoded_features = pd.get_dummies(
        base_features,
        columns=["product", "parameter"],
        dtype=float,
    )

    numeric_columns = [
        "value_num",
        "relative_position_in_range",
        "parameter_zscore",
        "product_parameter_zscore",
    ]
    for col in numeric_columns:
        encoded_features[col] = encoded_features[col].astype(float)

    return filtered_df, encoded_features