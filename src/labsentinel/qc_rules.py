from __future__ import annotations

from typing import Final

import pandas as pd


UNIT_RULES: Final[dict[str, str]] = {
    "Water": "mg/kg",
    "Sulfur": "mg/kg",
    "Chloride": "mg/kg",
    "Ash": "% m/m",
    "Viscosity": "cSt",
}

VALUE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "Water": (0.0, 500.0),
    "Sulfur": (0.0, 500.0),
    "Chloride": (0.0, 100.0),
    "Ash": (0.0, 5.0),
    "Viscosity": (0.0, 100.0),
}


def apply_unit_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expected unit and check whether the reported unit is correct.
    """
    out = df.copy()

    out["expected_unit"] = out["parameter"].map(UNIT_RULES)
    out["unit_ok"] = out["unit"] == out["expected_unit"]

    # If parameter is unknown, we do not want a fake True/False certainty
    out.loc[out["expected_unit"].isna(), "unit_ok"] = pd.NA

    return out


def apply_completeness_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether core fields required for processing are present.
    """
    out = df.copy()

    required_cols = ["sample_id", "product", "parameter", "unit", "date_dt"]
    for col in required_cols:
        out[f"{col}_ok"] = out[col].notna()

    out["row_complete_ok"] = out[[f"{col}_ok" for col in required_cols]].all(axis=1)
    return out


def apply_date_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether parsed date is valid.
    """
    out = df.copy()
    out["date_ok"] = out["date_dt"].notna()
    return out


def apply_value_range_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate whether numeric value falls into an expected domain range.
    """
    out = df.copy()

    lower_bounds = out["parameter"].map(lambda p: VALUE_RANGES[p][0] if p in VALUE_RANGES else pd.NA)
    upper_bounds = out["parameter"].map(lambda p: VALUE_RANGES[p][1] if p in VALUE_RANGES else pd.NA)

    out["range_lower"] = lower_bounds
    out["range_upper"] = upper_bounds

    out["value_in_range_ok"] = (
        (out["value_num"] >= out["range_lower"]) &
        (out["value_num"] <= out["range_upper"])
    )

    out.loc[out["range_lower"].isna() | out["range_upper"].isna(), "value_in_range_ok"] = pd.NA

    return out


def build_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all baseline QC checks and create a summary quality flag.
    """
    out = df.copy()

    out = apply_unit_rules(out)
    out = apply_completeness_rules(out)
    out = apply_date_rules(out)
    out = apply_value_range_rules(out)

    check_columns = [
        "unit_ok",
        "row_complete_ok",
        "date_ok",
        "value_in_range_ok",
    ]

    def summarize_row(row: pd.Series) -> str:
        failures: list[str] = []

        for col in check_columns:
            value = row[col]
            if pd.isna(value):
                failures.append(f"{col}:unknown")
            elif value is False:
                failures.append(f"{col}:fail")

        if not failures:
            return "ok"

        return "|".join(failures)

    out["qc_status"] = out.apply(summarize_row, axis=1)
    out["qc_passed"] = out["qc_status"] == "ok"

    return out