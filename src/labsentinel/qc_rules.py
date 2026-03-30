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
    out = df.copy()
    out["expected_unit"] = out["parameter"].map(UNIT_RULES)
    out["unit_ok"] = (out["unit"] == out["expected_unit"]).astype("boolean")
    out.loc[out["expected_unit"].isna(), "unit_ok"] = pd.NA
    return out


def apply_completeness_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required_cols = ["sample_id", "product", "parameter", "unit", "date_dt"]
    for col in required_cols:
        out[f"{col}_ok"] = out[col].notna().astype("boolean")

    out["row_complete_ok"] = out[[f"{col}_ok" for col in required_cols]].all(axis=1).astype("boolean")
    return out


def apply_date_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_ok"] = out["date_dt"].notna().astype("boolean")
    return out


def apply_value_range_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["range_lower"] = out["parameter"].map(
        lambda p: VALUE_RANGES[p][0] if p in VALUE_RANGES else pd.NA
    )
    out["range_upper"] = out["parameter"].map(
        lambda p: VALUE_RANGES[p][1] if p in VALUE_RANGES else pd.NA
    )

    out["value_in_range_ok"] = (
        (out["value_num"] >= out["range_lower"]) &
        (out["value_num"] <= out["range_upper"])
    ).astype("boolean")

    out.loc[out["range_lower"].isna() | out["range_upper"].isna(), "value_in_range_ok"] = pd.NA
    return out


def build_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
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

        return "ok" if not failures else "|".join(failures)

    out["qc_status"] = out.apply(summarize_row, axis=1)
    out["qc_passed"] = (out["qc_status"] == "ok").astype("boolean")

    return out