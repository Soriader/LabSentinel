from __future__ import annotations

from typing import Final

import pandas as pd


MISSING_TOKENS: Final[set[str]] = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "missing",
    "error",
    "bad_reading",
    "-",
}


def normalize_text(value: object) -> object:
    """
    Normalize text values:
    - strip spaces
    - lowercase only for comparison-ready cleaning
    - keep non-strings unchanged
    """
    if not isinstance(value, str):
        return value

    return value.strip()


def clean_numeric_value(value: object) -> object:
    """
    Clean a raw numeric field so it can later be converted to float.

    Handles:
    - empty-like tokens
    - decimal commas
    - surrounding spaces
    """
    if value is None:
        return pd.NA

    if not isinstance(value, str):
        return value

    cleaned = value.strip()
    lowered = cleaned.lower()

    if lowered in MISSING_TOKENS:
        return pd.NA

    cleaned = cleaned.replace(",", ".")
    return cleaned


def standardize_unit(unit: object) -> object:
    """
    Standardize unit formatting.
    """
    if unit is None:
        return pd.NA

    if not isinstance(unit, str):
        return unit

    cleaned = unit.strip()

    if cleaned == "":
        return pd.NA

    replacements = {
        "mg / kg": "mg/kg",
        "mg\\kg": "mg/kg",
        "mgkg": "mg/kg",
        "%m/m": "% m/m",
        "% w/w": "% m/m",
        "cst.": "cSt",
        "cst": "cSt",
    }

    normalized_key = cleaned.lower()
    lookup = {k.lower(): v for k, v in replacements.items()}

    return lookup.get(normalized_key, cleaned)


def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the base laboratory DataFrame for downstream QC and ML.

    Expected input columns:
    - sample_id
    - product
    - parameter
    - value
    - unit
    - date

    Output adds:
    - value_clean
    - value_num
    - date_dt
    """
    required_columns = {"sample_id", "product", "parameter", "value", "unit", "date"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    out = df.copy()

    for col in ["sample_id", "product", "parameter", "value", "unit", "date"]:
        out[col] = out[col].map(normalize_text)

    out["value_clean"] = out["value"].map(clean_numeric_value)
    out["value_num"] = pd.to_numeric(out["value_clean"], errors="coerce")

    out["unit"] = out["unit"].map(standardize_unit)

    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")

    return out


def filter_numeric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows that have a valid numeric value.
    """
    if "value_num" not in df.columns:
        raise ValueError("Column 'value_num' not found. Run prepare_base_df() first.")

    return df[df["value_num"].notna()].copy()