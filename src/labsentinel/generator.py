from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


PRODUCTS = ["Fuel_A", "Fuel_B", "Fuel_C"]
PARAMETERS = ["Water", "Sulfur", "Chloride", "Ash", "Viscosity"]

UNIT_RULES = {
    "Water": "mg/kg",
    "Sulfur": "mg/kg",
    "Chloride": "mg/kg",
    "Ash": "% m/m",
    "Viscosity": "cSt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic laboratory measurements dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/lab_measurements.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_base_value(parameter: str) -> float:
    if parameter == "Water":
        return round(np.random.uniform(5, 20), 3)
    if parameter == "Sulfur":
        return round(np.random.uniform(3, 7), 3)
    if parameter == "Chloride":
        return round(np.random.uniform(0.2, 2.5), 3)
    if parameter == "Ash":
        return round(np.random.uniform(0.05, 0.5), 3)
    if parameter == "Viscosity":
        return round(np.random.uniform(2, 7), 3)
    raise ValueError(f"Unsupported parameter: {parameter}")


def inject_error(
    value: float,
    parameter: str,
    unit: str,
    date_str: str,
) -> tuple[str, str, str, bool, str]:
    """
    Returns:
    value_str, unit, date_str, is_injected_error, error_type
    """
    error_roll = random.random()

    if error_roll < 0.03:
        return "", unit, date_str, True, "missing_value"

    if error_roll < 0.05:
        wrong_unit = "mg/L" if unit != "mg/L" else "unknown"
        return str(value), wrong_unit, date_str, True, "unit_mismatch"

    if error_roll < 0.07:
        return str(value), unit, "bad_date", True, "bad_date"

    if error_roll < 0.09:
        boosted = round(value * random.uniform(3, 8), 3)
        return str(boosted), unit, date_str, True, "out_of_range"

    if error_roll < 0.12:
        # soft anomaly close to upper bound
        near_boundary_map = {
            "Water": 49.0,
            "Sulfur": 19.9,
            "Chloride": 4.95,
            "Ash": 0.98,
            "Viscosity": 9.9,
        }
        return str(near_boundary_map[parameter]), unit, date_str, True, "near_boundary_anomaly"

    if error_roll < 0.14:
        contextual_shift_map = {
            "Water": 42.0,
            "Sulfur": 13.5,
            "Chloride": 3.8,
            "Ash": 0.75,
            "Viscosity": 8.8,
        }
        return str(contextual_shift_map[parameter]), unit, date_str, True, "contextual_shift"

    return str(value), unit, date_str, False, ""


def generate_dataset(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)

    records: list[dict[str, object]] = []

    for i in range(1, rows + 1):
        sample_id = f"S{i:04d}"
        product = random.choice(PRODUCTS)
        parameter = random.choice(PARAMETERS)
        unit = UNIT_RULES[parameter]

        month = random.randint(1, 3)
        day = random.randint(1, 28)
        date_str = f"2026-{month:02d}-{day:02d}"

        base_value = generate_base_value(parameter)

        value_str, final_unit, final_date, is_error, error_type = inject_error(
            value=base_value,
            parameter=parameter,
            unit=unit,
            date_str=date_str,
        )

        records.append(
            {
                "sample_id": sample_id,
                "product": product,
                "parameter": parameter,
                "value": value_str,
                "unit": final_unit,
                "date": final_date,
                "is_injected_error": is_error,
                "error_type": error_type,
            }
        )

    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    args = parse_args()

    df = generate_dataset(rows=args.rows, seed=args.seed)
    saved_path = save_dataset(df, args.out)

    print("Synthetic dataset generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {saved_path}")
    print(df.head(10))