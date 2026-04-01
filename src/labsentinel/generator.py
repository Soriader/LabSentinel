from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "lab_measurements.csv"

PARAMETER_RULES = {
    "Water": {
        "unit": "mg/kg",
        "min": 0.0,
        "max": 50.0,
    },
    "Sulfur": {
        "unit": "mg/kg",
        "min": 0.0,
        "max": 20.0,
    },
    "Chloride": {
        "unit": "mg/kg",
        "min": 0.0,
        "max": 5.0,
    },
    "Ash": {
        "unit": "% m/m",
        "min": 0.0,
        "max": 1.0,
    },
    "Viscosity": {
        "unit": "cSt",
        "min": 1.0,
        "max": 10.0,
    },
}

PRODUCTS = ["Fuel_A", "Fuel_B", "Fuel_C"]

ERROR_TYPES = [
    "unit_mismatch",
    "missing_value",
    "bad_date",
    "out_of_range",
    "contextual_shift",
    "near_boundary_anomaly",
]


def random_date(start_date: datetime, end_date: datetime) -> str:
    """
    Generate a random ISO date string between two datetimes.
    """
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 24 * 60 * 60 - 1)

    dt = start_date + timedelta(days=random_days, seconds=random_seconds)
    return dt.strftime("%Y-%m-%d")


def generate_valid_value(parameter: str) -> float:
    """
    Generate a valid numeric value for a given parameter.
    """
    rule = PARAMETER_RULES[parameter]
    return round(random.uniform(rule["min"], rule["max"]), 3)

def generate_contextual_shift_value(parameter: str) -> float:
    """
    Generate a value that is still within the formal QC range,
    but unusually high relative to the normal generated distribution.
    """
    rule = PARAMETER_RULES[parameter]
    min_value = rule["min"]
    max_value = rule["max"]

    # Push value into the upper band of the allowed range
    lower_soft_bound = min_value + 0.75 * (max_value - min_value)
    upper_soft_bound = min_value + 0.95 * (max_value - min_value)

    return round(random.uniform(lower_soft_bound, upper_soft_bound), 3)


def generate_near_boundary_value(parameter: str) -> float:
    """
    Generate a value very close to the upper range boundary,
    but still valid according to QC rules.
    """
    rule = PARAMETER_RULES[parameter]
    min_value = rule["min"]
    max_value = rule["max"]

    if max_value == min_value:
        return round(max_value, 3)

    lower_bound = max_value - 0.03 * (max_value - min_value)
    upper_bound = max_value - 0.001 * (max_value - min_value)

    return round(random.uniform(lower_bound, upper_bound), 3)


def inject_error(row: dict, error_type: str) -> dict:
    """
    Inject a controlled error or anomaly into a generated row.
    """
    parameter = row["parameter"]

    if error_type == "unit_mismatch":
        wrong_units = ["mg/L", "ppm", "%", "cP", "unknown"]
        valid_unit = PARAMETER_RULES[parameter]["unit"]
        candidates = [u for u in wrong_units if u != valid_unit]
        row["unit"] = random.choice(candidates)

    elif error_type == "missing_value":
        row["value"] = ""

    elif error_type == "bad_date":
        row["date"] = "bad_date"

    elif error_type == "out_of_range":
        max_value = PARAMETER_RULES[parameter]["max"]
        row["value"] = str(round(max_value * random.uniform(1.2, 2.0), 3))

    elif error_type == "contextual_shift":
        row["value"] = str(generate_contextual_shift_value(parameter))

    elif error_type == "near_boundary_anomaly":
        row["value"] = str(generate_near_boundary_value(parameter))

    row["error_type"] = error_type
    row["is_injected_error"] = True

    return row


def generate_dataset(
    n_samples: int = 100,
    error_rate: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic laboratory dataset with controlled injected errors.

    Parameters:
    - n_samples: number of sample groups
    - error_rate: fraction of rows that should contain injected errors
    - seed: random seed for reproducibility
    """
    random.seed(seed)

    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 3, 31)

    rows: list[dict] = []

    parameters = list(PARAMETER_RULES.keys())

    for sample_number in range(1, n_samples + 1):
        sample_id = f"S{sample_number:04d}"
        product = random.choice(PRODUCTS)
        date_value = random_date(start_date, end_date)

        for parameter in parameters:
            value = generate_valid_value(parameter)
            unit = PARAMETER_RULES[parameter]["unit"]

            row = {
                "sample_id": sample_id,
                "product": product,
                "parameter": parameter,
                "value": str(value),
                "unit": unit,
                "date": date_value,
                "is_injected_error": False,
                "error_type": "",
            }

            if random.random() < error_rate:
                error_type = random.choice(ERROR_TYPES)
                row = inject_error(row, error_type)

            rows.append(row)

    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, output_path: Path = OUTPUT_FILE) -> None:
    """
    Save generated dataset to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    df = generate_dataset(n_samples=100, error_rate=0.15, seed=42)
    save_dataset(df)

    print("Synthetic dataset generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUTPUT_FILE}")
    print(df.head(10))