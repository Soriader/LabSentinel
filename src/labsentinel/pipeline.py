from __future__ import annotations

from pathlib import Path

import pandas as pd

from labsentinel.cleaning import prepare_base_df
from labsentinel.qc_rules import build_qc_flags


def run_cleaning_and_qc(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    df = pd.read_csv(path)
    df = prepare_base_df(df)
    df = build_qc_flags(df)
    return df


if __name__ == "__main__":
    result = run_cleaning_and_qc("data/raw/lab_measurements.csv")
    print(result.head(10))