from __future__ import annotations

from pathlib import Path

import pandas as pd

from labsentinel.cleaning import prepare_base_df
from labsentinel.io_utils import create_run_dir, generate_run_id, save_dataframe
from labsentinel.qc_rules import build_qc_flags


def run_cleaning_and_qc(input_path: str) -> tuple[pd.DataFrame, Path]:
    """
    Read raw laboratory data, clean it, apply QC rules,
    and save the processed output into a timestamped run directory.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    df = pd.read_csv(path)
    df = prepare_base_df(df)
    df = build_qc_flags(df)

    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)

    save_dataframe(df, run_dir / "samples_cleaned.csv")

    return df, run_dir


if __name__ == "__main__":
    result_df, run_dir = run_cleaning_and_qc("data/raw/lab_measurements.csv")
    print("Pipeline finished successfully.")
    print(f"Run directory: {run_dir}")
    print(result_df.head(10))