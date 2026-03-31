from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")


def generate_run_id() -> str:
    """
    Generate a timestamp-based run identifier.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_run_dir(run_id: str) -> Path:
    """
    Create and return the output directory for a given pipeline run.
    """
    run_dir = PROCESSED_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save DataFrame as CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)