from __future__ import annotations

from pathlib import Path

import pandas as pd

from labsentinel.alerts import build_qc_alerts
from labsentinel.cleaning import prepare_base_df
from labsentinel.evaluation import build_qc_evaluation
from labsentinel.features import prepare_ml_dataset
from labsentinel.io_utils import create_run_dir, generate_run_id, save_dataframe, save_json
from labsentinel.ml_model import build_ml_alerts, run_isolation_forest
from labsentinel.qc_rules import build_qc_flags
from labsentinel.reporting import build_qc_summary


def run_cleaning_and_qc(
    input_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, Path]:
    """
    Read raw laboratory data, clean it, apply QC rules,
    build QC alerts, summarize QC results, evaluate QC against injected errors,
    run ML anomaly detection on QC-passed rows, and save outputs.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    df = pd.read_csv(path)
    df = prepare_base_df(df)
    df = build_qc_flags(df)

    qc_alerts_df = build_qc_alerts(df)
    summary = build_qc_summary(df)
    evaluation = build_qc_evaluation(df)

    ml_source_df, ml_features_df = prepare_ml_dataset(df)
    ml_scored_df = run_isolation_forest(ml_source_df, ml_features_df)
    ml_alerts_df = build_ml_alerts(ml_scored_df)

    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)

    save_dataframe(df, run_dir / "samples_cleaned.csv")
    save_dataframe(qc_alerts_df, run_dir / "alerts_qc.csv")
    save_dataframe(ml_alerts_df, run_dir / "alerts_ml.csv")
    save_json(summary, run_dir / "qc_summary.json")
    save_json(evaluation, run_dir / "qc_evaluation.json")

    return df, qc_alerts_df, ml_alerts_df, summary, evaluation, run_dir


if __name__ == "__main__":
    result_df, qc_alerts_df, ml_alerts_df, summary, evaluation, run_dir = run_cleaning_and_qc(
        "data/raw/lab_measurements.csv"
    )

    print("Pipeline finished successfully.")
    print(f"Run directory: {run_dir}")

    print("\nCleaned data preview:")
    print(result_df.head(10))

    print("\nQC alerts preview:")
    print(qc_alerts_df.head(10))

    print("\nML alerts preview:")
    print(ml_alerts_df.head(10))

    print("\nQC summary:")
    print(summary)

    print("\nQC evaluation:")
    print(evaluation)