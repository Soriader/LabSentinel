from __future__ import annotations

from pathlib import Path

import pandas as pd

from labsentinel.alerts import build_qc_alerts
from labsentinel.cleaning import prepare_base_df
from labsentinel.comparison import build_comparison_summary
from labsentinel.evaluation import build_qc_evaluation
from labsentinel.features import prepare_ml_dataset
from labsentinel.hybrid import build_hybrid_alerts, build_hybrid_evaluation
from labsentinel.io_utils import create_run_dir, generate_run_id, save_dataframe, save_json
from labsentinel.ml_evaluation import build_ml_evaluation
from labsentinel.ml_model import build_ml_alerts, run_isolation_forest
from labsentinel.qc_rules import build_qc_flags
from labsentinel.reporting import build_qc_summary


def run_cleaning_and_qc(
    input_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict, dict, dict, Path]:
    """
    Full LabSentinel pipeline:
    cleaning + QC + ML + hybrid + comparison summary.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    df = pd.read_csv(path)
    df = prepare_base_df(df)
    df = build_qc_flags(df)

    qc_alerts_df = build_qc_alerts(df)
    summary = build_qc_summary(df)
    qc_evaluation = build_qc_evaluation(df)

    ml_source_df, ml_features_df = prepare_ml_dataset(df)
    ml_scored_df = run_isolation_forest(ml_source_df, ml_features_df)
    ml_alerts_df = build_ml_alerts(ml_scored_df)
    ml_evaluation = build_ml_evaluation(ml_scored_df)

    hybrid_alerts_df = build_hybrid_alerts(qc_alerts_df, ml_alerts_df)
    hybrid_evaluation = build_hybrid_evaluation(hybrid_alerts_df)

    comparison_summary = build_comparison_summary(
        full_df=df,
        qc_summary=summary,
        qc_evaluation=qc_evaluation,
        ml_evaluation=ml_evaluation,
        hybrid_alerts_df=hybrid_alerts_df,
    )

    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)

    save_dataframe(df, run_dir / "samples_cleaned.csv")
    save_dataframe(qc_alerts_df, run_dir / "alerts_qc.csv")
    save_dataframe(ml_alerts_df, run_dir / "alerts_ml.csv")
    save_dataframe(hybrid_alerts_df, run_dir / "alerts_hybrid.csv")

    save_json(summary, run_dir / "qc_summary.json")
    save_json(qc_evaluation, run_dir / "qc_evaluation.json")
    save_json(ml_evaluation, run_dir / "ml_evaluation.json")
    save_json(hybrid_evaluation, run_dir / "hybrid_evaluation.json")
    save_json(comparison_summary, run_dir / "comparison_summary.json")

    return (
        df,
        qc_alerts_df,
        ml_alerts_df,
        hybrid_alerts_df,
        summary,
        qc_evaluation,
        ml_evaluation,
        hybrid_evaluation,
        comparison_summary,
        run_dir,
    )


if __name__ == "__main__":
    (
        result_df,
        qc_alerts_df,
        ml_alerts_df,
        hybrid_alerts_df,
        summary,
        qc_evaluation,
        ml_evaluation,
        hybrid_evaluation,
        comparison_summary,
        run_dir,
    ) = run_cleaning_and_qc("data/raw/lab_measurements.csv")

    print("Pipeline finished successfully.")
    print(f"Run directory: {run_dir}")

    print("\nCleaned data preview:")
    print(result_df.head(10))

    print("\nQC alerts preview:")
    print(qc_alerts_df.head(10))

    print("\nML alerts preview:")
    print(ml_alerts_df.head(10))

    print("\nHybrid alerts preview:")
    print(hybrid_alerts_df.head(10))

    print("\nQC summary:")
    print(summary)

    print("\nQC evaluation:")
    print(qc_evaluation)

    print("\nML evaluation:")
    print(ml_evaluation)

    print("\nHybrid evaluation:")
    print(hybrid_evaluation)

    print("\nComparison summary:")
    print(comparison_summary)