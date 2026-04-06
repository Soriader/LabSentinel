from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from labsentinel.alerts import build_qc_alerts
from labsentinel.cleaning import prepare_base_df
from labsentinel.comparison import build_comparison_summary
from labsentinel.evaluation import build_qc_evaluation
from labsentinel.features import prepare_ml_dataset
from labsentinel.hybrid import build_hybrid_alerts, build_hybrid_evaluation
from labsentinel.io_utils import create_run_dir, generate_run_id, save_dataframe, save_json
from labsentinel.manual_review import build_manual_labels_template
from labsentinel.ml_evaluation import build_ml_evaluation
from labsentinel.ml_model import build_ml_alerts, run_isolation_forest
from labsentinel.qc_rules import build_qc_flags
from labsentinel.ranking_metrics import build_ranking_metrics
from labsentinel.reporting import build_qc_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LabSentinel QC + ML + Hybrid pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/lab_measurements.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Top-K ML alerts to keep and evaluate.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_cleaning_and_qc(
    input_path: str,
    seed: int = 42,
    k: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict, dict, dict, dict, Path]:
    """
    Full LabSentinel pipeline:
    cleaning + QC + ML + hybrid + comparison summary + ranking metrics.
    """
    set_seed(seed)

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

    if "anomaly_score" in ml_alerts_df.columns:
        ml_alerts_df = ml_alerts_df.sort_values(by="anomaly_score", ascending=True).head(k).copy()
    else:
        ml_alerts_df = ml_alerts_df.head(k).copy()

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

    ranking_metrics = build_ranking_metrics(
        ml_alerts_df=ml_alerts_df,
        hybrid_alerts_df=hybrid_alerts_df,
        k=k,
    )

    run_id = generate_run_id()
    run_dir = create_run_dir(run_id)

    manual_labels_df = build_manual_labels_template(
        hybrid_alerts_df=hybrid_alerts_df,
        run_id=run_id,
        k=k,
    )

    run_config = {
        "input_path": str(path),
        "seed": seed,
        "k": k,
    }

    save_dataframe(df, run_dir / "samples_cleaned.csv")
    save_dataframe(qc_alerts_df, run_dir / "alerts_qc.csv")
    save_dataframe(ml_alerts_df, run_dir / "alerts_ml.csv")
    save_dataframe(hybrid_alerts_df, run_dir / "alerts_hybrid.csv")
    save_dataframe(manual_labels_df, run_dir / "manual_labels_template.csv")

    save_json(summary, run_dir / "qc_summary.json")
    save_json(qc_evaluation, run_dir / "qc_evaluation.json")
    save_json(ml_evaluation, run_dir / "ml_evaluation.json")
    save_json(hybrid_evaluation, run_dir / "hybrid_evaluation.json")
    save_json(comparison_summary, run_dir / "comparison_summary.json")
    save_json(ranking_metrics, run_dir / "ranking_metrics.json")
    save_json(run_config, run_dir / "run_config.json")

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
        ranking_metrics,
        run_dir,
    )


if __name__ == "__main__":
    args = parse_args()

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
        ranking_metrics,
        run_dir,
    ) = run_cleaning_and_qc(
        input_path=args.input,
        seed=args.seed,
        k=args.k,
    )

    print("Pipeline finished successfully.")
    print(f"Run directory: {run_dir}")
    print(f"Seed: {args.seed}")
    print(f"Input: {args.input}")
    print(f"Top-K ML alerts: {args.k}")

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

    print("\nRanking metrics:")
    print(ranking_metrics)

    print("\nManual labels template saved:")
    print(run_dir / "manual_labels_template.csv")