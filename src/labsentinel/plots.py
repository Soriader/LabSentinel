from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate charts for the latest or selected LabSentinel run."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Run ID to visualize, or 'latest'.",
    )
    return parser.parse_args()


def get_run_dir(run_id: str) -> Path:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError("Processed directory does not exist.")

    run_dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError("No processed runs found.")

    if run_id == "latest":
        return sorted(run_dirs)[-1]

    run_dir = PROCESSED_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_metrics(run_dir: Path, charts_dir: Path) -> None:
    comparison = load_json(run_dir / "comparison_summary.json")

    methods = ["QC", "ML", "Hybrid"]
    recall_values = [
        comparison["qc"]["recall"],
        comparison["ml"]["recall_on_scored_rows"],
        comparison["hybrid"]["recall"],
    ]
    precision_values = [
        comparison["qc"]["precision"],
        comparison["ml"]["precision_on_alerts"],
        comparison["hybrid"]["precision"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, recall_values)
    ax.set_title("Recall Comparison: QC vs ML vs Hybrid")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    save_plot(fig, charts_dir / "comparison_recall.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, precision_values)
    ax.set_title("Precision Comparison: QC vs ML vs Hybrid")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    save_plot(fig, charts_dir / "comparison_precision.png")


def plot_tuning_metrics(run_dir: Path, charts_dir: Path) -> None:
    path = run_dir / "tuning_contamination.csv"
    if not path.exists():
        return

    tuning_df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tuning_df["contamination"],
        tuning_df["ml_recall_on_injected_scored_rows"],
        marker="o",
        label="Recall",
    )
    ax.plot(
        tuning_df["contamination"],
        tuning_df["ml_precision_on_alerts"],
        marker="o",
        label="Precision",
    )
    ax.plot(
        tuning_df["contamination"],
        tuning_df["precision_at_k"],
        marker="o",
        label="Precision@K",
    )
    ax.set_title("Isolation Forest Tuning")
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    save_plot(fig, charts_dir / "tuning_contamination.png")


def plot_ranking_metrics(run_dir: Path, charts_dir: Path) -> None:
    ranking = load_json(run_dir / "ranking_metrics.json")

    methods = ["ML", "Hybrid"]
    values = [
        ranking["ml"]["precision_at_k"],
        ranking["hybrid"]["precision_at_k"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, values)
    ax.set_title(f"Precision@{ranking['k']} Comparison")
    ax.set_ylabel("Precision@K")
    ax.set_ylim(0, 1.05)
    save_plot(fig, charts_dir / "precision_at_k.png")


def plot_stability(run_dir: Path, charts_dir: Path) -> None:
    path = run_dir / "stability_jaccard.csv"
    if not path.exists():
        return

    stability_df = pd.read_csv(path)

    labels = [
        f"{row.seed_a} vs {row.seed_b}"
        for row in stability_df.itertuples(index=False)
    ]
    values = stability_df["jaccard_similarity"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("Top-K Alert Stability (Jaccard Similarity)")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(0, 1.05)
    save_plot(fig, charts_dir / "stability_jaccard.png")


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir(args.run_id)
    charts_dir = run_dir / "charts"

    plot_comparison_metrics(run_dir, charts_dir)
    plot_tuning_metrics(run_dir, charts_dir)
    plot_ranking_metrics(run_dir, charts_dir)
    plot_stability(run_dir, charts_dir)

    print("Charts generated successfully.")
    print(f"Run directory: {run_dir}")
    print(f"Charts directory: {charts_dir}")


if __name__ == "__main__":
    main()