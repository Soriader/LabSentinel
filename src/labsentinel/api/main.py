from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(
    title="LabSentinel API",
    version="0.1.0",
    description="API for accessing latest LabSentinel QC, ML and Hybrid results.",
)

PROCESSED_DIR = Path("data/processed")


def get_latest_run_dir() -> Path:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError("Processed directory does not exist.")

    run_dirs = [p for p in PROCESSED_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError("No processed runs found.")

    return sorted(run_dirs)[-1]


def read_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def dataframe_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if limit is not None:
        df = df.head(limit).copy()

    return df.where(pd.notnull(df), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/runs/latest")
def get_runs_latest() -> dict:
    try:
        run_dir = get_latest_run_dir()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = {
        "run_id": run_dir.name,
        "run_path": str(run_dir),
    }

    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        response["run_config"] = read_json_file(run_config_path)

    return response


@app.get("/alerts/latest")
def get_alerts_latest(
    type: Literal["qc", "ml", "hybrid"] = Query(..., description="Alert type to return."),
    limit: int = Query(50, ge=1, le=1000),
) -> dict:
    try:
        run_dir = get_latest_run_dir()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_map = {
        "qc": run_dir / "alerts_qc.csv",
        "ml": run_dir / "alerts_ml.csv",
        "hybrid": run_dir / "alerts_hybrid.csv",
    }

    try:
        df = read_csv_file(file_map[type])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "run_id": run_dir.name,
        "alert_type": type,
        "count": len(df),
        "items": dataframe_to_records(df, limit=limit),
    }


@app.get("/metrics/latest")
def get_metrics_latest() -> dict:
    try:
        run_dir = get_latest_run_dir()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    file_map = {
        "qc_summary": run_dir / "qc_summary.json",
        "qc_evaluation": run_dir / "qc_evaluation.json",
        "ml_evaluation": run_dir / "ml_evaluation.json",
        "hybrid_evaluation": run_dir / "hybrid_evaluation.json",
        "comparison_summary": run_dir / "comparison_summary.json",
        "ranking_metrics": run_dir / "ranking_metrics.json",
    }

    response = {"run_id": run_dir.name}

    for key, path in file_map.items():
        if path.exists():
            response[key] = read_json_file(path)

    return response