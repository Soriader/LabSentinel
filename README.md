# LabSentinel — Hybrid Data Quality Monitoring (QC + ML)

## Overview

LabSentinel is an end-to-end data quality monitoring system for laboratory measurements.
It combines **rule-based validation (QC)** with **unsupervised anomaly detection (ML)** to improve detection of data issues.

The project demonstrates how hybrid approaches outperform traditional rule-based systems, especially for subtle anomalies.

---

## Problem

Traditional data quality systems rely on static rules:

- range checks
- completeness checks
- unit validation

These approaches work well for **hard errors**, but fail to detect:

- contextual anomalies
- near-boundary issues
- subtle distribution shifts

In real-world systems, these undetected issues can lead to:

- incorrect analytics
- poor model performance
- faulty business decisions

---

## Solution

LabSentinel introduces a **hybrid detection architecture**:

### 1. QC Layer (Rule-Based)

Detects deterministic issues:

- missing values
- incorrect units
- invalid dates
- out-of-range values

### 2. ML Layer (Unsupervised)

Detects non-obvious anomalies using:

- relative position within expected range
- parameter-level z-score
- product-parameter z-score

### 3. Hybrid Layer

Combines QC + ML alerts to:

- maximize recall
- maintain acceptable precision
- detect both hard errors and soft anomalies

---

## Architecture

Pipeline:

```
Generator → Cleaning → QC → Feature Engineering → ML → Hybrid → Evaluation
```

Modules:

- `labsentinel.generator` — synthetic data generation with injected errors
- `labsentinel.pipeline` — full processing pipeline
- `labsentinel.features` — ML feature engineering
- `labsentinel.evaluation` — metrics computation

---

## Reproducibility

### Generate dataset

```bash
python -m labsentinel.generator --seed 42 --rows 600 --out data/raw/lab_measurements.csv
```

### Run full pipeline

```bash
python -m labsentinel.pipeline --input data/raw/lab_measurements.csv --seed 42 --k 50
```

Configuration is saved in:

```
data/processed/<run_id>/run_config.json
```

---

## Results

### Detection Performance

| Approach | Recall | Precision | Notes |
|----------|--------|----------|------|
| QC       | 0.60   | 1.00     | Misses soft anomalies |
| ML       | 0.93   | 0.62     | Strong on subtle patterns |
| Hybrid   | **0.97** | **0.81** | Best overall performance |

---

### Error Type Breakdown

| Error Type              | QC   | ML   | Hybrid |
|------------------------|------|------|--------|
| bad_date               | 1.00 | —    | 1.00   |
| missing_value          | 1.00 | —    | 1.00   |
| unit_mismatch          | 1.00 | —    | 1.00   |
| out_of_range           | 0.80 | 0.00 | 0.80   |
| near_boundary_anomaly  | 0.00 | 1.00 | 1.00   |
| contextual_shift       | 0.00 | 1.00 | 1.00   |

---

### Ranking Quality (Top-K)

| Method  | Precision@50 |
|---------|-------------|
| ML      | 0.62        |
| Hybrid  | **0.66**    |

---

## Key Insights

- Rule-based systems are insufficient for real-world anomaly detection
- ML significantly improves detection of soft anomalies
- Hybrid approach provides the best balance of recall and precision
- Ranking (Top-K) enables efficient manual validation

---

## Manual Review

The system supports human-in-the-loop validation:

Generated file:

```
data/processed/<run_id>/manual_labels_template.csv
```

Fields:

- `validator_label` (true_issue / false_alarm / uncertain)
- `validator_notes`

Metric:

```
precision@k_manual = true_issues / k
```

This simulates real production workflows.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- CLI-based pipeline
- Modular architecture

---

## Why This Project Matters

This project demonstrates:

- real-world data quality challenges
- hybrid QC + ML system design
- anomaly detection without labels
- evaluation under limited ground truth
- production-oriented thinking (CLI, reproducibility, metrics)

---

## Future Improvements

- Replace heuristic ML with Isolation Forest / LOF
- Add streaming (Kafka) integration
- Build dashboard (e.g. Streamlit / Power BI)
- Add alert prioritization model
- Integrate real-world datasets

---

## Author

Project created as part of AI / Data Engineering learning path.

Focus areas:

- Data Quality
- Anomaly Detection
- Data Engineering pipelines
- Applied Machine Learning