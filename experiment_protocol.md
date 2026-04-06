# Experiment Protocol — LabSentinel

## 1. Experiment Objective

The objective of the LabSentinel project is to compare three approaches to detecting data quality issues in laboratory data:

1. **QC only** — rule-based approach  
2. **ML only** — unsupervised approach  
3. **Hybrid QC + ML** — combination of domain rules and an ML model  

The project aims to answer the following research question:

> Does combining data quality rules (QC) with an ML model improve the detection of measurement errors compared to a purely rule-based approach?

Additional research question:

> How to evaluate alert quality when labeled data is limited?

---

## 2. Experiment Scope

The experiment includes synthetic data generated locally by the `labsentinel.generator` module.

Each record describes a single laboratory measurement and contains the following fields:

- `sample_id`
- `product`
- `parameter`
- `value`
- `unit`
- `date`
- `is_injected_error`
- `error_type`

### The experiment covers the following parameters:

- `Water`
- `Sulfur`
- `Chloride`
- `Ash`
- `Viscosity`

### The experiment includes the following error types:

#### Hard errors

- `missing_value`
- `unit_mismatch`
- `bad_date`
- `out_of_range`

#### Soft anomalies

- `near_boundary_anomaly`
- `contextual_shift`

---

## 3. Input Data

The input data is generated synthetically using a controlled generator and saved as a CSV file.

Default command:

```bash
python -m labsentinel.generator --seed 42 --rows 600 --out data/raw/lab_measurements.csv
```

### Parameters

- `--seed` — ensures reproducibility  
- `--rows` — number of generated records  
- `--out` — output file path  

---

## 4. Processing Pipeline

The full pipeline is executed using:

```bash
python -m labsentinel.pipeline --input data/raw/lab_measurements.csv --seed 42 --k 50
```

### Pipeline steps

1. Data cleaning and normalization  
2. QC rule evaluation  
3. ML feature engineering  
4. ML anomaly detection  
5. Hybrid alert generation (QC + ML)  
6. Evaluation metrics computation  
7. Ranking of alerts (Top-K)
8. Manual review template generation  

---

## 5. Evaluation Metrics

### 5.1 QC Evaluation

- Recall on injected errors  
- Breakdown by error type  

### 5.2 ML Evaluation

- Recall on injected errors within QC-passed data
- Precision on alerts  
- Detection of soft anomalies  

### 5.3 Hybrid Evaluation

- Total detected injected errors  
- Precision on alerts  
- Combined recall and precision performance

### 5.4 Ranking Metrics

For Top-K alerts:

```
precision@k = true_positives_in_top_k / k
```

This is calculated separately for:

- ML  
- Hybrid approach  

---

## 6. Manual Review Protocol

To validate the real-world usefulness of anomaly detection, a manual review process is introduced.

### 6.1 Purpose

The goal of manual review is to:

- assess whether detected anomalies are truly problematic  
- measure real-world precision of the system  
- simulate human-in-the-loop validation in production  

---

### 6.2 Input for manual review

Manual validation is performed on:

- **Top-K hybrid alerts**

Generated using:

```bash
python -m labsentinel.pipeline --input ... --seed ... --k K
```

Results are saved to:

```
data/processed/<run_id>/manual_labels_template.csv
```

---

### 6.3 Annotation fields

Each record must be labeled manually using:

- `validator_label`  
- `validator_notes`  

---

### 6.4 Label definitions

Each alert should be classified into one of the following:

#### true_issue

- The record represents a real anomaly or data quality issue  
- Would require action in a production  

#### false_alarm

- The alert is not practically relevant  
- No action required in production  

#### uncertain

- Cannot confidently classified  
- Requires expert input or additional data  

---

### 6.5 Evaluation metric

After manual labeling:

```
precision@k_manual = true_issues / k
```

Where:

- `true_issues` = number of rows labeled as `true_issue`  
- `k` = number of reviewed alerts  

---

### 6.6 Notes

- Manual labels override synthetic labels (`is_injected_error`)  
- This step reflects real-world validation conditions  
- It is critical for evaluating real-world business value, not only model performance  