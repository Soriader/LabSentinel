# Experiment Protocol — LabSentinel

## 1. Experiment Objective

The objective of the LabSentinel project is to compare three approaches to detecting data quality issues in laboratory data:

1. **QC only** — rule-based approach,
2. **ML only** — unsupervised approach,
3. **Hybrid QC + ML** — combination of domain rules and an ML model.

The project aims to answer the following research question:

> Does combining data quality rules (QC) with an ML model improve the detection of measurement errors compared to a purely rule-based approach?

Additional research question:

> How to measure alert quality in a situation of limited labels?

---

## 2. Experiment Scope

The experiment includes synthetic data generated locally by the `labsentinel.generator` module.

Each record describes a single laboratory measurement and contains at least:

- `sample_id`
- `product`
- `parameter`
- `value`
- `unit`
- `date`
- `is_injected_error`
- `error_type`

The current experiment includes the following parameters:

- `Water`
- `Sulfur`
- `Chloride`
- `Ash`
- `Viscosity`

The current experiment includes the following error types:

### Hard errors
- `missing_value`
- `unit_mismatch`
- `bad_date`
- `out_of_range`

### Soft anomalies
- `near_boundary_anomaly`
- `contextual_shift`

---

## 3. Input Data

The input data is generated synthetically using a controlled generator and saved to a CSV file.

Default command:

```bash
python -m labsentinel.generator --seed 42 --rows 600 --out data/raw/lab_measurements.csv