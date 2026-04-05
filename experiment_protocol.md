# Experiment Protocol — LabSentinel

## 1. Cel eksperymentu

Celem projektu LabSentinel jest porównanie trzech podejść do wykrywania problemów jakości danych laboratoryjnych:

1. **QC only** — podejście regułowe,
2. **ML only** — podejście niesuperwizyjne,
3. **Hybrid QC + ML** — połączenie reguł domenowych i modelu ML.

Projekt ma odpowiedzieć na pytanie badawcze:

> Czy połączenie reguł jakości danych (QC) i modelu ML poprawia wykrywalność błędów pomiarowych względem samego podejścia regułowego?

Dodatkowe pytanie badawcze:

> Jak mierzyć jakość alertów w sytuacji ograniczonych etykiet?

---

## 2. Zakres eksperymentu

Eksperyment obejmuje dane syntetyczne generowane lokalnie przez moduł `labsentinel.generator`.

Każdy rekord opisuje pojedynczy pomiar laboratoryjny i zawiera co najmniej:

- `sample_id`
- `product`
- `parameter`
- `value`
- `unit`
- `date`
- `is_injected_error`
- `error_type`

Obecny eksperyment obejmuje parametry:

- `Water`
- `Sulfur`
- `Chloride`
- `Ash`
- `Viscosity`

Obecny eksperyment obejmuje typy błędów:

### Hard errors
- `missing_value`
- `unit_mismatch`
- `bad_date`
- `out_of_range`

### Soft anomalies
- `near_boundary_anomaly`
- `contextual_shift`

---

## 3. Dane wejściowe

Dane wejściowe są generowane syntetycznie z użyciem kontrolowanego generatora i zapisywane do pliku CSV.

Domyślna komenda:

```bash
python -m labsentinel.generator --seed 42 --rows 600 --out data/raw/lab_measurements.csv