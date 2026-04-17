# Legacy vs Extra Trees Benchmark

## Decision

Promote `extra_trees` over the legacy baseline for the production recommendation model.

## Benchmark Setup

- training rows: `80000`
- EMI filter: keep only `emi_date` on the `5th`
- train months: `2025-10`, `2025-11`, `2025-12`
- validation month: `2026-01`
- test month: `2026-02`

## Calibrated Validation Metrics

### Legacy

- ROC AUC: `0.7790`
- Average Precision: `0.9469`
- Accuracy: `0.8343`
- Brier Score: `0.0948`

### Extra Trees

- ROC AUC: `0.8740`
- Average Precision: `0.9711`
- Accuracy: `0.9151`
- Brier Score: `0.0825`

## Calibrated Test Metrics

### Legacy

- ROC AUC: `0.7389`
- Average Precision: `0.9221`
- Accuracy: `0.8283`
- Brier Score: `0.1116`

### Extra Trees

- ROC AUC: `0.8061`
- Average Precision: `0.9481`
- Accuracy: `0.8688`
- Brier Score: `0.1012`

## Improvement of Extra Trees over Legacy on Test

- ROC AUC: `+0.0672`
- Average Precision: `+0.0260`
- Accuracy: `+0.0405`
- Brier Score: `-0.0104` lower is better

## Conclusion

`extra_trees` is clearly stronger than the legacy model on this benchmark and is the better production default under the current feature and evaluation setup.
