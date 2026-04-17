# Legacy vs Weighted Benchmark

## Objective

Compare the original legacy baseline against the new weighted-success challenger on the same time-based validation setup.

## Benchmark Setup

- EMI filter: keep only accounts with `emi_date` on the `5th`
- Prediction window: `D-5` to `D+5`, excluding `D`
- Train months: `2025-10`, `2025-11`, `2025-12`
- Validation month: `2026-01`
- Test month: `2026-02`
- Sample size used for benchmark: `100000` rows

## Model Definitions

### Legacy baseline

- channel-aware binary success target
- success states treated as equally positive inside a channel
- examples:
  - `SMS`: `DELIVERED`, `READ`, `CLICKED`
  - `WHATSAPP`: `DELIVERED`, `READ`, `CLICKED`
  - `VOICE`: `CONNECTED`, `CALL_CONNECTED`

### Weighted challenger

- channel-aware success target
- stronger engagement states are given higher weight
- score mapping:
  - `DELIVERED = 0.4`
  - `READ = 0.8`
  - `CLICKED = 1.0`
  - `CONNECTED / CALL_CONNECTED = 1.0`
- training uses weighted sample emphasis so stronger engagement affects the classifier more

## Results Summary

The detailed comparison is in [legacy_vs_weighted_benchmark.csv](./legacy_vs_weighted_benchmark.csv).

Headline result:

- the **legacy model is still the better benchmark baseline** on standard binary classification metrics
- the weighted model is slightly behind on:
  - ROC AUC
  - average precision
  - calibrated accuracy
  - calibrated Brier score
- the weighted model only edges ahead on a small part of the raw Brier score comparison

## Practical Interpretation

This result is expected because the two models are not optimizing exactly the same thing:

- the legacy model is optimized for binary communication success
- the weighted challenger is optimized to value stronger engagement more than weaker engagement

So if the project KPI is:

- **binary communication success**: keep the legacy model as the current production benchmark
- **engagement quality**: the weighted model is directionally better aligned, but it needs evaluation metrics that match the weighted target

## Recommendation

Use this decision for now:

1. Keep the **legacy model** as the benchmark baseline.
2. Keep the **weighted-success model** as the challenger experiment.
3. Add weighted-target evaluation before replacing the baseline, for example:
   - average predicted weighted success by split
   - weighted success calibration
   - bucket-level comparison by risk, day, and channel

## Source Metrics

- Legacy metrics file:
  `d:\aiml\campaign-recommenadtion-legacy-benchmark\outputs\model\metrics.json`
- Weighted metrics file:
  `d:\aiml\campaign-recommenadtion\outputs\model\metrics.json`
