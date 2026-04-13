# Model Benchmark Experiments

Compared models on the same month-based split and the same training sample.

Split setup:
- Train: 2025-10, 2025-11, 2025-12
- Validation: 2026-01
- Test: 2026-02
- Training rows used: 80000

Models benchmarked:
- logistic_baseline
- random_forest
- extra_trees
- hist_gradient_boosting
- adaboost

Detailed metrics CSV: `model_benchmark_experiments.csv`
Detailed metrics JSON: `model_benchmark_experiments.json`