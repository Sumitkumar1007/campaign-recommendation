# Application Logging

This project now has application-level logging for the CatBoost training and inference flows.

The goal is to make every long-running job observable:

- which step is running
- how many rows were loaded or produced
- how long each step took
- where output files were saved
- full stack trace if the application fails

## Default Log Location

Logs are written under:

- [artifacts/logs/](/home/ubuntu/aiml/recommendation/artifacts/logs)

Default log files:

- `artifacts/logs/catboost_training.log`
- `artifacts/logs/catboost_inference.log`

You can override the location with:

```bash
--log-file artifacts/logs/my_custom_run.log
```

## CatBoost Training Logging

Script:

- [train_next_month_strategy_model_catboost.py](/home/ubuntu/aiml/recommendation/scripts/train_next_month_strategy_model_catboost.py)

The training log records:

- command arguments
- dataset preparation start/end
- feature and schedule file paths
- prepared dataset row count
- available source months
- train/validation/test/prediction split row counts
- feature matrix shapes
- target matrix shapes
- start/end of each day model:
  - `D-5`
  - `D-4`
  - `D-3`
  - `D-2`
  - `D-1`
  - `D+1`
  - `D+2`
  - `D+3`
  - `D+4`
  - `D+5`
- model evaluation metrics
- model artifact path and file size
- metrics artifact path and file size
- prediction artifact path and file size
- full exception traceback if anything fails

Example command:

```bash
source venv/bin/activate

python scripts/train_next_month_strategy_model_catboost.py \
  --feature-file data/features/strategy_monthly_features.csv \
  --schedule-file data/schedules/strategy_schedule_dataset_all_months.csv \
  --model-file artifacts/models/next_month_strategy_catboost_3m.joblib \
  --metrics-file artifacts/metrics/next_month_strategy_catboost_3m_metrics.json \
  --prediction-file artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv \
  --target-offset-months 1 \
  --history-window-months 3 \
  --train-source-months NOV-2025 DEC-2025 JAN-2026 \
  --validation-source-months FEB-2026 \
  --prediction-source-months APR-2026 \
  --n-jobs 16 \
  --iterations 500 \
  --learning-rate 0.05 \
  --depth 8 \
  --log-file artifacts/logs/catboost_3m_training.log
```

Watch progress:

```bash
tail -f artifacts/logs/catboost_3m_training.log
```

## CatBoost Inference Logging

Script:

- [predict_next_month_strategy_catboost.py](/home/ubuntu/aiml/recommendation/scripts/predict_next_month_strategy_catboost.py)

The inference log records:

- command arguments
- model bundle load step
- model feature count
- target offset
- history window size
- dataset preparation start/end
- inference candidate row count
- final future prediction row count
- prediction matrix shape
- each predicted day column
- prediction output path and file size
- full exception traceback if anything fails

Example command:

```bash
source venv/bin/activate

python scripts/predict_next_month_strategy_catboost.py \
  --model-file artifacts/models/next_month_strategy_catboost_3m.joblib \
  --feature-file data/features/strategy_monthly_features.csv \
  --schedule-file data/schedules/strategy_schedule_dataset_all_months.csv \
  --prediction-file artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv \
  --prediction-source-months APR-2026 \
  --log-file artifacts/logs/catboost_3m_inference.log
```

Watch progress:

```bash
tail -f artifacts/logs/catboost_3m_inference.log
```

## How To Diagnose A Failed Run

If a run fails, check the log bottom first:

```bash
tail -80 artifacts/logs/catboost_3m_training.log
```

The log will show:

- the last completed step
- the step that failed
- elapsed seconds for the failed step
- Python traceback

This avoids the earlier problem where a long run stopped without leaving a clear error reason.

## Step Names To Look For

Training step names:

- `prepare_dataset`
- `split_dataset`
- `build_feature_matrices`
- `fit_all_day_models`
- `fit_day_model`
- `evaluate_models`
- `save_model_and_metrics`
- `write_predictions`

Inference step names:

- `load_model`
- `prepare_dataset`
- `select_prediction_rows`
- `build_prediction_matrix`
- `predict_day_columns`
- `write_predictions`

Each step has a matching `START` and `END` log line. If you see `FAILED`, the traceback immediately below it is the reason.
