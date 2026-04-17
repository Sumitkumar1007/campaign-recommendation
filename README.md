# Recommendation Project Layout

This repository is organized so scripts, input data, generated artifacts, and reports live in separate folders.
The scripts use `scripts/project_paths.py`, so the default paths stay relative to this project instead of relying on machine-specific locations.

## Folders

- `scripts/`: training, benchmarking, and dataset-building scripts
- `data/communication/`: raw monthly communication CSV files
- `data/training/`: generated training datasets
- `data/schedules/`: generated schedule-style datasets
- `data/features/`: aggregated feature tables
- `artifacts/models/`: saved model files
- `artifacts/metrics/`: metrics JSON outputs
- `artifacts/predictions/`: prediction CSV outputs
- `artifacts/catboost_info/`: CatBoost training scratch/output directories
- `reports/benchmarks/`: benchmark JSON and CSV summaries
- `venv/`: local Python virtual environment

## Main Scripts

- `scripts/generate_strategy_dataset.py`: builds the day-level training dataset from communication files
- `scripts/build_monthly_feature_dataset.py`: aggregates day-level data into monthly features
- `scripts/build_strategy_schedule_dataset.py`: converts training data into schedule-shaped targets
- `scripts/train_strategy_model.py`: trains the multiclass strategy model
- `scripts/train_strategy_model_low_ram.py`: trains the low-memory month-level model
- `scripts/train_next_month_strategy_model.py`: trains the next-month multi-output model
- `scripts/train_next_month_strategy_model_logistic.py`: logistic next-month baseline
- `scripts/train_next_month_strategy_model_catboost.py`: trains and saves the CatBoost next-month model bundle
- `scripts/predict_next_month_strategy_catboost.py`: inference-only CatBoost next-month prediction
- `scripts/run_monthly_inference_pipeline.py`: fetches one source month, refreshes processed data, runs inference, and stores DB snapshots
- `scripts/benchmark_next_month_models.py`: benchmarks supported next-month models

## Core Files

- monthly features: `data/features/strategy_monthly_features.csv`
- all-month training rows: `data/training/strategy_training_dataset_all_months.csv`
- all-month schedule targets: `data/schedules/strategy_schedule_dataset_all_months.csv`
- benchmark summary: `reports/benchmarks/next_month_model_benchmark_summary.csv`
- benchmark detail: `reports/benchmarks/next_month_model_benchmark.json`

## Run Commands

Activate the local environment:

```bash
source venv/bin/activate
```

Run the next-month benchmark:

```bash
python scripts/benchmark_next_month_models.py
```

Run the next-month logistic model and generate April predictions:

```bash
python scripts/train_next_month_strategy_model_logistic.py
```

Train and save the next-month CatBoost model bundle:

```bash
python scripts/train_next_month_strategy_model_catboost.py
```

Run inference only with the saved CatBoost bundle:

```bash
python scripts/predict_next_month_strategy_catboost.py --prediction-source-months MAR-2026
```

Run the monthly Postgres-backed inference pipeline:

Set credentials with environment variables first:

```bash
export PGHOST=your_postgres_host
export PGPORT=5432
export PGDATABASE=postgres
export PGUSER=postgres
export PGPASSWORD=your_password_here
```

Then run:

```bash
python scripts/run_monthly_inference_pipeline.py \
  --host "$PGHOST" \
  --port "$PGPORT" \
  --dbname "$PGDATABASE" \
  --user "$PGUSER" \
  --password "$PGPASSWORD" \
  --source-schema digital_collections \
  --source-table communications \
  --target-schema digital_collections \
  --source-month 2026-04 \
  --predict-month 2026-05 \
  --model catboost
```

The default split used by the next-month scripts is:

- train source months: `NOV-2025`, `DEC-2025`, `JAN-2026`
- validation source month: `FEB-2026`
- prediction source month: `MAR-2026`
- prediction target month: `APR-2026`

## Typical Workflow

1. Build the day-level training dataset from raw communication files.

```bash
python scripts/generate_strategy_dataset.py
```

2. Convert the day-level dataset into schedule-style targets.

```bash
python scripts/build_strategy_schedule_dataset.py
```

3. Aggregate the day-level data into monthly features.

```bash
python scripts/build_monthly_feature_dataset.py
```

4. Benchmark candidate next-month models.

```bash
python scripts/benchmark_next_month_models.py
```

5. Choose the production candidate from the benchmark summary.

Current strongest options from the latest benchmark:

- `CatBoost` for best validation average day accuracy
- `MLP` for best validation exact match

6. Train the chosen model once and save the production bundle.

For CatBoost:

```bash
python scripts/train_next_month_strategy_model_catboost.py
```

For logistic baseline:

```bash
python scripts/train_next_month_strategy_model_logistic.py
```

7. Run inference for each new source month without retraining.

For CatBoost inference only:

```bash
python scripts/predict_next_month_strategy_catboost.py --prediction-source-months MAR-2026
```

For the incremental Postgres pipeline:

```bash
python scripts/run_monthly_inference_pipeline.py \
  --host "$PGHOST" \
  --port "$PGPORT" \
  --dbname "$PGDATABASE" \
  --user "$PGUSER" \
  --password "$PGPASSWORD" \
  --source-schema digital_collections \
  --source-table communications \
  --target-schema digital_collections \
  --source-month 2026-04 \
  --predict-month 2026-05 \
  --model catboost
```

## Examples

- Benchmark outputs: `reports/benchmarks/`
- Saved predictions: `artifacts/predictions/`
- Saved models: `artifacts/models/`
- Saved metrics: `artifacts/metrics/`
