# MLflow Model Registry

Use MLflow for model binaries and model versions. Keep GitHub for code, docs, reports, and lightweight analysis files.

## What Gets Logged

The registration script logs:

- CatBoost 3-month joblib bundle as an MLflow pyfunc model.
- Training and validation metrics from `artifacts/metrics/next_month_strategy_catboost_3m_metrics.json`.
- Supporting analysis docs, including the 27 all-blank APAC review.
- Parameters such as source months, target offset, history window, CatBoost depth, learning rate, and iterations.

The pyfunc model expects prepared monthly feature rows, not raw DB communication rows. The raw DB fetch and rolling feature preparation still belongs in the inference pipeline.

## Install MLflow

```bash
source venv/bin/activate
pip install "mlflow>=2.12,<3"
```

## Option 1: Local MLflow Store

This is best for first testing on the server.

```bash
source venv/bin/activate
python scripts/register_catboost_model_mlflow.py \
  --tracking-uri artifacts/mlruns \
  --experiment-name campaign-recommendation \
  --registered-model-name campaign_next_month_catboost_3m \
  --run-name catboost-3m-may-2026-v1
```

Open the UI:

```bash
source venv/bin/activate
mlflow ui --backend-store-uri artifacts/mlruns --host 0.0.0.0 --port 5000
```

## Option 2: Remote MLflow Tracking Server

Use this for production or a shared team registry.

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-host:5000

source venv/bin/activate
python scripts/register_catboost_model_mlflow.py \
  --experiment-name campaign-recommendation \
  --registered-model-name campaign_next_month_catboost_3m \
  --run-name catboost-3m-may-2026-v1
```

## Recommended Naming

Experiment:

```text
campaign-recommendation
```

Registered model:

```text
campaign_next_month_catboost_3m
```

Run name:

```text
catboost-3m-may-2026-v1
```

## Production Flow

1. Train model and write the `.joblib` bundle plus metrics JSON.
2. Register the model with `scripts/register_catboost_model_mlflow.py`.
3. Promote the MLflow model version to `Staging` or `Production`.
4. In the inference service, load the model by version or alias from MLflow.
5. Fetch only the latest source month from Postgres.
6. Build rolling 3-month features locally or from the processed feature store.
7. Call the MLflow model on prepared feature rows.

## Load Registered Model

After registering, prefer loading by alias:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/campaign_next_month_catboost_3m@production")
predictions = model.predict(prepared_monthly_feature_rows)
```

The `production` alias should point to the approved model version. This avoids hardcoding a numeric model version in inference jobs.

## Important Note

Do not commit `artifacts/mlruns/`, model binaries, raw data, or generated predictions to GitHub. They are ignored by `.gitignore`.
