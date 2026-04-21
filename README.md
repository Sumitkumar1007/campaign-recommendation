# Campaign Recommendation

ML-based EMI campaign recommendation engine for deciding **who to contact, on which day, through which channel, at what time, and in which language**.

The current production flow predicts a day-wise communication schedule for the EMI window:

```text
D-5, D-4, D-3, D-2, D-1, D, D+1, D+2, D+3, D+4, D+5
```

`D` is always kept as `-` because no communication is planned on EMI day.

## Business Rule

The model returns strategy labels like:

```text
SMS-9AM-ENGLISH
WH-9AM-TELUGU
IVR-2PM-ENGLISH
-
```

Risk-based output count:

- `LOW`: top `1` recommendation per day.
- `MEDIUM`: top `2` recommendations per day, joined with `|`.
- `HIGH`: top `3` recommendations per day, joined with `|`.
- `-`: no communication recommended.

Example:

```text
-|SMS-3PM-ENGLISH|WH-9AM-ENGLISH
```

This means no-contact is rank 1, `SMS-3PM-ENGLISH` is rank 2, and `WH-9AM-ENGLISH` is rank 3.

## Current Production Model

Current production model:

```text
campaign_next_month_catboost_3m
```

Model type:

```text
CatBoost next-month strategy model
```

Feature window:

```text
Latest 3 months of monthly communication features per APAC/account
```

MLflow registry:

```text
Experiment: campaign-recommendation
Registered model: campaign_next_month_catboost_3m
Production alias: models:/campaign_next_month_catboost_3m@production
```

Local model bundle path, if running without MLflow:

```text
artifacts/models/next_month_strategy_catboost_3m.joblib
```

Large model files, raw data, generated predictions, logs, and local MLflow artifacts are not committed to GitHub.

## Project Structure

- `scripts/`: data build, training, inference, MLflow registration, and Postgres pipeline scripts.
- `src/campaign_recommendation/`: package-style baseline pipeline kept from earlier project structure.
- `config/`: static project configuration.
- `data/communication/`: raw monthly communication extracts from Postgres or local CSVs.
- `data/training/`: generated day-level training datasets.
- `data/schedules/`: generated schedule-shaped target datasets.
- `data/features/`: generated monthly feature datasets.
- `artifacts/models/`: local model bundles.
- `artifacts/metrics/`: model metrics JSON files.
- `artifacts/predictions/`: generated prediction CSVs.
- `artifacts/logs/`: application logs.
- `artifacts/analysis/`: lightweight analysis files safe to review/share.
- `docs/`: technical explainers and operational guides.
- `reports/`: benchmark reports from earlier experiments.

## Environment Setup

Create `.env` from the template:

```bash
cp .env.example .env
```

Edit `.env` on the server and fill real values. Do not commit `.env`.
The scripts load `.env` automatically, so DB credentials do not need to be passed as command arguments.

Required database variables:

```bash
PGHOST=your_postgres_host
PGPORT=5432
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=replace_with_secret
```

Source and target table variables:

```bash
SOURCE_SCHEMA=digital_collections
SOURCE_TABLE=communications
TARGET_SCHEMA=digital_collections
FEATURE_TABLE=ai_ml_recommendations_feature
PREDICTION_TABLE=ai_ml_recommendations_data
AUDIT_TABLE=ai_ml_audit_table
```

Monthly inference variables:

```bash
MODEL_NAME=catboost_3m
FEATURE_MONTH_SOURCE=created_date
```

MLflow variables:

```bash
MLFLOW_TRACKING_URI=http://your_mlflow_host:5000
MLFLOW_EXPERIMENT_NAME=campaign-recommendation
MLFLOW_REGISTERED_MODEL_NAME=campaign_next_month_catboost_3m
MLFLOW_RUN_NAME=catboost-3m-may-2026-v1
```

All DB credentials and deployment-specific values should come from `.env`, a secrets manager, or the deployment platform. They should not be hardcoded in scripts or committed to Git.

## Python Setup

Activate the existing environment:

```bash
source venv/bin/activate
```

Install project dependencies if needed:

```bash
pip install -e .
pip install "psycopg[binary]" "catboost>=1.2,<2" "mlflow>=2.12,<3"
```

## End-To-End Monthly Inference

Use this when predicting a new target month from the latest source month.

Example: predict May 2026 using April 2026 communication data.

```bash
source venv/bin/activate
python scripts/run_monthly_inference_pipeline.py
```

The script reads these values from env:

```text
PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
SOURCE_SCHEMA, SOURCE_TABLE
TARGET_SCHEMA, FEATURE_TABLE, PREDICTION_TABLE, AUDIT_TABLE
MODEL_NAME, FEATURE_MONTH_SOURCE
```

The pipeline does this:

1. Fetches latest communication rows from Postgres for the configured `emi_cycle` dates in the current month/year.
2. Saves raw latest extract under `data/communication/MFL_COMMUNICATION_DATA/`.
3. Selects previous two month extract files plus the latest extract.
4. Rebuilds the day-level training-style dataset.
5. Rebuilds schedule targets.
6. Rebuilds monthly features with source `risk` from communications.
7. Rolls the latest three months of features per APAC/account.
8. Loads the saved model bundle.
9. Predicts schedules for `PREDICT_MONTH`.
10. Saves prediction CSV under `artifacts/predictions/`.
11. Stores processed feature snapshots in Postgres.
12. Stores prediction snapshots in Postgres table `ai_ml_recommendations_data`.
13. Stores a pipeline audit record in Postgres table `ai_ml_audit_table`.

When omitted, `SOURCE_MONTH` defaults to the current month and `PREDICT_MONTH` defaults to the following month. If provided, `PREDICT_MONTH` must be exactly one month after `SOURCE_MONTH` for the current next-month model.

For production, provide database credentials through environment variables or a secrets manager. Avoid passing `--password` on the command line because command arguments may be visible in process listings.

Retraining is not required just because the prediction month changes. Use the existing promoted model for monthly inference, and retrain only when new labeled outcomes are available, model quality drops, drift appears, feature/source schema changes, business rules change, or the model is stale. See [docs/retraining_policy.md](docs/retraining_policy.md).

The command can still be overridden through CLI args:

```bash
python scripts/run_monthly_inference_pipeline.py \
  --source-month 2026-04 \
  --predict-month 2026-05 \
  --model catboost_3m
```

## Inference Only

Use this when raw data and monthly features are already built locally and you only want to regenerate predictions.

```bash
source venv/bin/activate
python scripts/predict_next_month_strategy_catboost.py \
  --model-file artifacts/models/next_month_strategy_catboost_3m.joblib \
  --feature-file data/features/strategy_monthly_features.csv \
  --schedule-file data/schedules/strategy_schedule_dataset_all_months.csv \
  --prediction-file artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv \
  --prediction-source-months APR-2026
```

## Register Model In MLflow

Use MLflow for model binaries and versions instead of GitHub.

```bash
set -a
source .env
set +a

source venv/bin/activate
python scripts/register_catboost_model_mlflow.py
```

Load the production model from MLflow:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/campaign_next_month_catboost_3m@production")
predictions = model.predict(prepared_monthly_feature_rows)
```

More detail: [MLflow Model Registry](docs/mlflow_model_registry.md)

## Training

Train the 3-month CatBoost model:

```bash
source venv/bin/activate
python scripts/train_next_month_strategy_model_catboost.py \
  --history-window-months 3 \
  --model-file artifacts/models/next_month_strategy_catboost_3m.joblib \
  --metrics-file artifacts/metrics/next_month_strategy_catboost_3m_metrics.json \
  --prediction-file artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv \
  --train-source-months NOV-2025 DEC-2025 JAN-2026 \
  --validation-source-months FEB-2026 \
  --prediction-source-months APR-2026
```

Current split used during development:

- Train source months: `NOV-2025`, `DEC-2025`, `JAN-2026`
- Validation source month: `FEB-2026`
- Prediction source month: `APR-2026`
- Prediction target month: `MAY-2026`

## Data Build Scripts

Build day-level strategy dataset:

```bash
python scripts/generate_strategy_dataset.py
```

Build schedule-shaped targets:

```bash
python scripts/build_strategy_schedule_dataset.py
```

Build monthly feature dataset:

```bash
python scripts/build_monthly_feature_dataset.py
```

## Key Outputs

Prediction CSV:

```text
artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv
```

Metrics JSON:

```text
artifacts/metrics/next_month_strategy_catboost_3m_metrics.json
```

Application logs:

```text
artifacts/logs/
```

Postgres snapshot tables:

```text
digital_collections.ai_ml_recommendations_feature
digital_collections.ai_ml_recommendations_data
digital_collections.ai_ml_audit_table
```

## Explainability

The inference pipeline stores processed monthly feature snapshots and prediction snapshots in Postgres. This allows stakeholder-facing explainability such as:

- Latest 3 months used for the APAC/account.
- Channel intensity by `SMS`, `WH`, and `VOICE`.
- Success and failure counts by channel, time, and language.
- Predicted day-wise recommendation.
- Blank/no-contact cases and best alternatives.

Example analysis file:

```text
artifacts/analysis/may_2026_all_blank_27_apac_analysis.md
```

More detail: [Three Month CatBoost Training Explainer](docs/three_month_catboost_training_explainer.md)

Generate model-derived explanation files for one loan account:

```bash
source venv/bin/activate
python scripts/explain_strategy_prediction.py \
  --loan-number MFLKTKSECUL000005349103 \
  --output-dir artifacts/explanations/single_account
```

This creates:

```text
artifacts/explanations/single_account/MFLKTKSECUL000005349103_prediction_explanation.md
artifacts/explanations/single_account/prediction_explanation_account_summary.csv
artifacts/explanations/single_account/prediction_explanation_day_detail.csv
```

Generate explanation files for all accounts in the prediction file:

```bash
source venv/bin/activate
python scripts/explain_strategy_prediction.py \
  --output-dir artifacts/explanations/may_2026_all
```

This creates:

```text
artifacts/explanations/may_2026_all/prediction_explanation_summary.md
artifacts/explanations/may_2026_all/prediction_explanation_account_summary.csv
artifacts/explanations/may_2026_all/prediction_explanation_day_detail.csv
```

The explainer writes:

- `prediction_explanation_account_summary.csv`: one row per APAC/account with 3-month feature and schedule history summary.
- `prediction_explanation_day_detail.csv`: one row per APAC/account/day with prediction, top probability, blank probability, best non-blank alternative, top candidates, and reason.
- `<loan_number>_prediction_explanation.md`: single-account stakeholder-readable Markdown when `--loan-number` is used.
- `prediction_explanation_summary.md`: all-account summary when explaining every account.

`artifacts/explanations/` exists in GitHub with a `.gitkeep` placeholder only. Generated explanation files are not committed because they can become large and are environment/data specific.

One committed sample is available here:

```text
docs/examples/MFLKTKSECUL000005349103_prediction_explanation.md
```

## Operational Checks

Check logs:

```bash
tail -f artifacts/logs/catboost_inference.log
tail -f artifacts/logs/mlflow_register_catboost.log
```

Check prediction row count:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv")
print(len(df))
print(df["SOURCE_RISK"].value_counts())
PY
```

Expected recent April-to-May test result:

```text
Prediction rows: 2,214
LOW: 1,565
MEDIUM: 339
HIGH: 310
Exact all-blank rows: 27
```

## Important Notes

- Do not commit `.env`, raw data, model binaries, generated predictions, or logs.
- GitHub stores code and docs only.
- MLflow stores large model artifacts and versions.
- `VOICE` is mapped to `IVR` in strategy output.
- `VOICE_BOT` and `WHATSAPP_BUTTON` are excluded from strategy dataset creation.
- Risk is read from `communications.risk`; it is not derived from channel counts.
- The model predicts communication strategy, not direct payment conversion.
- Validation/testing are month-based, not random-row based.

## Extra Documentation

- [Application Logging](docs/application_logging.md)
- [MLflow Model Registry](docs/mlflow_model_registry.md)
- [Three Month CatBoost Training Explainer](docs/three_month_catboost_training_explainer.md)
- [Implementation And Deployment Guide](docs/implementation_and_deployment_guide.md)
- [Stakeholder Summary](docs/stakeholder_summary.md)
