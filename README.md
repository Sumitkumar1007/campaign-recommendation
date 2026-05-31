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
CAMPAIGN_TABLE=ai_ml_campaign_recommendations
```

Monthly inference variables:

```bash
MODEL_NAME=catboost_3m
FEATURE_MONTH_SOURCE=emi_date
modelserving=local
CAMPAIGN_VERTICAL=LAP
CAMPAIGN_VENDOR=prutech-cpass
```

`FEATURE_MONTH_SOURCE=emi_date` is the recommended production setting. It assigns communication activity to the EMI month, so pre-due communication for a 5th EMI does not get counted under the previous calendar month just because it was sent before month-end.

Campaign scheduler defaults:

- `CAMPAIGN_VERTICAL=LAP` is a temporary fixed value until the real vertical source column is available.
- Scheduler vendor is read from `data_config.value` where `key_name='voice.service.vendor-list'`.
- `CAMPAIGN_VENDOR=prutech-cpass` is fallback only when `data_config` is missing or has no usable vendor value.
- `CAMPAIGN_TABLE=ai_ml_campaign_recommendations` stores campaign-level scheduler rows, not customer-level rows.

MLflow variables:

```bash
MLFLOW_TRACKING_URI=http://your_mlflow_host:5000
MLFLOW_EXPERIMENT_NAME=campaign-recommendation
MLFLOW_REGISTERED_MODEL_NAME=campaign_next_month_catboost_3m
MLFLOW_MODEL_URI=models:/campaign_next_month_catboost_3m@production
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

To load the model bundle from MLflow Registry instead of a local joblib file:

```bash
source venv/bin/activate

SOURCE_MONTH=2026-04 \
PREDICT_MONTH=2026-05 \
MODEL_NAME=catboost_3m \
modelserving=mlflow \
MLFLOW_TRACKING_URI=http://your_mlflow_host:5000 \
MLFLOW_MODEL_URI=models:/campaign_next_month_catboost_3m@production \
FEATURE_MONTH_SOURCE=emi_date \
python scripts/run_monthly_inference_pipeline.py
```

The script reads these values from env:

```text
PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
SOURCE_SCHEMA, SOURCE_TABLE
TARGET_SCHEMA, FEATURE_TABLE, PREDICTION_TABLE, AUDIT_TABLE, CAMPAIGN_TABLE
MODEL_NAME, MODEL_SERVING, MLFLOW_MODEL_URI, FEATURE_MONTH_SOURCE
```

The pipeline does this:

1. Fetches communication rows from Postgres for the configured `emi_cycle` dates for the source month and previous two months.
2. Saves raw extracts under `data/communication/MFL_COMMUNICATION_DATA/`.
3. Selects previous two month extract files plus the latest source-month extract.
4. Prepares the day-level inference feature dataset.
5. Rebuilds schedule targets.
6. Rebuilds monthly features with source `risk` from communications.
7. Rolls the latest three months of features per APAC/account.
8. Loads the saved model bundle from local artifacts or MLflow Registry.
9. Predicts schedules for `PREDICT_MONTH`.
10. Saves prediction CSV under `artifacts/predictions/`.
11. Stores processed feature snapshots in Postgres.
12. Stores prediction snapshots in Postgres table `ai_ml_recommendations_data`, including `prediction_payload` and rank-aware `prediction_reason`.
13. Stores a pipeline audit record in Postgres table `ai_ml_audit_table`.
14. Stores campaign scheduler staging rows in Postgres table `ai_ml_campaign_recommendations`.
15. Stores per-loan campaign mapping staging rows in `ai_ml_campaign_mapping`, including the business-readable reason for each mapped campaign.
16. Final MCollect export writes `dataset`, `qrtz_job_details`, `qrtz_triggers`, and `qrtz_cron_triggers` when `scripts/export_mcollect_scheduler.py --write` is run. `digital_rules` templates/verbiages are predefined, managed manually in MCollect, and only referenced by template name from campaign rows.

When omitted, `SOURCE_MONTH` defaults to the current month and `PREDICT_MONTH` defaults to the following month. If provided, `PREDICT_MONTH` must be exactly one month after `SOURCE_MONTH` for the current next-month model.

For production, provide database credentials through environment variables or a secrets manager. Avoid passing `--password` on the command line because command arguments may be visible in process listings.

Retraining is not required just because the prediction month changes. Use the existing promoted model for monthly inference, and retrain only when new labeled outcomes are available, model quality drops, drift appears, feature/source schema changes, business rules change, or the model is stale. See [docs/retraining_policy.md](docs/retraining_policy.md).

## Configuration Rules

EMI cycles are configured in [config/default_config.json](config/default_config.json):

```json
"emi_cycle": ["5"]
```

For the default `["5"]`, the fetcher queries exact EMI dates such as `05/04/2026`. Multiple cycles can be configured, for example:

```json
"emi_cycle": ["5", "10"]
```

Send-hour candidates are configured as a window, not as hardcoded channel-specific lists:

```json
"send_hour_window": {
  "start_hour": 9,
  "end_hour": 18,
  "step_hours": 1
}
```

This generates hourly candidates from `09:00:00` through `18:00:00`. In feature preparation, communication sent before `start_hour` is bucketed into `start_hour`, and communication sent at or after `end_hour` is bucketed into `end_hour`. If future operations require every two hours, set `step_hours` to `2`.


## Prediction Reason Output

`ai_ml_recommendations_data.prediction_reason` stores one business-readable reason per EMI-relative day. It follows the ranking in `prediction_payload`:

- If a day is only `-`, the reason starts with `No campaign is recommended...`.
- If a day starts with a campaign, for example `SMS-9AM-ENGLISH|-`, the reason starts with `SMS at 9AM in English is recommended...`.
- If a day starts with `-` and then has a campaign, for example `-|SMS-8AM-REGIONAL`, the reason starts with `No campaign is the primary recommendation; SMS at 8AM in Regional is kept as an alternate option...`.
- `D` always says no campaign is recommended because it is the EMI due date.

The same rank-aware business reason is also copied into `ai_ml_campaign_mapping.prediction_reason` for mapped campaign rows. This keeps scheduler/account mapping output readable without requiring a separate explanation job.

Example:

```json
{
  "D+2": "No campaign is the primary recommendation; SMS at 8AM in Regional is kept as an alternate option because SMS is a suitable follow-up channel based on past communication history.",
  "D-4": "SMS at 9AM in English is recommended because SMS has shown a positive response pattern for this customer."
}
```

## Campaign Scheduler Output

The pipeline first creates campaign-level staging rows in `digital_collections.ai_ml_campaign_recommendations`. This table is separate from account-level recommendations in `digital_collections.ai_ml_recommendations_data` and per-loan scheduler mappings in `digital_collections.ai_ml_campaign_mapping`. The final MCollect tables are populated in a second step by `scripts/export_mcollect_scheduler.py --write`.

Current campaign scheduler rules:

- `D-5,D-4,D-3,D-2,D-1` become `PRE` / `PREDUE`.
- `D+1,D+2,D+3,D+4,D+5` become `POST` / `POSTDUE`.
- Channel mapping is `SMS -> SMS`, `WH -> WHATSAPP`, and `IVR -> VOICE`.
- Risk mapping is `LOW -> LR`, `MEDIUM -> MR`, and `HIGH -> HR`.
- Scheduler rows are unique campaign definitions, not per-customer rows.
- If the same campaign definition has multiple model-selected hours, the `time` field stores comma-separated values such as `09:00:00,10:00:00`.
- Metadata columns include `source_month`, `prediction_month`, `model_name`, `emi_cycle`, `risk`, `vertical`, `campaign_type`, `due_type`, `created_at`, `modified_at`, `created_by`, and `modified_by`.

Scheduler naming pattern:

```text
{PREDUE/POSTDUE}_AIML_{SMS/WA/VOICE}_{VERTICAL}_{LANGUAGE}_{LR/MR/HR}_{EMI_DATE}TH_{VENDOR}_{DDMMYY}_{N}
```

Example:

```text
PREDUE_AIML_SMS_LAP_ENGLISH_MR_5TH_KALEYRA_200526_2
```

Template name pattern:

```text
{PREDUE/POSTDUE}_AIML_{SMS/WA/VOICE}_{LANGUAGE}
```

Example:

```text
PREDUE_AIML_SMS_ENGLISH
```

Dataset name pattern:

```text
{PREDUE/POSTDUE} AIML {SMS/WA/VOICE} {VERTICAL} {LANGUAGE} {LR/MR/HR} EMI {EMI_DATE}TH [{D-4,D-2}] {HH}
```

Example:

```text
PREDUE AIML SMS LAP ENGLISH MR EMI 5TH [D-4,D-2] 09
```

## Audit Output

Every monthly inference run writes one audit row to `digital_collections.ai_ml_audit_table`.

Audit rows use:

```text
audit_key=model
audit_value=recommendation
```

The audit table is upserted by `audit_key`, `audit_value`, `model_name`, `source_month`, and `prediction_month`. Re-running the same model/month updates the existing audit record instead of inserting a duplicate.

The audit record stores:

- Run status: `SUCCESS` or `FAILED`.
- Prediction completed and failed counts.
- Failure reason when the pipeline fails.
- Duration in seconds.
- Feature table, prediction table, and prediction file path.
- `created_by` and `modified_by` as `campaign-model`.

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

Postgres snapshot and staging tables:

```text
digital_collections.ai_ml_recommendations_feature
digital_collections.ai_ml_recommendations_data
digital_collections.ai_ml_audit_table
digital_collections.ai_ml_campaign_recommendations
digital_collections.ai_ml_campaign_mapping
```

Final MCollect target tables after export:

```text
digital_collections.dataset
digital_collections.qrtz_job_details
digital_collections.qrtz_triggers
digital_collections.qrtz_cron_triggers
```

Manually managed MCollect template table:

```text
digital_collections.digital_rules
```

## Explainability

The inference pipeline stores processed monthly feature snapshots, prediction snapshots, and rank-aware business reasons in Postgres. This allows stakeholder-facing explainability such as:

- Latest 3 months used for the APAC/account.
- Channel intensity by `SMS`, `WH`, and `VOICE`.
- Success and failure counts by channel, time, and language.
- Predicted day-wise recommendation.
- Blank/no-contact cases and best alternatives.
- Business-readable `prediction_reason` text that respects rank order in the payload.

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
