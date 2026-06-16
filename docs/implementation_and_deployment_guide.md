# Campaign Recommendation Implementation And Deployment Guide

## 1. Project Objective

This project builds an ML-based campaign recommendation engine for EMI collections communication.

The system predicts the most suitable communication strategy for each loan account across the EMI window:

- `D-5` to `D+5`
- `D` is excluded

The business rule is applied **per day**:

- `low` risk: up to `1` communication per day
- `medium` risk: up to `2` communications per day
- `high` risk: up to `3` communications per day

If the previous month has no campaign signal for a given day for a customer, that day is returned as `-`.

## 2. High-Level Architecture

The pipeline is organized into these stages:

1. Read raw communication history data
2. Filter only valid EMI records
3. Build channel-aware target labels
4. Build previous-month customer history features
5. Build previous-month day-availability features
6. Create time-based train, validation, and test splits
7. Train the recommendation model
8. Calibrate probabilities
9. Score candidate day/channel/hour combinations
10. Apply daily business quotas by risk
11. Export strategy outputs and summary reports

## 3. Repository Structure

- [README.md](../README.md)
  Short project overview and quick-start commands.

- [config/default_config.json](../config/default_config.json)
  Main configuration file for paths, target definitions, EMI cycle, and training settings.

- [src/campaign_recommendation](../src/campaign_recommendation)
  Main application package.

- [training-data](../training-data)
  Input raw dataset.

- `outputs/`
  Generated model artifacts, recommendations, summaries, and explanations.

- [reports](../reports)
  Benchmark and comparison reports.

## 4. Input Data Used

Training dataset:

- [mfl_last_6_months_communication_data.csv](../training-data/mfl_last_6_months_communication_data.csv)

Important raw columns used:

- `apac_card_number`
- `comm_status`
- `communication_type`
- `payment_unique_id`
- `created_date`
- `risk`
- `collectable_amount`
- `campaign_identifier`
- `emi_date`

## 5. Data Filtering Rules

### EMI Cycle Filter

Only rows where `emi_date` day is present in the configured `emi_cycle` list are used.

This is controlled by:

- `emi_cycle = ["5"]` in [default_config.json](../config/default_config.json)

Example multi-cycle setup:

- `emi_cycle = ["5", "10", "12"]`

### Allowed Strategy Days

Only the following day offsets are considered:

- `-5, -4, -3, -2, -1, 1, 2, 3, 4, 5`

`D` or `day_offset = 0` is excluded from campaign generation.

## 6. Target Label Design

### Binary Success

The model still trains as a classifier with binary success target:

- `target_success = 1` for positive communication outcome
- `target_success = 0` otherwise

### Channel-Aware Success

Success is defined differently by channel:

- `SMS`: `DELIVERED`, `READ`, `CLICKED`
- `WHATSAPP`: `DELIVERED`, `READ`, `CLICKED`
- `VOICE`: `CONNECTED`, `CALL_CONNECTED`

This logic is implemented in:

- [labeling.py](../src/campaign_recommendation/labeling.py)

### Weighted Success Strength

To make stronger engagement more important:

- `DELIVERED = 0.4`
- `READ = 0.8`
- `CLICKED = 1.0`
- `CONNECTED / CALL_CONNECTED = 1.0`

This creates:

- `target_success_score`
- `target_sample_weight`

The classifier still predicts binary success probability, but stronger events influence training more.

## 7. Feature Engineering

Feature engineering is implemented in:

- [features.py](../src/campaign_recommendation/features.py)
- [data.py](../src/campaign_recommendation/data.py)

### Base Features

- `risk`
- `communication_type`
- `campaign_identifier`
- `day_offset`
- `send_hour`
- `collectable_amount`
- `emi_day`
- `emi_month_number`
- `is_predue`
- `is_postdue`

### Previous-Month History Features

For each customer, previous-month aggregates are created:

- `prev_month_total_comms`
- `prev_month_success_count`
- `prev_month_payment_count`
- `prev_month_sms_count`
- `prev_month_whatsapp_count`
- `prev_month_voice_count`
- `prev_month_sms_success_count`
- `prev_month_whatsapp_success_count`
- `prev_month_voice_success_count`
- `prev_month_success_rate`
- `prev_month_payment_rate`

### Previous-Month Day Availability

These indicate whether a customer had campaign activity in the previous month for that specific EMI-relative day:

- `prev_month_day_m5_available`
- `prev_month_day_m4_available`
- `prev_month_day_m3_available`
- `prev_month_day_m2_available`
- `prev_month_day_m1_available`
- `prev_month_day_p1_available`
- `prev_month_day_p2_available`
- `prev_month_day_p3_available`
- `prev_month_day_p4_available`
- `prev_month_day_p5_available`

And a runtime helper feature:

- `prev_month_same_day_available`

This is used to decide whether a day should produce recommendations or `-`.

## 8. Train/Validation/Test Split

The split is time-based, not random.

This is important because recommendation systems for EMI communication must be validated on future months, not mixed-row random samples.

Current split logic:

- train = all older months except last two
- validation = second latest EMI month
- test = latest EMI month

Example from current runs:

- train: `2025-10`, `2025-11`, `2025-12`
- validation: `2026-01`
- test: `2026-02`

## 9. Model Selection Journey

### Initial Legacy Baseline

The first production baseline used logistic regression with calibrated probabilities.

### Weighted Success Challenger

Then channel-aware weighted success was introduced.

### Tree-Based Experiment Benchmark

Multiple experiments were run:

- logistic baseline
- random forest
- extra trees
- histogram gradient boosting
- adaboost

Benchmark reports:

- [model_benchmark_experiments.csv](../reports/model_benchmark_experiments.csv)
- [model_benchmark_experiments.md](../reports/model_benchmark_experiments.md)

### Final Current Default

The current default production model is:

- `ExtraTreesClassifier`

This decision was based on stronger benchmark results against the legacy model.

Decision report:

- [legacy_vs_extra_trees.md](../reports/legacy_vs_extra_trees.md)

## 10. Current Production Model Implementation

Main model code:

- [modeling.py](../src/campaign_recommendation/modeling.py)

Current production training stack:

1. Feature engineering
2. Preprocessing using `ColumnTransformer`
3. `ExtraTreesClassifier`
4. Probability calibration using logistic regression on validation predictions

## 11. Where The Model Artifact Is Stored

The trained model artifact is stored in:

- `outputs/model/model.joblib`

Important:

- this is the primary serialized model artifact
- it is a `joblib` file, not a `.pkl`
- functionally it serves the same deployment purpose as a pickle-style model artifact

Related saved artifacts:

- `outputs/model/metrics.json`
- `outputs/model/run_summary.json`
- `outputs/model/evaluation_sample.csv`
- `outputs/model/validation_slice_report.csv`
- `outputs/model/calibration_report.csv`

## 12. How Training Works

Training entrypoint:

- [__main__.py](../src/campaign_recommendation/__main__.py)

Training pipeline:

- [pipelines/train.py](../src/campaign_recommendation/pipelines/train.py)

Command:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation train --max-rows 80000
```

What happens during training:

1. Load config
2. Read CSV in chunks
3. Filter only EMI day 5 rows
4. Build target labels and weights
5. Build previous-month aggregates
6. Build time-based splits
7. Train Extra Trees classifier
8. Fit probability calibrator
9. Evaluate on validation and test
10. Save model and metrics

## 13. How Inference Works

There are two main inference modes.

### A. Batch Recommendation Generation

Command:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation recommend --model-dir outputs\model
```

Pipeline:

- [pipelines/recommend.py](../src/campaign_recommendation/pipelines/recommend.py)
- [recommend.py](../src/campaign_recommendation/recommend.py)

What it does:

1. Load `model.joblib`
2. Build base population
3. Generate candidate communications for all allowed days, channels, and hourly send times from 9 AM through 6 PM
4. Score every candidate with calibrated model probability
5. Apply per-day quota by risk
6. Insert blank `-` rows for unavailable days
7. Export strategy outputs

### B. Single Account Explanation

Command:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation explain --model-dir outputs\model --account-id MFLAPDSECUL000005010246
```

What it exports:

- day-level best scores
- final selected recommendations
- strategy calendar
- top contributing features for the top recommendation

## 14. What Input Is Needed For Inference

If you already have a prepared account file for current-period inference, the expected fields are:

- `apac_card_number`
- `risk`
- `collectable_amount` or `outstanding_balance`
- `campaign_identifier`
- `emi_date`

Optional but useful if available:

- previous-month feature columns
- day availability columns

If current-state features like `POS`, `bucket`, or additional account descriptors are later added to training data, they can also be added to inference.

## 15. Recommendation Output Files

### Main Strategy Output

- `outputs/recommendations.csv`

This contains the day-by-day strategy calendar with:

- account
- risk
- day
- slot rank
- strategy label
- availability flag

### Bucket Summaries

- `outputs/bucket_counts.csv`
- `outputs/bucket_counts_by_day.csv`

These summarize:

- count by `risk + strategy label`
- count by `risk + day + strategy label`

### Account-Level Explanation Outputs

Examples:

- `outputs/explanations/MFLAPDSECUL000005010246_day_scores.csv`
- `outputs/explanations/MFLAPDSECUL000005010246_final_recommendation.csv`
- `outputs/explanations/MFLAPDSECUL000005010246_strategy_calendar.csv`

## 16. How The Final Strategy Is Built

This is implemented in:

- [recommend.py](../src/campaign_recommendation/recommend.py)

Rule flow:

1. Generate candidates for each account and each day
2. Predict success probability
3. Rank by probability inside the day
4. Apply day-wise quota by risk
5. If the previous month had no activity for that day, return `-`

So the final calendar is not just top-N overall. It is top-N **per day**.

## 17. Explainability

Explainability code:

- [explain.py](../src/campaign_recommendation/explain.py)

What is supported:

- top contributing features for a chosen prediction
- validation slice reporting by:
  - split
  - risk
  - communication type
  - day offset
- calibration report

How explanation is calculated:

1. Reuse model preprocessor
2. Transform candidate row
3. Use underlying model feature contributions
4. Rank highest absolute contributions

## 18. Deployment Notes

Current implementation is a CLI-first batch system.

### Current Deployable Pattern

The simplest deployment pattern is:

1. Train offline on historical batch data
2. Store `model.joblib`
3. Run batch recommendation inference on current EMI population
4. Export CSV outputs for campaign operations

### If Exposing As An API Later

Recommended API flow:

1. Load `outputs/model/model.joblib` at service startup
2. Accept account input payload
3. Build candidate day/channel/hour rows
4. Score candidates
5. Apply per-day quota logic
6. Return JSON calendar and summaries

### Example Inference Service Contract

Input:

- account id
- emi date
- risk
- outstanding balance
- campaign identifier
- previous-month aggregates if externally precomputed

Output:

- daily strategy calendar
- selected communications per day
- `prediction_reason` payload with one business-readable reason per day


### Prediction Reason Storage

Monthly inference writes rank-aware business reasons into:

- `ai_ml_recommendations_data.prediction_reason` as JSONB keyed by `D-5` through `D+5`
- `ai_ml_campaign_mapping.prediction_reason` as text for the mapped campaign day(s)

The reason text follows the strategy rank order. If the payload starts with `-`, no campaign is the primary recommendation and any following campaign is described as an alternate option. If the payload starts with a campaign, that campaign is described as recommended.

## 19. GitHub And Versioning

The project was pushed to:

- `https://github.com/Sumitkumar1007/campaign-recommendation.git`

Repository setup includes:

- Git LFS for the large training CSV
- main branch push completed

Large file tracking:

- [training-data/mfl_last_6_months_communication_data.csv](../training-data/mfl_last_6_months_communication_data.csv) is tracked through Git LFS

## 20. How To Reproduce End To End

### Step 1: Activate environment

```powershell
$env:PYTHONPATH="src"
```

### Step 2: Train model

```powershell
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation train --max-rows 80000
```

### Step 3: Generate recommendations

```powershell
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation recommend --model-dir outputs\model
```

### Step 4: Explain single account

```powershell
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation explain --model-dir outputs\model --account-id MFLAPDSECUL000005010246
```

### Step 5: Benchmark models

```powershell
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation benchmark-models --max-rows 80000
```

## 21. Current Limitations

- current inference mostly relies on communication history and account-level risk/balance context
- richer variables like `POS`, `bucket`, and more loan features are not yet present in the training file
- candidate send hours are generated hourly from 9 AM through 6 PM
- recommendation generation is batch-first and can be slow on large volumes

## 22. Recommended Next Steps

1. Add `POS`, `bucket`, and other current-state loan features if available in training data.
2. Persist precomputed previous-month aggregates for faster inference.
3. Add API serving layer for real-time or on-demand scoring.
4. Add pivoted business report output in Excel-style calendar format.
5. Add experiment registry for model lineage and approval workflow.

## 23. Final Summary

This project now supports:

- EMI-specific campaign recommendation
- day-wise communication quotas by risk
- blank-day handling based on previous-month availability
- channel-aware success logic
- weighted engagement-aware training
- benchmarked model selection
- CatBoost 3-month next-month model as the current production default
- batch inference, account-level explanation, and rank-aware prediction reasons
- saved model artifact and reproducible CLI workflow

## 24. New Server Setup

This section is the operational runbook for moving the project to a different server.

Use it when you want to:

- move the API service to a new machine
- move the monthly inference/training/export scripts to a new machine
- re-create the same runtime with the same DB and SFTP integrations

### 24.1 What Must Be Available On The New Server

Before starting, confirm the new server has:

- Linux shell access with `sudo`
- Python `3.11+` or `3.12`
- Git
- network access to the source Postgres database
- network access to the target Postgres database if different
- network access to the SFTP host if workbook upload is enabled
- a service account/user that can run the app, usually `ubuntu`

Recommended OS packages:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential libpq-dev
```

### 24.2 Decide What You Are Migrating

There are 2 operational modes in this repo.

1. API mode
This is the Digital-team integration mode.
It exposes:

- `/api/v1/auth`
- `/api/v1/training`
- `/api/v1/inference`
- `/api/status`
- `/api/health`
- `/api/v1/transactions/{transactionId}`

2. Script mode
This is direct CLI execution for monthly inference/export/training.

For current usage, API mode is the primary deployment pattern.

### 24.3 Clone Repository

Clone the integration branch directly:

```bash
git clone -b feat/integration-with-digital https://github.com/Sumitkumar1007/campaign-recommendation.git
```

### 24.4 Python Environment

Python requirement:

- Python `3.11+` or `3.12`

Set up the virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

### 24.5 Prepare Required Directories

Create the minimum required artifact directories:

```bash
mkdir artifacts/models
mkdir artifacts/metrics
```

### 24.6 Copy Required Artifacts And Config

Copy these files and directories from the existing server to the new server:

```text
artifacts/models/next_month_strategy_catboost_3m.joblib
artifacts/metrics/next_month_strategy_catboost_3m_metrics.json
artifacts/checkpoints/catboost_3m/
config/default_config.json
```

If using copied artifacts, confirm permissions:

```bash
chmod -R u+rwX artifacts data
```

### 24.7 Create `.env`

Create `.env` with the deployment values below:

```bash
# Postgres source and snapshot storage
PGHOST=10.1.1.45
PGPORT=5432
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=mysecretpassword

# Source communication table
SOURCE_SCHEMA=digital_collections
SOURCE_TABLE=communications

# Target schema/tables used by the pipeline
TARGET_SCHEMA=digital_collections
CAMPAIGN_TABLE=ai_ml_campaign_recommendations
CAMPAIGN_MAPPING_TABLE=ai_ml_campaign_mapping

# Monthly inference defaults
MODEL_NAME=catboost_3m
FEATURE_MONTH_SOURCE=emi_date

# MLflow registry
MLFLOW_TRACKING_URI=http://10.1.1.45:5000
MLFLOW_EXPERIMENT_NAME=campaign-recommendation
MLFLOW_REGISTERED_MODEL_NAME=campaign_next_month_catboost_3m
MLFLOW_RUN_NAME=catboost-3m-may-2026-v1

# API service
API_HOST=0.0.0.0
API_PORT=8040
API_AUTH_USERNAME=aiml
API_AUTH_PASSWORD=aiml
API_AUTH_SECRET=4b32d0dbabc3749210ee55d98bb91bef34071a30126cdb42b24bb4c98c0c1bb8
API_TOKEN_TTL_SECONDS=28800
AI_CONFIG_TABLE=ai_configurations
API_MODEL_BASE_VERSION=v1.1.0
API_EXPORT_AFTER_INFERENCE=true
API_EXPORT_WRITE=true
# SFTP upload after workbook generation
SFTP_EXPORT_PATH=/app/muthoot/digital/kafka-web/aiml/campaign-recommendation/artifacts/exports/sftp
SFTP_REMOTE_DATASET_PATH=/app/muthoot/digital/kafka-web/upload/dataset/sftp
SFTP_REMOTE_SCHEDULER_PATH=/app/muthoot/digital/kafka-web/upload/scheduler/sftp
SFTP_UPLOAD_ENABLED=true
SFTP_HOST=10.1.1.45
SFTP_PORT=22
SFTP_USERNAME=sumit
SFTP_PASSWORD=go4it*22
SFTP_PRIVATE_KEY_PATH=
SFTP_PRIVATE_KEY_PASSPHRASE=
SFTP_REMOTE_PATH=/home/sumit/sftp-test/output
SFTP_RETRIES=3
SFTP_RETRY_DELAY_SECONDS=5
SFTP_TIMEOUT_SECONDS=30
SFTP_FAIL_ON_ERROR=true
```

### 24.8 Update And Install Systemd Service

Update the service file first:

- user
- group
- `WorkingDirectory`
- `EnvironmentFile`
- `ExecStart`

Service file path:

- [deploy/systemd/recommendation.service](../deploy/systemd/recommendation.service)

Then install and start it:

```bash
sudo cp deploy/systemd/recommendation.service /etc/systemd/system/recommendation.service
sudo systemctl daemon-reload
sudo systemctl enable recommendation.service
sudo systemctl restart recommendation.service
```

Check service status:

```bash
sudo systemctl status recommendation.service --no-pager
```

### 24.9 Smoke Test

#### Health

```bash
curl -fsS http://127.0.0.1:8040/api/health
```

#### Auth

```bash
curl -sS -X POST http://127.0.0.1:8040/api/v1/auth \
  -H 'Content-Type: application/json' \
  -d '{"username":"aiml","password":"<api_password>"}'
```

#### Training pre-check

Insert one test row in `ai_configurations` first if not already inserted by the Digital app.

Then call training:

```bash
curl -sS -X POST http://127.0.0.1:8040/api/v1/training \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"transactionId":"TRN202606080003","months":6}'
```

#### Inference pre-check

```bash
curl -sS -X POST http://127.0.0.1:8040/api/v1/inference \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"transactionId":"TRN202606080004"}'
```

#### Poll status

```bash
curl -sS -H 'Authorization: Bearer <token>' \
  http://127.0.0.1:8040/api/v1/transactions/TRN202606080003
```

### 24.10 DB Objects Expected By The App

Source read table:

- `digital_collections.communications`

Target/update tables used by runtime:

- `digital_collections.ai_configurations`
- `digital_collections.ai_ml_campaign_recommendations`
- `digital_collections.ai_ml_campaign_mapping`

Quartz/Digital export target tables when export write is enabled:

- `dataset`
- `qrtz_job_details`
- `qrtz_triggers`
- `qrtz_cron_triggers`

Important API behavior:

- Digital application must insert `transactionId` first into `ai_configurations`
- AIML API updates the existing row
- AIML API does not create the transaction row

### 24.11 Notes

- This setup is for API-mode deployment on the new server.
- The branch used is `feat/integration-with-digital`.
- The service listens on port `8040`.
- `SFTP_FAIL_ON_ERROR=true` means SFTP upload failure will fail the inference/export job.
- If the filesystem path on the new server is different, update both `.env` and `deploy/systemd/recommendation.service` before starting the service.

### 24.12 Manual API Commands Reference

These are the same smoke-test commands, grouped here as a quick reference.

```bash
curl -fsS http://127.0.0.1:8040/api/health

curl -sS -X POST http://127.0.0.1:8040/api/v1/auth \
  -H 'Content-Type: application/json' \
  -d '{"username":"aiml","password":"<api_password>"}'

curl -sS -X POST http://127.0.0.1:8040/api/v1/training \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"transactionId":"TRN202606080003","months":6}'

curl -sS -X POST http://127.0.0.1:8040/api/v1/inference \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"transactionId":"TRN202606080004"}'

curl -sS -H 'Authorization: Bearer <token>' \
  http://127.0.0.1:8040/api/v1/transactions/TRN202606080003
```

### 24.13 Training-Specific Notes On New Server

Current API training behavior:

- `months` now rebuilds training data from Postgres before CatBoost training
- the service prepares raw month extracts, day-level training data, schedule data, and monthly features automatically

Example request:

```json
{
  "transactionId": "TRN202606080003",
  "months": 6
}
```

How it works:

- requested `months=6` resolves a larger historical source window from Postgres
- the training flow rebuilds:
  - `data/training/strategy_training_dataset_all_months.csv`
  - `data/schedules/strategy_schedule_dataset_all_months.csv`
  - `data/features/strategy_monthly_features.csv`
- then CatBoost training starts

Important limitation still applies:

- if any day target like `D-1` has only one unique class in final `train_df`, training will fail by design
- current validation message is explicit, for example:

```text
Training target for D-1 contains only one unique value: '-' (rows=390).
```

### 24.14 Inference And Export On New Server

Monthly inference CLI example:

```bash
./venv/bin/python scripts/run_monthly_inference_pipeline.py \
  --source-month 2026-05 \
  --predict-month 2026-06 \
  --model catboost_3m
```

Workbook export example:

```bash
./venv/bin/python scripts/export_recommendation_workbooks.py \
  --source-month 2026-05 \
  --prediction-month 2026-06 \
  --model catboost_3m \
  --output-dir artifacts/exports/sftp \
  --write
```

Generated files:

- `MD_UB_DATASET_DDMMYYYY_XX.xlsx`
- `MD_UB_SCHEDULER_DDMMYYYY_XX.xlsx`

### 24.15 Logs To Check

Application/service logs:

```bash
sudo journalctl -u recommendation.service -n 200 --no-pager
```

Repo log files:

```text
artifacts/logs/aiml_api.log
artifacts/logs/monthly_inference_pipeline.log
artifacts/logs/catboost_training.log
artifacts/logs/prepare_training_window.log
```

### 24.16 Post-Deployment Verification SQL

Verify API status rows:

```sql
select transaction_id, type, status, message, model_version, accuracy, drift, modified_on
from digital_collections.ai_configurations
order by id desc
limit 20;
```

Verify campaign staging rows:

```sql
select count(*) from digital_collections.ai_ml_campaign_recommendations;
select count(*) from digital_collections.ai_ml_campaign_mapping;
```

Verify export-side tables if enabled:

```sql
select count(*) from digital_collections.dataset;
select count(*) from digital_collections.qrtz_job_details;
select count(*) from digital_collections.qrtz_triggers;
select count(*) from digital_collections.qrtz_cron_triggers;
```

### 24.17 Common Migration Failure Points

1. Wrong working directory in systemd service
- service starts but cannot find scripts, models, or `.env`

2. `.env` copied with old server paths
- SFTP output path or key path may be invalid on new server

3. Postgres reachable from old server but not new server
- allowlist, route, security group, firewall

4. Model artifact not copied
- inference cannot run if local model file is expected but missing

5. Digital app inserts no `transactionId`
- training/inference API returns `transactionId ... not found in ai_configurations`

6. Training data has no day-level class variation
- training fails with single-class target validation such as `D-1` all `'-'`

7. SFTP upload enabled but credentials wrong
- workbook generation may succeed but upload may fail depending on `SFTP_FAIL_ON_ERROR`

### 24.18 Migration Checklist

Use this checklist during cutover.

- server packages installed
- repo cloned/copied
- virtualenv created
- dependencies installed
- model artifact copied
- `.env` created and reviewed
- DB reachability verified
- SFTP reachability verified
- directories created and writable
- manual `/api/health` test passed
- systemd unit installed
- systemd service active on port `8040`
- `/api/v1/auth` test passed
- `/api/v1/training` test passed to `ACCEPTED`
- `/api/v1/inference` test passed to `ACCEPTED`
- transaction polling works
- export files generated/uploaded if enabled
- journal logs clean

### 24.19 Rollback Plan

If migration fails:

1. stop new server traffic
2. keep old server active
3. restore old DNS/IP routing if changed
4. compare:
- `.env`
- model artifact paths
- DB connectivity
- service logs
5. retry cutover only after smoke tests pass on new server
