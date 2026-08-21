# Campaign Recommendation Model: Detailed Build And Sample Trace

## Scope

This document explains how campaign recommendation model is built in this repo, end to end:

- raw communication rows
- preprocessing and feature generation
- training label generation
- CatBoost model bundle
- inference input/output
- campaign scheduler rows
- real sample records from local project data

Main current entry points:

- training: `scripts/train_next_month_strategy_model_catboost.py`
- inference orchestration: `scripts/run_monthly_inference_pipeline.py`
- raw-to-day features: `scripts/generate_strategy_dataset.py`
- monthly features: `scripts/build_monthly_feature_dataset.py`
- schedule targets: `scripts/build_strategy_schedule_dataset.py`
- inference-only scorer: `scripts/predict_next_month_strategy_catboost.py`
- common model prep: `scripts/pipeline_common.py`

## One-Line Model Definition

This is supervised next-month campaign schedule prediction.

For each customer/entity and EMI-relative day, model predicts best campaign strategy label:

```text
CHANNEL-HOUR-LANGUAGE
```

Examples:

```text
SMS-11AM-ENGLISH
WH-4PM-ENGLISH
VOICE_BOT-10AM-ENGLISH
IVR-3PM-TAMIL
-
```

`-` means no campaign recommended.

Important: this is not collaborative filtering. Account id is not used as model feature. Model learns population-level behavior from engineered communication counts.

## Current Day Coverage

Current code defines trainable day columns in `scripts/pipeline_common.py`:

```text
D-5, D-4, D-3, D-2, D-1,
D+1, D+2, ... D+20
```

That is 25 modelable days. `D` is EMI due date and always carried as `-` in schedule output.

Artifact caveat:

- `artifacts/models/next_month_strategy_catboost_3m.joblib` currently has 10 day models.
- versioned bundles such as `next_month_strategy_catboost_3m_v1.1.1.joblib` have 25 day models.
- current source code expects 25 day columns because `DAY_COLUMNS` includes post-due `D+1` through `D+20`.

## Actual Local Dataset Shapes

Current local inference data:

| File | Shape | Notes |
|---|---:|---|
| `data/training/strategy_training_dataset_inference.csv` | 634,374 x 101 | day-level engineered features |
| `data/features/strategy_monthly_features_inference.csv` | 128,136 x 98 | monthly account/entity features |
| `data/schedules/strategy_schedule_dataset_inference.csv` | 128,136 x 54 | schedule targets |
| `data/cases/digital_cases_AUG2026.csv` | 9,248 x 6 | August base population |
| `artifacts/predictions/2026_08_strategy_predictions_catboost_3m.csv` | 9,061 x 30 | August prediction output |

Training data currently present:

| File | Shape | Months |
|---|---:|---|
| `data/training/strategy_training_dataset_train.csv` | 758,930 x 110 | NOV-2025 to JUL-2026 |
| `data/features/strategy_monthly_features_train.csv` | 159,274 x 107 | NOV-2025 to JUL-2026 |
| `data/schedules/strategy_schedule_dataset_train.csv` | 159,274 x 54 | NOV-2025 to JUL-2026 |

## Raw Inputs

### 1. Communication History

Source: Postgres `communications`, fetched by `scripts/fetch_month_from_postgres.py`.

Local CSV example:

```text
data/communication/comm_data_JUL2026.csv
```

Important raw columns:

| Column | Meaning |
|---|---|
| `apac_card_number` | loan/account number shown in final output |
| `party_id` | optional party/customer id |
| `comm_status` | raw outcome such as `DELIVERED`, `NO_ANSWER`, `CONNECTED` |
| `communication_type` | raw channel such as `SMS`, `WHATSAPP`, `VOICE`, `VOICE_BOT` |
| `verbiage_language` | message/call language |
| `vertical` | business vertical |
| `risk` | source risk bucket |
| `emi_date` | due date anchor |
| `date` | communication event date |
| `created_date` | timestamp used for send-hour bucket |

### 2. Digital Cases / Prediction Population

Source: Postgres `digital_cases`, fetched by `scripts/fetch_cases_from_postgres.py`.

Local CSV example:

```text
data/cases/digital_cases_AUG2026.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `apac_card_number` | account to score |
| `party_id` | optional entity id |
| `risk` | current prediction-month risk |
| `vertical` | current vertical |
| `collectable_amount` | current balance |
| `emi_date` | prediction-month EMI date |

## Entity Key Logic

Code: `scripts/entity_keys.py`

Default behavior:

```text
ENTITY_KEY = apac_card_number
```

If env var `STRATEGY_USE_PARTY_ID=true`:

```text
ENTITY_KEY = party_id
```

All training joins use `ENTITY_KEY` when present. Final output still uses `Loan_number` / `apac_card_number` for downstream users and scheduler mapping.

## Raw-To-Feature Logic

Main function: `process_chunk(...)` in `scripts/generate_strategy_dataset.py`.

### Normalization

Raw channel normalization:

| Raw `communication_type` | Internal channel |
|---|---|
| `SMS` | `SMS` |
| `WHATSAPP` | `WH` |
| `VOICE` | `VOICE` |
| `VOICE_BOT` | `VOICE_BOT` |

Scheduler export later maps `VOICE` strategy labels to `IVR` display in schedule table, then to scheduler mode `VOICE`.

Status normalization:

```text
trim -> uppercase
```

Language normalization:

```text
trim -> uppercase -> replace non-alphanumeric with _
```

Examples:

```text
English  -> ENGLISH
Regional -> REGIONAL
Tamil    -> TAMIL
```

Risk/vertical normalization:

```text
missing -> UNKNOWN
trim -> uppercase
```

### Send-Hour Bucketing

Code uses `created_date.hour`, then clamps into configured send window from `config/default_config.json`.

Default buckets:

```text
9AM, 10AM, 11AM, 12PM, 1PM, 2PM, 3PM, 4PM, 5PM, 6PM
```

Rules:

- before start hour -> start hour
- at/after end hour -> end hour
- otherwise floor to nearest configured hour bucket

### EMI-Relative Day

Offset:

```text
offset = date - emi_date
```

Examples:

| `date` | `emi_date` | offset | day label |
|---|---|---:|---|
| 2026-07-01 | 2026-07-05 | -4 | `D-4` |
| 2026-07-05 | 2026-07-05 | 0 | excluded from model days |
| 2026-07-14 | 2026-07-05 | +9 | `D+9` |

Current code keeps offsets:

```text
-5 to +20, excluding 0
```

### Success Definitions

Success map in `generate_strategy_dataset.py`:

| Channel | Success statuses |
|---|---|
| `SMS` | `DELIVERED`, `CLICKED`, `SENT` |
| `WH` | `DELIVERED`, `READ`, `CLICKED`, `SENT` |
| `VOICE` | `CONNECTED`, `CALL_CONNECTED` |
| `VOICE_BOT` | `CONNECTED`, `CALL_CONNECTED` |

Non-success statuses become failure counts.

### Feature Families

For each `(ENTITY_KEY, MONTH, DAY)`:

1. Intensity features:

```text
SMS_TOTAL_INTENSITY
WH_TOTAL_INTENSITY
VOICE_TOTAL_INTENSITY
VOICE_BOT_TOTAL_INTENSITY
```

These count all attempts.

2. Failure features:

```text
SMS_FAILED_ENGLISH
VOICE_FAILED_TAMIL
VOICE_BOT_FAILED_HINDI
```

These count non-success outcomes by channel/language.

3. Success features:

```text
SMS_SUCCESS_12PM_REGIONAL
VOICE_BOT_SUCCESS_10AM_ENGLISH
VOICE_SUCCESS_4PM_TAMIL
```

These count successful outcomes by channel/hour/language.

If `STRATEGY_USE_STATUS_WEIGHTS=true`, successful statuses can use configured weights:

```text
SENT=0.5, DELIVERED=1.0, CLICKED=2.0, READ=1.5
```

In current local sample, some feature values such as `0.4` show weighted/count-scaled preprocessing was used in data artifacts.

## Label Generation

Training label column:

```text
PREDICTED_STRATEGY
```

For each `(ENTITY_KEY, MONTH, DAY)`:

1. keep successful communication rows
2. build strategy string:

```text
COMM_TYPE-TIME_LABEL-LANGUAGE
```

3. aggregate success score/count per strategy
4. sort by:

```text
count desc, feature asc
```

5. top strategy becomes `PREDICTED_STRATEGY`

If no successful strategy exists for that entity/month/day, `PREDICTED_STRATEGY` is blank and later schedule table fills `-`.

## Monthly Feature Table

Script: `scripts/build_monthly_feature_dataset.py`

Input:

```text
data/training/strategy_training_dataset_*.csv
```

Output:

```text
data/features/strategy_monthly_features_*.csv
```

Logic:

```text
group by ENTITY_KEY, MONTH
sum all numeric feature columns across days
keep first RISK
keep first VERTICAL
```

So model does not receive separate day rows as input. Model receives month-level behavior summary.

## Schedule Target Table

Script: `scripts/build_strategy_schedule_dataset.py`

Input:

```text
day-level rows with PREDICTED_STRATEGY
```

Output:

```text
one row per ENTITY_KEY, MONTH
columns: D-5, D-4, ..., D, D+1, ..., D+20
```

Logic:

- pivot `DAY` values into columns
- convert blank labels to `-`
- force `D = -`
- convert `VOICE-...` labels to `IVR-...` in schedule output
- create `D-5__WEIGHT`, ..., `D+20__WEIGHT` sample-weight columns

## Rolling History Window

Function: `build_rolling_feature_windows(...)` in `scripts/pipeline_common.py`.

For each entity:

1. sort rows by month
2. deduplicate same entity/month
3. take last `history_window_months`
4. sum numeric features across that rolling window
5. keep current row risk/vertical
6. output row ending at source month

Example with `history_window_months=3`:

```text
FEB + MAR + APR features -> source row APR-2026 -> target MAY-2026
MAR + APR + MAY features -> source row MAY-2026 -> target JUN-2026
APR + MAY + JUN features -> source row JUN-2026 -> target JUL-2026
MAY + JUN + JUL features -> source row JUL-2026 -> predict AUG-2026
```

Model metadata decides actual window:

| Bundle | Day models | Feature columns | Window |
|---|---:|---:|---:|
| `next_month_strategy_catboost_3m.joblib` | 10 | 143 | 3 |
| `next_month_strategy_catboost_3m_20260710_v001.joblib` | 10 | 173 | 3 |
| `next_month_strategy_catboost_3m_v1.1.1.joblib` | 25 | 183 | 3 |
| `next_month_strategy_catboost_3m_v1.1.20.joblib` | 25 | 40 | 10 |

## Supervised Training Join

Function: `prepare_next_month_dataset(...)` in `scripts/pipeline_common.py`.

Logic:

```text
source features MONTH -> SOURCE_MONTH
TARGET_MONTH_PERIOD = SOURCE_MONTH_PERIOD + target_offset_months
join schedule rows on ENTITY_KEY + TARGET_MONTH_PERIOD
```

Default:

```text
target_offset_months = 1
```

So:

```text
rolled APR behavior -> MAY schedule labels
```

Each joined row is one supervised example:

```text
X = rolled monthly features ending in SOURCE_MONTH
y[D-5] = strategy in next-month schedule D-5
y[D-4] = strategy in next-month schedule D-4
...
```

## Feature Matrix Given To CatBoost

Function: `build_feature_matrix(...)`.

It drops:

```text
day target columns
day weight columns
TARGET_MONTH
TARGET_MONTH_PERIOD
TARGET_RISK
VERTICAL
ENTITY_KEY / APAC_CARD_NUMBER
SOURCE_MONTH_PERIOD
```

It one-hot encodes:

```text
SOURCE_MONTH
RISK
```

It fills nulls with `0`.

Therefore model input features are:

- rolled numeric behavior counts
- one-hot source month
- one-hot risk

Not used as features:

- account id
- raw EMI date
- raw collectable amount
- vertical in current code matrix

## CatBoost Training

Script: `scripts/train_next_month_strategy_model_catboost.py`

For each day in `DAY_COLUMNS`, code trains one classifier:

```text
CatBoostClassifier(
  loss_function="MultiClass",
  random_seed=42,
  iterations=<arg>,
  learning_rate=<arg>,
  depth=<arg>
)
```

Each day has its own:

- model
- `LabelEncoder`
- class list
- checkpoint

Bundle saved by `joblib`:

```text
{
  "models": {day: CatBoostClassifier or ConstantDayModel},
  "label_encoders": {day: LabelEncoder},
  "feature_columns": [...],
  "target_columns": [...],
  "target_offset_months": 1,
  "history_window_months": N
}
```

Fallback logic:

- if a day has no training target values, constant model predicts `-`
- if a day has one unique target only, constant model predicts that value

This prevents training failure on sparse days.

## Inference Logic

Script: `scripts/predict_next_month_strategy_catboost.py`

Steps:

1. load model bundle
2. read monthly feature file
3. read schedule file
4. rebuild rolling feature windows using bundle `history_window_months`
5. load digital cases base population
6. select latest history row per base-population entity up to source month
7. if history exists, score with CatBoost
8. if no history exists, output `-` for all campaign days
9. write prediction CSV

Risk-aware top-k output:

| Risk | Strategies returned per day |
|---|---:|
| `LOW` | top 1 |
| `MEDIUM` | top 2, pipe-delimited |
| `HIGH` | top 3, pipe-delimited |

Example:

```text
LOW    -> SMS-11AM-ENGLISH
MEDIUM -> SMS-11AM-ENGLISH|IVR-9AM-HINDI
HIGH   -> SMS-11AM-ENGLISH|-|IVR-6PM-HINDI
```

## Real Sample Trace: `XXXX079X4952`

Sample chosen from current local August artifacts:

```text
Loan/entity: XXXX079X4952
Source month: JUL-2026
Prediction month: AUG-2026
Prediction risk: LOW
```

### Raw Communication Rows

JUN-2026 sample rows:

| apac_card_number | comm_status | communication_type | verbiage_language | risk | emi_date | date | created_date |
|---|---|---|---|---|---|---|---|
| XXXX079X4952 | NO_ANSWER | VOICE_BOT | English | high | 2026-06-05 | 2026-05-31 | 2026-05-31 10:30:23.521 |
| XXXX079X4952 | DELIVERED | SMS | English | high | 2026-06-05 | 2026-06-01 | 2026-06-01 12:30:05.720 |
| XXXX079X4952 | DELIVERED | SMS | Regional | high | 2026-06-05 | 2026-05-31 | 2026-05-31 12:30:22.982 |
| XXXX079X4952 | NO_ANSWER | VOICE | Tamil | high | 2026-06-05 | 2026-05-31 | 2026-05-31 15:00:20.300 |
| XXXX079X4952 | CONNECTED | VOICE_BOT | English | high | 2026-06-05 | 2026-06-01 | 2026-06-01 10:00:14.548 |

JUL-2026 sample rows:

| apac_card_number | comm_status | communication_type | verbiage_language | risk | emi_date | date | created_date |
|---|---|---|---|---|---|---|---|
| XXXX079X4952 | DELIVERED | SMS | Regional | low | 2026-07-05 | 2026-07-01 | 2026-07-01 15:00:05.876 |
| XXXX079X4952 | NO_ANSWER | VOICE | Tamil | low | 2026-07-05 | 2026-06-30 | 2026-06-30 11:02:59.562 |
| XXXX079X4952 | DELIVERED | SMS | Regional | low | 2026-07-05 | 2026-06-30 | 2026-06-30 15:00:02.339 |
| XXXX079X4952 | CONNECTED | VOICE_BOT | English | low | 2026-07-05 | 2026-07-02 | 2026-07-02 11:00:10.643 |
| XXXX079X4952 | CONNECTED | VOICE | Tamil | low | 2026-07-05 | 2026-07-02 | 2026-07-02 16:00:11.184 |

### Prediction-Month Case Row

From `data/cases/digital_cases_AUG2026.csv`:

| apac_card_number | risk | vertical | collectable_amount | emi_date |
|---|---|---|---:|---|
| XXXX079X4952 | low | two wheeler loan | NaN | 05/08/2026 |

### Day-Level Feature Rows

Selected rows from `strategy_training_dataset_inference.csv`:

| ENTITY_KEY | MONTH | DAY | RISK | SMS_TOTAL_INTENSITY | VOICE_TOTAL_INTENSITY | VOICE_BOT_TOTAL_INTENSITY | PREDICTED_STRATEGY |
|---|---|---|---|---:|---:|---:|---|
| XXXX079X4952 | JUN-2026 | D-5 | HIGH | 2 | 2 | 1 | VOICE-9AM-TAMIL |
| XXXX079X4952 | JUN-2026 | D-4 | HIGH | 2 | 2 | 1 | VOICE_BOT-10AM-ENGLISH |
| XXXX079X4952 | JUN-2026 | D-3 | HIGH | 1 | 2 | 2 | SMS-5PM-REGIONAL |
| XXXX079X4952 | JUN-2026 | D-2 | HIGH | 2 | 2 | 2 | SMS-12PM-REGIONAL |
| XXXX079X4952 | JUN-2026 | D-1 | HIGH | 2 | 1 | 2 | SMS-5PM-ENGLISH |
| XXXX079X4952 | JUL-2026 | D-5 | LOW | 1 | 2 | 0 | SMS-3PM-REGIONAL |
| XXXX079X4952 | JUL-2026 | D-4 | LOW | 1 | 1 | 2 | SMS-3PM-REGIONAL |
| XXXX079X4952 | JUL-2026 | D-3 | LOW | 1 | 1 | 1 | VOICE-4PM-TAMIL |
| XXXX079X4952 | JUL-2026 | D-2 | LOW | 1 | 1 | 2 | VOICE_BOT-3PM-ENGLISH |

Interpretation:

- raw `VOICE` successful label becomes `VOICE-...` at day-feature stage
- schedule builder later rewrites `VOICE-...` to `IVR-...`
- blank/no-success days later become `-`

### Monthly Feature Rows

From `strategy_monthly_features_inference.csv`:

| ENTITY_KEY | MONTH | RISK | SMS_TOTAL_INTENSITY | VOICE_TOTAL_INTENSITY | VOICE_BOT_TOTAL_INTENSITY | SMS_SUCCESS_12PM_REGIONAL | VOICE_BOT_SUCCESS_10AM_ENGLISH | VOICE_SUCCESS_4PM_TAMIL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| XXXX079X4952 | FEB-2026 | HIGH | 11 | 12 | 2 | 0.0 | 0 | 0 |
| XXXX079X4952 | MAR-2026 | HIGH | 11 | 12 | 2 | 0.0 | 0 | 0 |
| XXXX079X4952 | JUN-2026 | HIGH | 9 | 9 | 8 | 0.8 | 1 | 0 |
| XXXX079X4952 | JUL-2026 | LOW | 4 | 5 | 5 | 0.8 | 0 | 1 |

For a 3-month source row ending JUL-2026, numeric model features are sum of latest available months in window, then one-hot `SOURCE_MONTH_JUL-2026` and `RISK_LOW` are added.

### Schedule Rows

From `strategy_schedule_dataset_inference.csv`:

| RISK | ENTITY_KEY | MONTH | D-5 | D-4 | D-3 | D-2 | D-1 | D |
|---|---|---|---|---|---|---|---|---|
| HIGH | XXXX079X4952 | JUN-2026 | IVR-9AM-TAMIL | VOICE_BOT-10AM-ENGLISH | SMS-5PM-REGIONAL | SMS-12PM-REGIONAL | SMS-5PM-ENGLISH | - |
| LOW | XXXX079X4952 | JUL-2026 | SMS-3PM-REGIONAL | SMS-3PM-REGIONAL | IVR-4PM-TAMIL | VOICE_BOT-3PM-ENGLISH | - | - |

This table is used as target during training, and as historical context during inference prep.

### Final Prediction Output

From `artifacts/predictions/2026_08_strategy_predictions_catboost_3m.csv`:

| SOURCE_RISK | Loan_number | SOURCE_MONTH_USED | MONTH | D-5 | D-4 | D-3 | D-2 | D-1 | D | D+1 |
|---|---|---|---|---|---|---|---|---|---|---|
| LOW | XXXX079X4952 | JUL-2026 | AUG-2026 | IVR-3PM-TAMIL | VOICE_BOT-10AM-ENGLISH | IVR-4PM-TAMIL | VOICE_BOT-10AM-ENGLISH | VOICE_BOT-10AM-ENGLISH | - | - |

Because risk is `LOW`, each day returns one top strategy.

### Evidence Rows

From `artifacts/predictions/2026_08_strategy_predictions_catboost_3m_prediction_evidence.csv`.

Only rows where top-1 aligns with prediction CSV are shown here, because evidence files can be regenerated with a different model artifact/version:

| day | recommendedStrategy1 | prob1 | recommendedStrategy2 | prob2 | recommendedStrategy3 | prob3 | historical_successfeature1 | historical_successprecentage1 |
|---|---|---:|---|---:|---|---:|---|---:|
| D-4 | VOICE_BOT-10AM-ENGLISH | 0.320924 | IVR-4PM-TAMIL | 0.198329 | VOICE_BOT-2PM-ENGLISH | 0.165752 | VOICE_BOT_SUCCESS_10AM_ENGLISH | 11.11 |
| D-2 | VOICE_BOT-10AM-ENGLISH | 0.360103 | IVR-9AM-TAMIL | 0.130133 | SMS-12PM-ENGLISH | 0.112387 | VOICE_BOT_SUCCESS_3PM_ENGLISH | 10.00 |
| D-1 | VOICE_BOT-10AM-ENGLISH | 0.302278 | IVR-12PM-TAMIL | 0.285820 | - | 0.129467 | SMS_SUCCESS_9AM_REGIONAL | 8.00 |

Evidence file gives probability rankings and historical supporting feature summaries. Prediction CSV remains authoritative scheduler input.

## Campaign Scheduler Conversion

After prediction CSV, `run_monthly_inference_pipeline.py` builds two staging outputs:

```text
digital_collections.ai_ml_campaign_recommendations
digital_collections.ai_ml_campaign_mapping
```

### Strategy Parsing

Function: `_parse_strategy(...)`.

Rules:

- ignore `-`
- split label into channel/hour/language
- map channel to scheduler mode
- ignore `REGIONAL` language for campaign scheduler rows
- convert hour to `HH:MM:SS`

Channel mapping:

| Prediction label channel | Scheduler mode |
|---|---|
| `SMS` | `SMS` |
| `WH` / `WHATSAPP` | `WHATSAPP` |
| `IVR` / `VOICE` | `VOICE` |
| `VOICE_BOT` | `VOICE_BOT` |

Important: `REGIONAL` recommendations are present in model output but excluded from MCollect scheduler campaign rows by `_parse_strategy(...)`.

### Campaign Bucket

Function: `_due_bucket(...)`.

| Day | campaign_type | due_type |
|---|---|---|
| `D-5..D-1` | `PRE` | `PREDUE` |
| `D` | `DUE` | `DUEDATE` |
| `D+1..D+20` | `POST` | `POSTDUE` |

### Name Formats

Scheduler name:

```text
{due_type}_AIML_{channel}_{vertical}_{language}_{risk_code}_{emi_cycle}TH_{vendor}_{run_token}
```

Example:

```text
PREDUE_AIML_SMS_LAP_ENGLISH_MR_5TH_KALEYRA_200526
```

Template name:

```text
{due_type}_AIML_{channel}_{language}
```

Example:

```text
PREDUE_AIML_SMS_ENGLISH
```

Dataset display name:

```text
{due_type} AIML {channel} {vertical} {language} {risk_code} EMI {emi_cycle}TH {day_tokens} {hour_tokens}
```

Example:

```text
PREDUE AIML SMS LAP ENGLISH MR EMI 5TH DM4-DM2 09
```

### Mapping Rows

Campaign mapping rows connect generated scheduler campaign to individual loan numbers:

| Column | Meaning |
|---|---|
| `campaign_name` | generated scheduler name |
| `loan_number` | predicted account |
| `mode` | SMS/WHATSAPP/VOICE/VOICE_BOT |
| `date` | EMI-relative day label(s) |
| `time` | send time(s) |
| `vendor` | resolved vendor |
| `language` | normalized language |
| `risk` | risk code `LR/MR/HR` |
| `vertical` | business vertical |
| `prediction_reason` | day-level reason text from prediction JSON when present |

## Monthly Pipeline Flow

Main command:

```bash
./venv/bin/python scripts/run_monthly_inference_pipeline.py \
  --source-month 2026-07 \
  --predict-month 2026-08 \
  --model catboost_3m
```

High-level flow:

1. resolve source/prediction month
2. fetch source-month communication rows from Postgres
3. fetch prediction-month digital cases
4. select history files for configured window
5. generate day-level features
6. build monthly features
7. build schedule table
8. load local or MLflow model bundle
9. run CatBoost inference
10. generate prediction evidence
11. compute drift report
12. store prediction snapshots to Postgres unless `--skip-db-store`
13. build campaign recommendation rows
14. build campaign mapping rows
15. store campaign rows/mappings to Postgres unless `--skip-db-store`
16. write prediction summary JSON

## Output Tables

### Prediction Snapshot Table

Table:

```text
digital_collections.ai_ml_recommendations_data
```

Key fields:

| Column | Content |
|---|---|
| `loan_number` | account |
| `source_month_used` | historical source month |
| `prediction_month` | next month |
| `model_name` | model selection, e.g. `catboost_3m` |
| `source_risk` | input risk |
| `prediction_payload` | JSON day -> strategy |
| `prediction_reason` | JSON day -> business reason |

### Campaign Recommendation Table

Table:

```text
digital_collections.ai_ml_campaign_recommendations
```

Contains unique scheduler campaign definitions: name, mode, date, time, template, dataset, vendor, source month, prediction month, risk, vertical.

### Campaign Mapping Table

Table:

```text
digital_collections.ai_ml_campaign_mapping
```

Contains account-level assignments to campaigns.

## Key Limitations

- Model learns historical strategy assignment/communication success patterns, not direct payment/cure uplift.
- Account id is not used as feature, so model generalizes by behavior profile.
- `collectable_amount` exists in cases but is not currently used by `build_feature_matrix(...)`.
- `VERTICAL` is loaded and carried, but dropped from final model matrix in current code.
- If model bundle day coverage and current code day coverage differ, inference must use compatible artifact/version.
- Evidence CSV must be regenerated from same model/prediction run if probabilities must exactly match prediction CSV.

## Mental Model

End-to-end:

```text
raw communications
  -> normalize channel/status/language/risk/vertical
  -> compute EMI-relative day + hour bucket
  -> count intensity/failure/success features per entity/month/day
  -> choose winning historical strategy per entity/month/day
  -> aggregate numeric behavior to entity/month
  -> pivot winning strategies to schedule labels
  -> roll latest N months into source feature row
  -> join next month schedule as target
  -> train one CatBoost classifier per day
  -> score current digital cases
  -> return risk-aware top-k campaign labels
  -> convert campaign labels to scheduler campaign + mapping rows
```

