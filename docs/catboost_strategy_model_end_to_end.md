# CatBoost Campaign Recommendation Model: End-to-End Flow

## 1. System Overview

The campaign recommendation engine predicts the best EMI-cycle communication strategy for each loan/customer account.

The model answers this business question:

> Based on this customer's recent communication history, what is the best communication action for each EMI-relative day?

The output is generated for each account across the configured campaign window:

- Pre-due days: `D-5` to `D-1`
- Due date: excluded from prediction
- Post-due days: `D+1` to `D+20`

Each day can receive a recommendation such as:

- `SMS-11AM-ENGLISH`
- `WH-4PM-HINDI`
- `VOICE-2PM-ENGLISH`
- `VOICE_BOT-10AM-REGIONAL`
- `-`, meaning no campaign is recommended

The active model design is:

> Whole-month historical behavior -> day-wise next-month strategy prediction.

That means the model does not train on one row per account per D-day as model input. Instead, it builds one monthly feature row per account/source month, creates rolling monthly lag features, and trains a separate CatBoost classifier for each campaign day.

## 2. Architecture Flowchart

```text
data_config
  |
  |-- upload.scheduler.emi-dates
  |      |
  |      |-- derives prediction month from full EMI date
  |      |-- derives source month as prediction month - 1
  |
  v
Postgres source tables
  |
  |-- communications / configured communication table
  |-- digital_cases / configured cases table
  |
  v
Raw CSV extracts
  |
  |-- data/communication/comm_data_<MMM><YYYY>_<cycle_day>.csv
  |-- data/cases/digital_cases_<MMM><YYYY>.csv
  |
  v
Day-level strategy dataset
  |
  |-- data/training/strategy_training_dataset_train.csv
  |-- one row per account + EMI month + relative day
  |
  v
Monthly feature dataset
  |
  |-- data/features/strategy_monthly_features_train.csv
  |-- one row per account + month
  |
  v
Schedule target dataset
  |
  |-- data/schedules/strategy_schedule_dataset_all_months.csv
  |-- one row per account + month with D-5 ... D+20 target columns
  |
  v
Rolling model input
  |
  |-- feature_M1, feature_M2, feature_M3 ... feature_Mn
  |-- M1 is latest/source month, M2 is previous month, etc.
  |
  v
Training
  |
  |-- separate CatBoost classifier per day
  |-- D-5 model, D-4 model, ... D+20 model
  |
  v
Versioned model artifacts
  |
  |-- artifacts/models/next_month_strategy_catboost_3m_v<version>.joblib
  |-- artifacts/metrics/next_month_strategy_catboost_3m_metrics_v<version>.json
  |
  v
Inference
  |
  |-- loads latest successful trained model version
  |-- builds current account feature matrix
  |-- predicts strategy for each D-day
  |
  v
Business outputs
  |
  |-- strategy prediction CSV
  |-- prediction evidence CSV
  |-- ai_ml_campaign_recommendations
  |-- ai_ml_campaign_mapping
  |-- ai_configurations status/metrics
  |-- api_audit_log request/response audit
```

## 3. Data Pipeline and Transformation

### 3.1 EMI Date and Month Selection

The pipeline uses the configured EMI dates from `data_config`.

Important key:

```text
upload.scheduler.emi-dates
```

The value is expected as full dates, not only day numbers.

Example:

```text
05/09/2026,15/09/2026
```

For each EMI date:

- Prediction month is the month of the EMI date.
- Source month is one month before the EMI date month.
- Cases are fetched for the prediction month.
- Communication history is fetched up to the source month.

Example:

```text
Configured EMI date: 05/09/2026
Prediction month: SEP-2026
Source month: AUG-2026
Current cases: September 2026 digital cases
Communication history: months ending August 2026
```

If `history_window_months = 3`, the model input for September prediction uses:

```text
M1 = AUG-2026
M2 = JUL-2026
M3 = JUN-2026
```

If `history_window_months = 6`, the model input for September prediction uses:

```text
M1 = AUG-2026
M2 = JUL-2026
M3 = JUN-2026
M4 = MAY-2026
M5 = APR-2026
M6 = MAR-2026
```

### 3.2 Raw Communication Fetch

Communication data is fetched from the configured Postgres communication table.

The output file naming format is:

```text
data/communication/comm_data_<MMM><YYYY>_<cycle_day>.csv
```

Example:

```text
data/communication/comm_data_AUG2026_5.csv
```

This avoids overwriting files when the same month has multiple EMI cycles, such as the 5th and 15th.

The communication extract includes fields such as:

```csv
apac_card_number,party_id,communication_type,comm_status,verbiage_language,vertical,risk,emi_date,date,created_date
XXXX100X5124,P1001,SMS,DELIVERED,English,LAP,LOW,05/09/2026,31/08/2026,2026-08-31 11:10:00
XXXX100X5124,P1001,VOICE_BOT,CONNECTED,Hindi,LAP,LOW,05/09/2026,10/09/2026,2026-09-10 16:05:00
```

If multi-APAC mode is enabled, `party_id` becomes the entity key so multiple APACs for the same customer can share communication history. If it is disabled, `apac_card_number` remains the entity key.

### 3.3 Raw Case Fetch

Current cases are fetched from the configured cases table for the prediction month.

Example:

```text
Configured EMI date: 05/09/2026
Cases fetched for: SEP-2026
```

The case file is used as the prediction population. If an account is not in the current cases file, it cannot appear in the final inference output for that run.

## 4. Strategy and Feature Engineering

### 4.1 Channel Normalization

The pipeline normalizes communication types into model channels.

```text
SMS       -> SMS
WHATSAPP  -> WH
VOICE     -> VOICE
VOICE_BOT -> VOICE_BOT
```

Only normalized channels are used in strategy features and predictions.

### 4.2 Success Status Logic

The model converts communication statuses into success or failure signals.

Current success status mapping:

```text
SMS success:       SENT, DELIVERED, CLICKED
WH success:        SENT, DELIVERED, READ, CLICKED
VOICE success:     CONNECTED, CALL_CONNECTED
VOICE_BOT success: CONNECTED, CALL_CONNECTED
```

Any communication row that belongs to a valid channel but does not match the success list becomes a failure signal.

There is no automatic extra model importance given to `CLICKED` unless status weighting is enabled through environment configuration.

### 4.3 Optional Status Weighting

Weighting is controlled by:

```text
STRATEGY_USE_STATUS_WEIGHTS
STRATEGY_SUCCESS_SCORES_JSON
STRATEGY_SAMPLE_WEIGHT_POSITIVE_BOOST
```

When weighting is disabled:

```text
successful row score = 1.0
failed row score = 0.0
training sample weight = 1.0
```

When weighting is enabled, statuses can carry different scores.

Default score examples:

```text
SMS SENT       = 0.5
SMS DELIVERED  = 1.0
SMS CLICKED    = 2.0
WH READ        = 1.5
WH CLICKED     = 2.0
VOICE CONNECTED = 1.0
VOICE_BOT CONNECTED = 1.0
```

The model still receives the same feature columns, but each target training example can receive a larger sample weight when the selected historical strategy has stronger successful behavior.

### 4.4 Relative Day Calculation

For every communication row:

```text
relative_day = communication_date - emi_date
```

Examples:

```text
communication_date = 31/08/2026
emi_date = 05/09/2026
relative_day = -5
day label = D-5
```

```text
communication_date = 10/09/2026
emi_date = 05/09/2026
relative_day = +5
day label = D+5
```

Rows are retained only if:

```text
relative_day is between -5 and +20
relative_day is not 0
```

So the active day scope is:

```text
D-5, D-4, D-3, D-2, D-1,
D+1, D+2, ..., D+20
```

### 4.5 Day-Level Feature Dataset

The day-level dataset is created at this grain:

```text
entity/account + EMI month + relative day
```

Example input rows:

```csv
apac_card_number,communication_type,comm_status,verbiage_language,emi_date,date,created_date,risk,vertical
MOB-TEST-AI,SMS,DELIVERED,English,05/09/2026,31/08/2026,2026-08-31 11:10:00,MEDIUM,LAP
MOB-TEST-AI,SMS,FAILED,English,05/09/2026,31/08/2026,2026-08-31 12:20:00,MEDIUM,LAP
MOB-TEST-AI,WHATSAPP,READ,Hindi,05/09/2026,07/09/2026,2026-09-07 09:05:00,MEDIUM,LAP
```

Generated day-level row examples:

| ENTITY_KEY | MONTH | DAY | RISK | VERTICAL | SMS_SUCCESS_11AM_ENGLISH | SMS_FAILED_ENGLISH | SMS_TOTAL_INTENSITY | PREDICTED_STRATEGY |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| MOB-TEST-AI | SEP-2026 | D-5 | MEDIUM | LAP | 1 | 1 | 2 | SMS-11AM-ENGLISH |
| MOB-TEST-AI | SEP-2026 | D+2 | MEDIUM | LAP | 0 | 0 | 0 | WH-9AM-HINDI |

### 4.6 Feature Column Families

The model feature set is built from these families:

```text
<CHANNEL>_SUCCESS_<TIME>_<LANGUAGE>
<CHANNEL>_FAILED_<LANGUAGE>
<CHANNEL>_TOTAL_INTENSITY
RISK
SOURCE_MONTH
```

Examples:

```text
SMS_SUCCESS_9AM_ENGLISH
SMS_SUCCESS_12PM_REGIONAL
WH_SUCCESS_4PM_HINDI
VOICE_SUCCESS_2PM_ENGLISH
VOICE_BOT_SUCCESS_10AM_HINDI
SMS_FAILED_ENGLISH
WH_FAILED_REGIONAL
VOICE_TOTAL_INTENSITY
VOICE_BOT_TOTAL_INTENSITY
```

The success matrix captures which channel, time, and language worked historically.

The failure matrix captures unsuccessful attempts by channel and language.

The total intensity columns capture communication volume regardless of success or failure.

### 4.7 Target Strategy Creation

For each account/month/day, the target strategy is selected from the strongest successful historical signal for that same day.

Example:

```text
SMS_SUCCESS_11AM_ENGLISH = 5
SMS_SUCCESS_4PM_ENGLISH = 2
WH_SUCCESS_9AM_HINDI = 1
```

Target:

```text
PREDICTED_STRATEGY = SMS-11AM-ENGLISH
```

If there is no usable successful signal, the target can become:

```text
PREDICTED_STRATEGY = -
```

This means historical behavior suggests no campaign is the safest/default action for that day.

## 5. Monthly Roll-Up and Model Input

### 5.1 Monthly Feature Roll-Up

The day-level strategy dataset is rolled up into a monthly feature table.

Grain:

```text
entity/account + month
```

Calculation:

```text
monthly feature value = sum of all day-level values in that month
```

Example day-level rows:

```csv
ENTITY_KEY,MONTH,DAY,SMS_SUCCESS_11AM_ENGLISH,SMS_FAILED_ENGLISH,SMS_TOTAL_INTENSITY
MOB-TEST-AI,JUN-2026,D-5,2,1,3
MOB-TEST-AI,JUN-2026,D-2,1,0,1
MOB-TEST-AI,JUN-2026,D+5,0,2,2
```

Monthly roll-up:

```csv
ENTITY_KEY,MONTH,SMS_SUCCESS_11AM_ENGLISH,SMS_FAILED_ENGLISH,SMS_TOTAL_INTENSITY
MOB-TEST-AI,JUN-2026,3,3,6
```

This is what "rolled up" means in the active logic: all eligible communication behavior for that account and month is summed into a single monthly feature row.

### 5.2 Rolling History Window

After monthly roll-up, the pipeline creates lagged history columns.

If `history_window_months = 3`, each monthly feature becomes:

```text
<feature>_M1 = current/source month value
<feature>_M2 = previous month value
<feature>_M3 = two months before source month value
```

Example for predicting July 2026 using source month June 2026:

```csv
ENTITY_KEY,SOURCE_MONTH,SMS_SUCCESS_11AM_ENGLISH_M1,SMS_SUCCESS_11AM_ENGLISH_M2,SMS_SUCCESS_11AM_ENGLISH_M3
MOB-TEST-AI,JUN-2026,3,5,1
```

Meaning:

```text
M1 = JUN-2026
M2 = MAY-2026
M3 = APR-2026
```

If `history_window_months = 6`, the same feature extends to:

```text
M1, M2, M3, M4, M5, M6
```

### 5.3 What Goes Into the Model

The model receives numeric and encoded behavior columns.

Fed into CatBoost:

```text
SMS_SUCCESS_9AM_ENGLISH_M1
SMS_SUCCESS_9AM_ENGLISH_M2
SMS_SUCCESS_9AM_ENGLISH_M3
SMS_FAILED_ENGLISH_M1
WH_SUCCESS_4PM_HINDI_M1
VOICE_TOTAL_INTENSITY_M1
VOICE_BOT_TOTAL_INTENSITY_M1
SOURCE_MONTH_JUN-2026
RISK_LOW
RISK_MEDIUM
RISK_HIGH
```

Not fed as learning features:

```text
ENTITY_KEY / APAC_CARD_NUMBER
target day columns
target weight columns
TARGET_MONTH
TARGET_MONTH_PERIOD
VERTICAL
```

The account identifier is used only for joining and tracking, not as a predictive model feature.

## 6. Schedule Target Dataset

The schedule dataset converts day-level targets into a wide table.

Grain:

```text
entity/account + month
```

Example:

```csv
ENTITY_KEY,MONTH,RISK,D-5,D-4,D-3,D-2,D-1,D+1,D+2,D+3,D+4,D+5
MOB-TEST-AI,JUL-2026,MEDIUM,SMS-11AM-ENGLISH,-,WH-9AM-HINDI,-,SMS-5PM-ENGLISH,SMS-4PM-ENGLISH,-,-,VOICE-2PM-HINDI,-
```

Each day column becomes the target for one day-specific model.

For example:

```text
D-5 CatBoost model target = D-5 column
D-4 CatBoost model target = D-4 column
D+20 CatBoost model target = D+20 column
```

## 7. Month Splits for Training

### 7.1 Target Offset

The model uses:

```text
target_offset_months = 1
```

Meaning:

```text
source month behavior predicts next month strategy
```

Example:

```text
JUN-2026 behavior -> JUL-2026 campaign strategy
```

### 7.2 Automatic Split Logic

If explicit train/validation/test months are not provided, the training script uses source months that have a next-month target.

Automatic split:

```text
train = all targetable source months except last two
validation = second-last targetable source month
test = last targetable source month
prediction_candidates = latest source month in the full dataset
```

### 7.3 Example with 6 Months

Suppose the available source months are:

```text
JAN-2026, FEB-2026, MAR-2026, APR-2026, MAY-2026, JUN-2026
```

Because `target_offset_months = 1`, the latest source month `JUN-2026` predicts `JUL-2026`.

If `JUL-2026` targets are not available yet, `JUN-2026` is used as prediction candidate, not as supervised training.

The effective training split can look like:

```text
Training source months: JAN-2026, FEB-2026, MAR-2026, APR-2026
Validation source month: MAY-2026
Test source month: may be empty or only present when target rows exist
Prediction source month: JUN-2026
```

In local sample artifacts, the current training model input file has:

```text
data/features/strategy_model_input_train.csv
Rows: 61,379
Columns: 1,118
```

These numbers are environment-specific and change with data volume, configured months, enabled channels, languages, and history window size.

## 8. Model Training

### 8.1 Model Type

The active model is CatBoost.

Business explanation:

> CatBoost is a machine learning algorithm that learns patterns from historical communication behavior and predicts which strategy is most likely to be suitable next.

Technical explanation:

> The pipeline trains one multiclass CatBoost classifier per EMI-relative day. Each classifier predicts one label from possible communication strategies for that day.

### 8.2 Why One Model Per Day

Customer behavior can be different before and after EMI date.

Examples:

```text
D-5 may work better for light SMS reminders.
D-1 may work better for final reminder messages.
D+5 may work better for stronger follow-up.
D+11 may work better for voice or voice-bot contact.
```

So the system trains separate models:

```text
D-5 model
D-4 model
D-3 model
D-2 model
D-1 model
D+1 model
...
D+20 model
```

Each model receives the same account-level rolled history features, but learns a different target day behavior.

### 8.3 Training Input Example

For one account, the model input might look like:

```csv
ENTITY_KEY,SOURCE_MONTH,RISK_LOW,SMS_SUCCESS_11AM_ENGLISH_M1,SMS_SUCCESS_11AM_ENGLISH_M2,SMS_SUCCESS_11AM_ENGLISH_M3,SMS_FAILED_ENGLISH_M1,WH_SUCCESS_9AM_HINDI_M1,VOICE_BOT_TOTAL_INTENSITY_M1
MOB-TEST-AI,JUN-2026,0,3,5,1,2,1,4
```

The target columns are separate:

```csv
ENTITY_KEY,SOURCE_MONTH,TARGET_MONTH,D-5,D-4,D-3,D+1,D+2
MOB-TEST-AI,JUN-2026,JUL-2026,SMS-11AM-ENGLISH,-,WH-9AM-HINDI,SMS-4PM-ENGLISH,-
```

The `D-5` model sees:

```text
X = monthly lag features from JUN, MAY, APR
y = SMS-11AM-ENGLISH
```

The `D-4` model sees:

```text
X = same monthly lag features
y = -
```

### 8.4 Fallback Logic During Training

Training does not fail when one day has insufficient variation.

Fallback rules:

```text
If a day has no target data:
    train a constant fallback model that predicts '-'

If a day has only one unique target and that target is '-':
    train a constant fallback model that predicts '-'

If a day has only one unique non-blank target:
    train a constant fallback model that always predicts that target
```

This prevents errors such as:

```text
ValueError: Training target for D-1 contains only one unique value
```

The logs highlight fallback activation for the affected day.

### 8.5 Training Metrics

The training script evaluates:

```text
Train metrics
Validation metrics
Test metrics
```

Metrics include:

```text
rows
exact_match_accuracy
average_day_accuracy
per_day_accuracy
```

Meaning:

- `rows`: number of accounts/source-month rows evaluated.
- `exact_match_accuracy`: percentage of rows where the model predicted all day outputs correctly.
- `average_day_accuracy`: average accuracy across day-specific models.
- `per_day_accuracy`: accuracy for each individual day model, such as `D-5`, `D+10`, `D+20`.

If `test rows = 0`, it means the selected month split did not produce a usable test period with known target rows. This is not always a model failure; it usually means the data window does not contain enough future target months for a separate test slice.

## 9. Model Versioning

Training creates versioned artifacts.

Model file:

```text
artifacts/models/next_month_strategy_catboost_3m_v<model_version>.joblib
```

Metrics file:

```text
artifacts/metrics/next_month_strategy_catboost_3m_metrics_v<model_version>.json
```

Example:

```text
next_month_strategy_catboost_3m_v1.1.3.joblib
next_month_strategy_catboost_3m_metrics_v1.1.3.json
```

The `ai_configurations` table is used for model version tracking.

Expected behavior:

- If training succeeds, the new incremented model version is saved and recorded.
- If training fails, the model version should remain at the latest successful version.
- Inference loads the latest successful training version.
- If no successful training version exists, the pipeline can fall back to the configured base/default model.

## 10. Inference Flow

### 10.1 Inference Month Selection

Inference also uses the full EMI date from `data_config`.

Example:

```text
upload.scheduler.emi-dates = 05/09/2026
```

Then:

```text
Prediction month = SEP-2026
Source month = AUG-2026
Cases fetched = SEP-2026
Communication history used = months ending AUG-2026
```

### 10.2 Inference Model Input

The inference pipeline builds the same type of model input as training.

For a 3-month history window predicting September:

```text
M1 = AUG-2026
M2 = JUL-2026
M3 = JUN-2026
```

Example row:

```csv
ENTITY_KEY,SOURCE_MONTH,RISK_LOW,SMS_SUCCESS_11AM_ENGLISH_M1,SMS_SUCCESS_11AM_ENGLISH_M2,SMS_SUCCESS_11AM_ENGLISH_M3,VOICE_BOT_TOTAL_INTENSITY_M1
XXXX100X5124,AUG-2026,1,2,1,0,3
```

The same row is passed into each day-specific model.

### 10.3 Prediction Mechanics

For each account:

```text
Load latest model bundle.
Build the current feature row.
Run D-5 model -> predict D-5 strategy.
Run D-4 model -> predict D-4 strategy.
Run D+1 model -> predict D+1 strategy.
Continue through D+20.
```

Each day model produces class probabilities.

Example for `D-5`:

```text
SMS-11AM-ENGLISH = 0.62
-                = 0.21
VOICE_BOT-4PM-HINDI = 0.10
WH-9AM-HINDI = 0.07
```

The highest probability becomes the primary recommendation.

### 10.4 Risk-Based Top-K Selection

The number of strategies retained depends on risk.

```text
LOW risk    -> top 1 strategy
MEDIUM risk -> top 2 strategies
HIGH risk   -> top 3 strategies
```

Example:

```text
LOW risk D-5 output:
SMS-11AM-ENGLISH
```

```text
HIGH risk D-5 output:
SMS-11AM-ENGLISH|VOICE_BOT-4PM-HINDI|WH-9AM-HINDI
```

If the top label is `-` for LOW risk, only `-` is retained. Alternate strategies are not retained for LOW risk because the top-k policy is one.

### 10.5 Final Prediction CSV

The prediction file keeps the current naming pattern:

```text
artifacts/predictions/<YYYY>_<MM>_strategy_predictions_catboost_3m.csv
```

Example:

```text
artifacts/predictions/2026_09_strategy_predictions_catboost_3m.csv
```

Local sample shape:

```text
Rows: 9,248
Columns: 33
```

Example output:

```csv
SOURCE_RISK,SOURCE_VERTICAL,EMI_DATE,Loan_number,SOURCE_MONTH_USED,MONTH,D-5,D-4,D-3,D+1,PREDICTION_REASON
LOW,LAP,05/09/2026,XXXX100X5124,AUG-2026,SEP-2026,-,SMS-11AM-ENGLISH,VOICE_BOT-4PM-HINDI,SMS-9AM-REGIONAL,"At this stage, no campaign is recommended..."
```

Rows with `-` are valid predictions. They mean the model selected no campaign as the best action for that account/day.

## 11. Prediction Evidence File

The evidence file explains why recommendations were generated.

File naming:

```text
<YYYY>_<MM>_strategy_predictions_catboost_3m_prediction_evidence.csv
```

Example:

```text
2026_09_strategy_predictions_catboost_3m_prediction_evidence.csv
```

The evidence file is generated from:

- the prediction CSV
- the trained model probabilities
- the same communication-derived strategy features
- the current cases population

The evidence rows are at this grain:

```text
loan/account + prediction month + day
```

Typical columns:

```text
loanNumber
sourceMonthUsed
predictionMonth
day
risk
vertical
recommendedStrategy1
recommendedStrategy1_SuccessProbability
recommendedStrategy2
recommendedStrategy2_SuccessProbability
recommendedStrategy3
recommendedStrategy3_SuccessProbability
historicalSuccessFeature1
historical_successprecentage1
historicalSuccessFeature2
historical_successprecentage2
historicalSuccessFeature3
historical_successprecentage3
SMS_SUCCESS_9AM_ENGLISH
SMS_SUCCESS_10AM_ENGLISH
WH_SUCCESS_4PM_HINDI
VOICE_SUCCESS_2PM_ENGLISH
VOICE_BOT_SUCCESS_10AM_REGIONAL
SMS_TOTAL_INTENSITY
MaxCount_SMS
WH_TOTAL_INTENSITY
MaxCount_WH
VOICE_TOTAL_INTENSITY
MaxCount_VOICE
VOICE_BOT_TOTAL_INTENSITY
MaxCount_VOICE_BOT
recommendedActionText
predictionReason
businessNarrative
fallbackReason
```

The evidence file may be limited by configuration for testing.

Example:

```text
PREDICTION_EVIDENCE_LIMIT=50
```

With 25 prediction days, that produces:

```text
50 accounts * 25 days = 1,250 evidence rows
```

If the limit is removed or set to a full-run value, the evidence generation can become slower because it must create account-day proof rows across the full prediction population.

## 12. Drift Calculation Logic

### 12.1 What Drift Measures

Drift checks whether the current inference input data looks different from the model's training input data.

It does not directly measure prediction accuracy.

It answers:

> Is today's input population statistically different from what the model saw during training?

### 12.2 Active Drift Baseline

The active drift logic uses a pooled full-training baseline.

Baseline:

```text
All training source months recorded in the model metrics metadata
```

Inference:

```text
Current inference source month model input
```

This is better than comparing against only one training month because it reduces false drift caused by a single unusual month.

### 12.3 PSI Formula

Drift is calculated using Population Stability Index.

For each feature:

```text
PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
```

Where:

```text
expected_pct = distribution in pooled training baseline
actual_pct = distribution in current inference data
```

Numeric features are bucketed before comparison.

Categorical and low-cardinality numeric features are compared as categories.

### 12.4 Drift Severity

Feature-level PSI severity:

```text
PSI < 0.10  -> stable
PSI >= 0.10 -> moderate
PSI >= 0.25 -> severe
```

Overall drift fields:

```text
overall_psi = average PSI across selected features
max_feature_psi = highest single-feature PSI
drift_percentage = percentage of features with moderate or severe PSI
status = severe if any severe feature exists, else moderate if any moderate feature exists, else stable
```

Example:

```text
feature_count = 1,000
moderate_or_severe_features = 84
drift_percentage = 8.4
```

This means 8.4 percent of monitored model-input features drifted beyond the moderate threshold.

### 12.5 Drift Feature Selection

The drift calculation excludes source-month one-hot columns such as:

```text
SOURCE_MONTH_JUN-2026
SOURCE_MONTH_JUL-2026
```

Reason:

Source month naturally changes every month. Including those columns would inflate drift even when customer behavior is stable.

## 13. Database and Output Tables

### 13.1 ai_configurations

This table tracks request and processing status.

Typical fields updated:

```text
transaction_id
type
status
message
model_version
current_accuracy
drift
created_on
modified_on
```

Training behavior:

- accepted request is logged
- background job runs
- success updates status and model version
- failure updates status and readable failure message
- failed training should keep the latest successful model version

Inference behavior:

- accepted request returns quickly
- background job runs
- completion updates status, message, drift, accuracy, and model version

### 13.2 api_audit_log

This table audits API request/response behavior.

It should be updated for the same transaction rather than creating duplicate AIML rows for completion.

### 13.3 Campaign Tables

Final campaign output is written to:

```text
ai_ml_campaign_recommendations
ai_ml_campaign_mapping
```

Only actionable strategies are inserted.

If a day prediction is:

```text
-
```

then no campaign mapping is generated for that account/day.

## 14. End-to-End Sample Trace

### 14.1 Raw Input

Assume:

```text
EMI date: 05/09/2026
Prediction month: SEP-2026
Source month: AUG-2026
History window: 3 months
History months: JUN-2026, JUL-2026, AUG-2026
```

Raw communication data:

```csv
apac_card_number,communication_type,comm_status,verbiage_language,emi_date,date,created_date,risk,vertical
MOB-TEST-AI,SMS,DELIVERED,English,05/08/2026,31/07/2026,2026-07-31 11:10:00,MEDIUM,LAP
MOB-TEST-AI,SMS,FAILED,English,05/08/2026,31/07/2026,2026-07-31 12:20:00,MEDIUM,LAP
MOB-TEST-AI,VOICE_BOT,CONNECTED,Hindi,05/08/2026,10/08/2026,2026-08-10 16:05:00,MEDIUM,LAP
```

### 14.2 Day-Level Transformation

```csv
ENTITY_KEY,MONTH,DAY,RISK,VERTICAL,SMS_SUCCESS_11AM_ENGLISH,SMS_FAILED_ENGLISH,SMS_TOTAL_INTENSITY,VOICE_BOT_SUCCESS_4PM_HINDI,VOICE_BOT_TOTAL_INTENSITY,PREDICTED_STRATEGY
MOB-TEST-AI,AUG-2026,D-5,MEDIUM,LAP,1,1,2,0,0,SMS-11AM-ENGLISH
MOB-TEST-AI,AUG-2026,D+5,MEDIUM,LAP,0,0,0,1,1,VOICE_BOT-4PM-HINDI
```

### 14.3 Monthly Roll-Up

```csv
ENTITY_KEY,MONTH,RISK,VERTICAL,SMS_SUCCESS_11AM_ENGLISH,SMS_FAILED_ENGLISH,SMS_TOTAL_INTENSITY,VOICE_BOT_SUCCESS_4PM_HINDI,VOICE_BOT_TOTAL_INTENSITY
MOB-TEST-AI,AUG-2026,MEDIUM,LAP,1,1,2,1,1
```

### 14.4 Rolling Input for Prediction

```csv
ENTITY_KEY,SOURCE_MONTH,RISK_MEDIUM,SMS_SUCCESS_11AM_ENGLISH_M1,SMS_SUCCESS_11AM_ENGLISH_M2,SMS_SUCCESS_11AM_ENGLISH_M3,VOICE_BOT_SUCCESS_4PM_HINDI_M1,VOICE_BOT_SUCCESS_4PM_HINDI_M2,VOICE_BOT_SUCCESS_4PM_HINDI_M3
MOB-TEST-AI,AUG-2026,1,1,3,0,1,0,2
```

### 14.5 Model Scoring

For `D-5`, the `D-5` model scores possible strategy labels:

```text
SMS-11AM-ENGLISH       0.58
-                      0.26
WH-9AM-HINDI           0.10
VOICE_BOT-4PM-HINDI    0.06
```

For MEDIUM risk, top two are retained:

```text
SMS-11AM-ENGLISH|-
```

For `D+5`, the `D+5` model may score:

```text
VOICE_BOT-4PM-HINDI    0.49
SMS-3PM-ENGLISH        0.31
-                      0.20
```

For MEDIUM risk, top two are retained:

```text
VOICE_BOT-4PM-HINDI|SMS-3PM-ENGLISH
```

### 14.6 Final Prediction Output

```csv
SOURCE_RISK,SOURCE_VERTICAL,EMI_DATE,Loan_number,SOURCE_MONTH_USED,MONTH,D-5,D+5,PREDICTION_REASON
MEDIUM,LAP,05/09/2026,MOB-TEST-AI,AUG-2026,SEP-2026,SMS-11AM-ENGLISH|-,VOICE_BOT-4PM-HINDI|SMS-3PM-ENGLISH,"The recommendation is based on recent historical communication behavior, risk profile, channel response, and time-language performance."
```

### 14.7 Campaign Mapping Output

From the prediction output:

```text
D-5 = SMS-11AM-ENGLISH
D+5 = VOICE_BOT-4PM-HINDI
```

The system creates campaign rows for actionable recommendations.

It skips:

```text
-
```

because `-` means no campaign.

## 15. Why the Model Can Predict an Unseen Combination

A CatBoost model can predict a strategy that has zero historical occurrences for one specific account/day because it does not only memorize exact account-level counts.

It learns patterns across the full training population.

Example:

```text
Account: XXXX100X5124
Day: D+11
Prediction: WH-12PM-ENGLISH
Account-level past 3-month count for WH-12PM-ENGLISH on D+11: 0
```

This can still happen because the model may have learned:

- similar accounts with the same risk responded well to WhatsApp
- English worked well for that segment
- 12 PM performed well for comparable behavior patterns
- the account had WhatsApp success at nearby times or days
- the account had low success for competing strategies

So the model is using generalized population patterns plus the account's feature vector, not only direct exact-match history.

## 16. How to Read Training Logs

Important log lines:

```text
Prepared dataset | rows=... columns=... source_months=[...]
```

This confirms the joined model dataset and available source months.

```text
Split rows | train=... validation=... test=... prediction_candidates=...
```

This confirms how many rows went into each split.

```text
Effective source months | train=[...] validation=[...] test=[...] prediction=[...]
```

This is the most important line for month split validation.

```text
Feature matrices | train=(rows, columns) validation=(rows, columns) test=(rows, columns)
```

This confirms the final model input shape.

```text
Day fallback activated | day=D-1 reason=single_unique_value value='-'
```

This means that day did not have enough target variation, so a constant fallback model was used.

```text
Train metrics: ...
Validation metrics: ...
Test metrics: ...
```

This shows model performance by split.

## 17. How to Validate a Run

Use these checks:

- Confirm `upload.scheduler.emi-dates` has the expected full EMI date.
- Confirm logs show the correct prediction month, source month, and cases month.
- Confirm communication files exist for all expected history months and cycle days.
- Confirm `strategy_training_dataset_train.csv` has rows for the expected D-day window.
- Confirm `strategy_monthly_features_train.csv` has monthly rows for expected accounts/months.
- Confirm `strategy_model_input_train.csv` has rolling lag columns like `_M1`, `_M2`, `_M3`.
- Confirm `Effective source months` log matches the desired train/validation/test/prediction split.
- Confirm the prediction file row count matches the current cases population after filters.
- Confirm the evidence file row count matches the configured evidence limit multiplied by the number of prediction days.
- Confirm drift report compares pooled training baseline against current inference input.

## 18. Practical Interpretation

The model is not saying:

> This customer definitely paid because of this exact campaign.

The model is saying:

> Based on historical communication behavior, risk profile, channel/time/language signals, and learned population patterns, this strategy has the strongest learned probability for this account and day.

The evidence file helps business teams review the historical signals behind the recommendation.

The metrics file helps technical teams evaluate whether the model is learning stable patterns.

The drift report helps operations teams detect when current input behavior has moved away from the training baseline.



## 19. Complete Script-by-Script Implementation Reference

This section maps the business flow to the actual scripts that run in the project.

### 19.1 API Entry Point

Main file:

```text
src/campaign_recommendation/api_service.py
```

Primary API responsibilities:

- Authenticate requests.
- Validate request body.
- Create or update `ai_configurations` and `api_audit_log` records.
- Start training or inference asynchronously.
- Return `ACCEPTED` immediately while the background job continues.
- Update final status after the background job completes.

Important API configuration:

```text
API_HOST
API_PORT
API_AUTH_USERNAME
API_AUTH_PASSWORD
API_AUTH_SECRET
API_TOKEN_TTL_SECONDS
AI_CONFIG_TABLE
API_AUDIT_TABLE
API_MODEL_BASE_VERSION
API_EXPORT_AFTER_INFERENCE
API_EXPORT_WRITE
API_EXPORT_TRIGGER_STATE
API_REQUEST_QUEUE_SIZE
VENV_PYTHON
```

Important database configuration:

```text
PGHOST
PGPORT
PGDATABASE
PGUSER
PGPASSWORD
SOURCE_SCHEMA
SOURCE_TABLE
TARGET_SCHEMA
CAMPAIGN_TABLE
CAMPAIGN_MAPPING_TABLE
```

The API timestamps are generated in Asia/Kolkata time before writing to the database.

### 19.2 Training API Flow

Training request example:

```json
{
  "transactionId": "TRN202606080003",
  "months": 6
}
```

The training API validates `months` using:

```text
data_config.key_name = aiml.training.month.duration
```

If the request month duration does not match the configured value, training is rejected.

If validation passes, the API starts the training process and returns an accepted response using the currently active model version. The new model version is written only after successful training.

Training flow:

```text
API request
  -> validate months against data_config
  -> create/update ai_configurations row as ACCEPTED
  -> create/update api_audit_log row as ACCEPTED
  -> run training window preparation
  -> run feature generation
  -> run monthly aggregation
  -> run schedule target generation
  -> train CatBoost day models
  -> save versioned model and metrics
  -> update ai_configurations as COMPLETED or FAILED
  -> update api_audit_log as COMPLETED or FAILED
```

### 19.3 Inference API Flow

Inference request example:

```json
{
  "transactionId": "TRN202606080002"
}
```

Inference flow:

```text
API request
  -> create/update ai_configurations row as ACCEPTED
  -> create/update api_audit_log row as ACCEPTED
  -> resolve latest successful training model version
  -> run monthly inference pipeline
  -> fetch current communication history
  -> fetch current digital cases
  -> rebuild inference features
  -> load model
  -> generate predictions
  -> generate evidence file
  -> compute drift
  -> write predictions and campaign tables
  -> update ai_configurations as COMPLETED or FAILED
  -> update api_audit_log as COMPLETED or FAILED
```

The inference API response contains the latest available metrics at accept time. Final completion values are written after the background job completes.

## 20. Complete File Handoff Between Pipeline Steps

### 20.1 Raw Communication Files

Generated by:

```text
scripts/fetch_month_from_postgres.py
```

Stored under:

```text
data/communication/
```

Naming:

```text
comm_data_<MMM><YYYY>_<cycle_day>.csv
```

Example:

```text
comm_data_AUG2026_5.csv
comm_data_AUG2026_15.csv
```

Each file represents communication rows for a given month and EMI cycle day.

### 20.2 Raw Case Files

Generated by:

```text
scripts/fetch_cases_from_postgres.py
```

Stored under:

```text
data/cases/
```

Example:

```text
digital_cases_SEP2026.csv
```

This file defines the current prediction population.

### 20.3 Day-Level Strategy Dataset

Generated by:

```text
scripts/generate_strategy_dataset.py
```

Typical output:

```text
data/training/strategy_training_dataset_train.csv
```

Grain:

```text
entity/account + EMI month + D-day offset
```

Purpose:

- Build success features.
- Build failure features.
- Build total intensity features.
- Select the strongest historical strategy for the day.
- Create `TARGET_SAMPLE_WEIGHT` when weighting is enabled.

### 20.4 Monthly Feature Dataset

Generated by:

```text
scripts/build_monthly_feature_dataset.py
```

Typical output:

```text
data/features/strategy_monthly_features_train.csv
```

Grain:

```text
entity/account + month
```

Purpose:

- Compress day-level features into monthly account behavior.
- Sum success/failure/intensity columns across the month.
- Keep first observed risk and vertical for that account/month.

### 20.5 Schedule Dataset

Generated by:

```text
scripts/build_strategy_schedule_dataset.py
```

Typical output:

```text
data/schedules/strategy_schedule_dataset_train.csv
```

Grain:

```text
entity/account + month
```

Purpose:

- Convert day-level target rows into wide target columns.
- Create one target column per campaign day.
- Create one weight column per campaign day.

Example target columns:

```text
D-5
D-4
D-3
D-2
D-1
D+1
D+2
...
D+20
```

Example weight columns:

```text
D-5__WEIGHT
D-4__WEIGHT
D+20__WEIGHT
```

### 20.6 Training Model Input Audit Files

Generated by:

```text
scripts/train_next_month_strategy_model_catboost.py
```

Files:

```text
data/features/strategy_model_input_train.csv
data/features/strategy_model_input_train_daywise.csv
```

Purpose:

- Show the actual prepared training data after source-target merge and rolling monthly lag expansion.
- Help validate what the model received.
- Help debug month split and feature count issues.

### 20.7 Inference Model Input Audit Files

Generated by:

```text
scripts/predict_next_month_strategy_catboost.py
```

Files:

```text
data/features/strategy_model_input_inference.csv
data/features/strategy_model_input_inference_daywise.csv
```

Purpose:

- Show the actual prepared inference rows before prediction.
- Confirm current source month and lag columns.
- Help validate that training and inference use the same feature schema.

### 20.8 Prediction File

Generated by:

```text
scripts/predict_next_month_strategy_catboost.py
```

Stored under:

```text
artifacts/predictions/
```

Naming:

```text
<YYYY>_<MM>_strategy_predictions_catboost_3m.csv
```

Example:

```text
2026_09_strategy_predictions_catboost_3m.csv
```

Purpose:

- Main model prediction output.
- One row per current account.
- Day columns contain recommended strategies.

### 20.9 Prediction Evidence File

Generated by:

```text
scripts/generate_prediction_evidence.py
```

Stored next to the prediction file.

Naming:

```text
<YYYY>_<MM>_strategy_predictions_catboost_3m_prediction_evidence.csv
```

Purpose:

- Explain top strategies and historical account-level evidence.
- Provide model probabilities and historical success percentages.
- Support campaign team validation.

### 20.10 Model and Metrics Artifacts

Generated by:

```text
scripts/train_next_month_strategy_model_catboost.py
```

Model naming:

```text
artifacts/models/next_month_strategy_catboost_3m_v<version>.joblib
```

Metrics naming:

```text
artifacts/metrics/next_month_strategy_catboost_3m_metrics_v<version>.json
```

Example:

```text
next_month_strategy_catboost_3m_v1.1.9.joblib
next_month_strategy_catboost_3m_metrics_v1.1.9.json
```

The model bundle stores:

```text
models
label_encoders
feature_columns
target_offset_months
history_window_months
train_source_months
validation_source_months
test_source_months
fallback_config
```

## 21. Complete Feature Inventory Logic

The exact number of model features depends on:

- Channels present in the data.
- Languages present in the data.
- Time buckets present in the data.
- `history_window_months`.
- One-hot encoded risk and source-month values.

In the current local latest model inspected:

```text
Model file: artifacts/models/next_month_strategy_catboost_3m_v1.1.9.joblib
Actual model input features: 832
History window months: 8
Day models: 25
```

### 21.1 Success Feature Formula

For each account/month/day/channel/hour/language:

```text
<CHANNEL>_SUCCESS_<TIME>_<LANGUAGE>
  = count of communication rows where status is successful
```

Example:

```text
SMS_SUCCESS_12PM_ENGLISH = 3
```

Meaning:

```text
The account had 3 successful SMS communications in English in the 12 PM bucket for that month/day grain.
```

### 21.2 Failure Feature Formula

For each account/month/day/channel/language:

```text
<CHANNEL>_FAILED_<LANGUAGE>
  = count of communication rows where status is not successful
```

Example:

```text
SMS_FAILED_ENGLISH = 2
```

Meaning:

```text
The account had 2 non-successful English SMS communications for that month/day grain.
```

### 21.3 Total Intensity Formula

For each account/month/day/channel:

```text
<CHANNEL>_TOTAL_INTENSITY
  = count of all communication rows for that channel
```

This includes both success and failure.

Example:

```text
SMS_TOTAL_INTENSITY = SMS success count + SMS failure count
```

### 21.4 Monthly Roll-Up Formula

For each account/month:

```text
monthly_feature = sum(day_level_feature across D-5 to D+20 excluding D)
```

Example:

```text
JUN SMS_SUCCESS_9AM_ENGLISH =
    D-5 SMS_SUCCESS_9AM_ENGLISH
  + D-4 SMS_SUCCESS_9AM_ENGLISH
  + ...
  + D+20 SMS_SUCCESS_9AM_ENGLISH
```

### 21.5 Rolling Lag Formula

For a source month `T` and history window `n`:

```text
feature_M1 = feature value in month T
feature_M2 = feature value in month T-1
feature_M3 = feature value in month T-2
...
feature_Mn = feature value in month T-(n-1)
```

Example for predicting September 2026 with source month August 2026 and 3-month history:

```text
SMS_SUCCESS_9AM_ENGLISH_M1 = August value
SMS_SUCCESS_9AM_ENGLISH_M2 = July value
SMS_SUCCESS_9AM_ENGLISH_M3 = June value
```

### 21.6 Risk Encoding

Risk is one-hot encoded.

Example for a LOW risk account:

- `RISK_LOW` is set to `1`.
- `RISK_MEDIUM` is set to `0`.
- `RISK_HIGH` is set to `0`.

### 21.7 Source Month Encoding

Source month is one-hot encoded for model training and inference.

Example for an August 2026 source month:

- `SOURCE_MONTH_AUG-2026` is set to `1`.
- Other source-month indicator columns are set to `0`.

For drift calculation, source-month one-hot features are excluded because the calendar month naturally changes and would create false drift.

## 22. Complete Training Algorithm Flow

### 22.1 Prepare Dataset

Training reads:

```text
feature_file = strategy_monthly_features_train.csv
schedule_file = strategy_schedule_dataset_train.csv
```

It calls:

```text
prepare_next_month_dataset(...)
```

This function:

- Reads monthly features.
- Applies rolling history window columns.
- Creates `SOURCE_MONTH` from feature `MONTH`.
- Creates `TARGET_MONTH_PERIOD = SOURCE_MONTH_PERIOD + target_offset_months`.
- Reads schedule targets.
- Creates target month period from schedule month.
- Merges source-month features with next-month schedule labels.

### 22.2 Example Source-Target Merge

If target offset is 1:

```text
Source feature month: JUN-2026
Target schedule month: JUL-2026
```

Merged row:

```text
JUN-2026 behavior features + JUL-2026 day strategy labels
```

This teaches:

```text
Given behavior up to June, what strategy happened or worked in July?
```

### 22.3 Build Feature Matrix

The training script removes non-feature columns and one-hot encodes categorical fields.

Removed before model fit:

```text
ENTITY_KEY
APAC_CARD_NUMBER
DAY target columns
DAY weight columns
TARGET_MONTH
TARGET_MONTH_PERIOD
SOURCE_MONTH_PERIOD
TARGET_RISK
VERTICAL
```

Encoded:

```text
SOURCE_MONTH
RISK
```

Final output:

```text
X_train = numeric matrix used by CatBoost
```

### 22.4 Build Day Targets

For each day:

```text
y_train_day = train_df[day].fillna("-")
```

Example:

```text
D-5 target vector = all D-5 labels for training rows
D+20 target vector = all D+20 labels for training rows
```

### 22.5 Train Day Models

For each day in `DAY_COLUMNS`:

```text
fit_day_model(day, X_train, y_train_day, sample_weight_if_enabled, ...)
```

This produces:

```text
models["D-5"]
models["D-4"]
...
models["D+20"]
```

Each model is a multiclass classifier.

The model objective is to learn probability scores over strategy labels.

Example labels for one day:

```text
-
SMS-9AM-ENGLISH
SMS-11AM-REGIONAL
WH-4PM-HINDI
VOICE-2PM-ENGLISH
VOICE_BOT-10AM-HINDI
```

### 22.6 Label Encoding

CatBoost receives numeric class IDs, so labels are encoded internally.

Example:

```text
-                    -> 0
SMS-9AM-ENGLISH      -> 1
WH-4PM-HINDI         -> 2
VOICE_BOT-10AM-HINDI -> 3
```

After prediction, numeric class IDs are converted back to business-readable labels.

### 22.7 Checkpoints

Each day model can be checkpointed under a version-specific checkpoint directory.

Example:

```text
artifacts/checkpoints/catboost_3m_v1.1.9/Dminus5.joblib
artifacts/checkpoints/catboost_3m_v1.1.9/Dplus20.joblib
```

A stale checkpoint is skipped when the stored checkpoint metadata does not match the current training run metadata. This prevents reusing a model trained with different months, different window length, or different weighting setting.

## 23. Complete Inference Algorithm Flow

### 23.1 Resolve Model Version

Inference loads the latest successful training model version from `ai_configurations`.

If no successful training row exists, it uses fallback model resolution from configured defaults.

The model bundle defines the exact feature columns and history window expected by the trained model.

### 23.2 Build Current Prediction Population

Inference reads current cases for the prediction month.

Example:

```text
EMI date: 05/09/2026
Current cases file: digital_cases_SEP2026.csv
```

Only accounts present in the cases file are eligible for final prediction output.

### 23.3 Build Current Feature Matrix

Inference rebuilds monthly communication features for source/history months, then applies the same rolling lag logic.

The matrix is aligned to the trained model feature list:

```text
X_inference = X_inference.reindex(columns=bundle["feature_columns"], fill_value=0)
```

This guarantees training and inference use the same columns in the same order.

If inference has a new feature not seen during training, it is ignored.

If training had a feature missing during inference, it is filled with `0`.

### 23.4 Score Each Day

For each account row:

```text
D-5 model receives X_inference row -> returns probabilities
D-4 model receives X_inference row -> returns probabilities
...
D+20 model receives X_inference row -> returns probabilities
```

Example:

```text
Account: MOB-TEST-AI
Day: D+5
```

Model probability output:

```text
SMS-3PM-ENGLISH     0.42
WH-9AM-HINDI        0.31
-                   0.19
VOICE-2PM-ENGLISH   0.08
```

Final recommendation depends on risk top-k policy:

```text
LOW    -> SMS-3PM-ENGLISH
MEDIUM -> SMS-3PM-ENGLISH|WH-9AM-HINDI
HIGH   -> SMS-3PM-ENGLISH|WH-9AM-HINDI|-
```

### 23.5 Convert Strategy to Campaign Rows

The strategy label is parsed as:

```text
<CHANNEL>-<TIME>-<LANGUAGE>
```

Example:

```text
SMS-3PM-ENGLISH
```

Becomes:

```text
mode = SMS
time = 15:00:00
language = ENGLISH
```

The scheduler/campaign output only uses actionable strategies. `-` is treated as no campaign.

## 24. Complete Prediction Evidence Logic

The prediction evidence file is designed for validation and business explanation.

It is not the model input. It is a proof file generated after prediction.

### 24.1 Evidence Data Sources

Evidence uses:

```text
prediction CSV
trained model bundle
monthly feature file
schedule file
base population file
```

The script also accepts communication file arguments for pipeline compatibility.

### 24.2 Evidence Ranking Columns

For each account/day, evidence includes top model recommendations:

```text
recommendedStrategy1
recommendedStrategy1_SuccessProbability
recommendedStrategy2
recommendedStrategy2_SuccessProbability
recommendedStrategy3
recommendedStrategy3_SuccessProbability
```

The success probability here is the model probability for that strategy class.

It is not the historical success percentage.

### 24.3 Historical Success Percentage Columns

Historical success percentages are calculated separately from model probabilities.

Naming:

```text
historicalSuccessFeature1
historical_successprecentage1
historicalSuccessFeature2
historical_successprecentage2
historicalSuccessFeature3
historical_successprecentage3
```

Formula:

```text
historical_success_percentage = (success_count / channel_total_intensity) * 100
```

If success count or channel total intensity is zero:

```text
historical_success_percentage = 0.0
```

Example:

```text
recommendedStrategy1 = SMS-4PM-ENGLISH
success feature = SMS_SUCCESS_4PM_ENGLISH
channel total intensity = SMS_TOTAL_INTENSITY
```

Calculation:

```text
SMS_SUCCESS_4PM_ENGLISH = 5
SMS_TOTAL_INTENSITY = 20
historical_success_percentage = 25.00
```

### 24.4 Evidence Counts

Evidence count columns show historical feature counts for that account and day.

Examples:

```text
SMS_SUCCESS_9AM_ENGLISH
SMS_SUCCESS_4PM_REGIONAL
WH_SUCCESS_10AM_HINDI
VOICE_SUCCESS_2PM_ENGLISH
VOICE_BOT_SUCCESS_5PM_HINDI
SMS_TOTAL_INTENSITY
WH_TOTAL_INTENSITY
VOICE_TOTAL_INTENSITY
VOICE_BOT_TOTAL_INTENSITY
MaxCount_SMS
MaxCount_WH
MaxCount_VOICE
MaxCount_VOICE_BOT
```

The success columns are same-day-offset historical counts.

The total intensity columns are cumulative channel totals used to calculate historical percentages.

### 24.5 Evidence Row Count

Evidence rows are generated per selected account and prediction day.

Formula:

```text
evidence_rows = selected_account_count * 25 prediction days
```

because the active day window has 25 days:

```text
5 pre-due days + 20 post-due days = 25
```

If:

```text
PREDICTION_EVIDENCE_LIMIT=50
```

then expected evidence rows are:

```text
50 * 25 = 1,250
```

If there are 9,248 prediction accounts and no evidence limit:

```text
9,248 * 25 = 231,200 evidence rows
```

This is why full evidence generation can take more time.

## 25. Complete Drift Logic and Interpretation

### 25.1 Active Data Used for Drift

Baseline data:

```text
strategy_monthly_features_train.csv
```

The baseline is filtered to the model's actual training source months stored in the metrics file.

Inference data:

```text
current inference model input for the active source month
```

Both baseline and inference are transformed using the same model feature matrix logic.

### 25.2 Drift Matrix Alignment

Both matrices are aligned to the trained model feature columns:

```text
baseline_matrix = baseline_matrix.reindex(columns=bundle["feature_columns"], fill_value=0)
inference_matrix = inference_matrix.reindex(columns=bundle["feature_columns"], fill_value=0)
```

This avoids comparing unrelated columns.

### 25.3 Pooled Training Baseline

The current drift design compares inference against the pooled training population.

This means:

```text
baseline = all rows from all training source months used by the active model
```

not:

```text
baseline = one isolated source month
```

This reduces false drift caused by month-to-month operational variation.

### 25.4 Sampling

The drift baseline can be sampled to keep calculation fast.

Current implementation caps the baseline at 10,000 rows before PSI calculation when it is larger than that.

This keeps drift computation manageable for large training datasets.

### 25.5 Feature Exclusion

Source month dummy columns are excluded:

```text
SOURCE_MONTH_*
```

Reason:

```text
The source month changes naturally every run, so including it would create artificial drift.
```

### 25.6 Output Fields

Drift report fields include:

```text
baseline_rows
inference_rows
blank_inference_rows
feature_count
overall_psi
max_feature_psi
moderate_feature_count
severe_feature_count
drift_percentage
status
feature_metrics
```

### 25.7 How `driftPercentage` Is Calculated

`driftPercentage` is not the same as `overall_psi`.

Formula:

```text
driftPercentage = (number of moderate or severe drift features / total drift features) * 100
```

Example:

```text
total drift features = 832
moderate or severe features = 7
```

```text
driftPercentage = (7 / 832) * 100 = 0.84
```

### 25.8 How `currentAccuracy` Is Chosen

The API reads the latest model metrics snapshot for the active model version.

In general, it comes from model metrics generated during training, such as validation accuracy. If a value is missing, API fallback defaults may be applied depending on the endpoint logic.

## 26. Complete Configuration Checklist

### 26.1 Database and Schema

```text
PGHOST
PGPORT
PGDATABASE
PGUSER
PGPASSWORD
SOURCE_SCHEMA
SOURCE_TABLE
TARGET_SCHEMA
PREDICTION_TABLE
CAMPAIGN_TABLE
CAMPAIGN_MAPPING_TABLE
```

### 26.2 Communication and Cases Columns

```text
COMMUNICATION_VERTICAL_COLUMN
COMMUNICATION_PARTY_ID_COLUMN
CASES_VERTICAL_COLUMN
CASES_PARTY_ID_COLUMN
```

These allow different clients to use different physical column names for vertical and party ID.

### 26.3 Month and EMI Configuration

```text
upload.scheduler.emi-dates in data_config
aiml.training.month.duration in data_config
SOURCE_MONTH
PREDICT_MONTH
FEATURE_MONTH_SOURCE
HISTORY_WINDOW_MONTHS
```

`SOURCE_MONTH` and `PREDICT_MONTH` can override derived months when explicitly configured, but normal operation derives them from `upload.scheduler.emi-dates`.

### 26.4 Model Configuration

```text
MODEL_NAME
MODEL_SERVING
MODEL_VERSION
MODEL_FILE
METRICS_FILE
MLFLOW_TRACKING_URI
MLFLOW_MODEL_URI
MLFLOW_REGISTERED_MODEL_NAME
```

### 26.5 Weighting Configuration

```text
STRATEGY_USE_STATUS_WEIGHTS
STRATEGY_SUCCESS_SCORES_JSON
STRATEGY_SAMPLE_WEIGHT_POSITIVE_BOOST
```

### 26.6 Evidence Configuration

```text
PREDICTION_EVIDENCE_LIMIT
```

Use `0` for no evidence row limit.

### 26.7 Export Configuration

```text
API_EXPORT_AFTER_INFERENCE
API_EXPORT_WRITE
API_EXPORT_TRIGGER_STATE
SFTP_EXPORT_PATH
SFTP_UPLOAD_ENABLED
SFTP_HOST
SFTP_PORT
SFTP_USERNAME
SFTP_PASSWORD
SFTP_PRIVATE_KEY_PATH
SFTP_PRIVATE_KEY_PASSPHRASE
SFTP_REMOTE_DATASET_PATH
SFTP_REMOTE_SCHEDULER_PATH
SFTP_RETRIES
SFTP_RETRY_DELAY_SECONDS
SFTP_TIMEOUT_SECONDS
SFTP_FAIL_ON_ERROR
```

## 27. Edge Cases and How the Pipeline Handles Them

### 27.1 Invalid EMI Dates

Example invalid date:

```text
31/02/2025
```

This is invalid because February does not have 31 days.

Invalid EMI dates should be skipped or rejected depending on the calling script path, and logs should explain the invalid value.

### 27.2 Missing Day Data

If no training data exists for a specific day such as `D+17`, the day model falls back to `-`.

Business meaning:

```text
There is not enough historical evidence to recommend a campaign for that day.
```

### 27.3 Single-Class Day Data

If a day has only one target label, CatBoost cannot learn a split for that day.

Fallback behavior:

```text
Only '-' exists -> always predict '-'
Only one campaign exists -> always predict that campaign
```

### 27.4 No Current Case Row

If an account has communication history but is absent from the current digital cases file, it is excluded from final prediction output.

Reason:

```text
The cases file defines who is eligible for this month's campaign.
```

### 27.5 No Campaign Prediction

If a model predicts `-`, the campaign tables do not get a campaign row for that account/day.

Reason:

```text
'-' is a deliberate no-campaign recommendation.
```

### 27.6 New Channel or Language Appears in Inference

If inference data contains a new feature that was not present during training, it is not used by the current model.

Reason:

```text
The model can only score columns it was trained with.
```

The new signal becomes useful after retraining.

### 27.7 Feature Missing in Inference

If the model expects a feature but the current inference month does not have it, the value is filled with zero.

Example:

```text
VOICE_BOT_SUCCESS_10AM_HINDI_M1 = 0
```

## 28. Operational Debugging Guide

### 28.1 If Prediction File Is Empty

Check:

- Did cases fetch return rows for the prediction month?
- Does `upload.scheduler.emi-dates` point to the intended prediction month?
- Is source month derived correctly as prediction month minus one?
- Are there matching entity keys between cases and feature history?
- Is multi-APAC mode using `party_id` consistently if enabled?

### 28.2 If Campaign Mapping Count Is Lower Than Prediction Count

This can be expected.

Reasons:

- Prediction rows are one per account.
- Campaign mapping rows are created only for actionable strategy cells.
- `-` predictions are skipped.
- Risk top-k affects how many strategies are retained.
- Vendor matching can expand or limit rows depending on configured vendor lists.

### 28.3 If Evidence File Has Fewer Rows Than Expected

Check:

```text
PREDICTION_EVIDENCE_LIMIT
```

Expected row formula:

```text
min(prediction_accounts, evidence_limit) * 25
```

If limit is `0`:

```text
prediction_accounts * 25
```

### 28.4 If Drift Is High

Check `feature_metrics` for:

```text
baseline_non_zero_rate
inference_non_zero_rate
baseline_mean
inference_mean
psi
```

Common causes:

- Communication volume changed materially.
- A channel stopped or started appearing.
- Vendor/client behavior changed.
- Training baseline has sparse lag features.
- Inference month has data from a different operational process.
- New languages/channels appeared after training.

### 28.5 If Accuracy Drops

Check:

- Whether target labels are dominated by `-`.
- Whether validation month behavior is very different from training months.
- Whether many day models used fallback.
- Whether feature counts are mostly zero for important channels.
- Whether training/inference are using the same entity key mode.
- Whether month split leaves too little targetable data.

## 29. Business Explanation of Model Behavior

### 29.1 Why Highest Global Count May Not Win

The model does not choose only the most common strategy in the full training table.

It predicts based on the current account's feature vector.

A globally common label like `SMS-11AM-ENGLISH` may lose for one account if that account's feature pattern looks closer to accounts that historically received `-`, `VOICE`, or `VOICE_BOT`.

### 29.2 Why `-` Can Be Predicted

`-` is a valid learned class.

It can be selected when the model sees patterns such as:

- low useful success signal
- high failed communication intensity
- similar accounts historically receiving no campaign
- risk-specific policy retaining only top one recommendation

For LOW risk, this is especially visible because only the top recommendation is retained.

### 29.3 Why VOICE_BOT Can Appear Even If Rare

VOICE_BOT is a valid channel in the model.

It can appear when:

- the account has voice-bot related history
- similar accounts show voice-bot success patterns
- competing channel evidence is weaker
- the day-specific model assigns highest probability to voice-bot for that row

### 29.4 Why Language May Look Missing

Language appears only when the predicted strategy label includes a language.

If the top label is `-`, there is no language to show.

If historical language data is sparse or normalized into `REGIONAL`, specific language bifurcation may not appear strongly in the recommendation.

## 30. Summary of Current Active Logic

Current active model logic:

```text
Raw communication rows
  -> normalize channel/status/language/risk/vertical
  -> calculate EMI-relative day D-5 to D+20 excluding D
  -> build day-level success/failure/intensity features
  -> pick strongest day strategy target
  -> roll day features into one monthly account row
  -> create rolling lag features M1..Mn
  -> merge source month behavior with next month target labels
  -> train one CatBoost classifier per D-day
  -> save versioned model and metrics
  -> during inference, rebuild same feature matrix for current cases
  -> score each day model
  -> apply risk-based top-k strategy selection
  -> write prediction, evidence, DB, and export outputs
  -> calculate PSI drift against pooled training baseline
```

The most important implementation principle is:

> Training and inference must build the same feature columns in the same order, using the same month derivation, entity key, channel normalization, and history window logic.
