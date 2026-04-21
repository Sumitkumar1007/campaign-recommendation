# 3-Month CatBoost Training Explainer

This document explains how the current next-month CatBoost strategy model works after the change to a `3`-month rolling history window.

It answers:

- what the model input looks like
- how monthly raw communication history becomes training features
- how the target labels are formed
- how `FEB + MAR + APR` history is used to predict `MAY`
- why some predictions are blank

## 1. High-Level Goal

For each `APAC_CARD_NUMBER` / loan account, we want to predict the next month's communication schedule:

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

Each day column is a strategy label such as:

- `SMS-9AM-ENGLISH`
- `WH-2PM-ENGLISH`
- `IVR-2PM-ENGLISH`
- `-` meaning no communication

The CatBoost pipeline trains one model per day column, so there are `10` CatBoost classifiers in the final bundle.

## 2. Input Data Layers

The pipeline uses these main layers:

1. Raw monthly communication data  
   Example file: [mfl_recomm_model_APR2026_comm_data.csv](/home/ubuntu/aiml/recommendation/data/communication/MFL_COMMUNICATION_DATA/mfl_recomm_model_APR2026_comm_data.csv)

2. Day-level strategy dataset  
   Example file: [strategy_training_dataset_all_months.csv](/home/ubuntu/aiml/recommendation/data/training/strategy_training_dataset_all_months.csv)

3. Monthly aggregated feature dataset  
   Example file: [strategy_monthly_features.csv](/home/ubuntu/aiml/recommendation/data/features/strategy_monthly_features.csv)

4. Schedule-style target dataset  
   Example file: [strategy_schedule_dataset_all_months.csv](/home/ubuntu/aiml/recommendation/data/schedules/strategy_schedule_dataset_all_months.csv)

5. CatBoost training / inference code  
   Main script: [train_next_month_strategy_model_catboost.py](/home/ubuntu/aiml/recommendation/scripts/train_next_month_strategy_model_catboost.py)

## 3. What the Monthly Feature Table Contains

The monthly feature table has one row per:

- `APAC_CARD_NUMBER`
- `MONTH`

and stores engineered counts like:

- `SMS_TOTAL_INTENSITY`
- `WH_TOTAL_INTENSITY`
- `VOICE_TOTAL_INTENSITY`
- `SMS_SUCCESS_10AM_REGIONAL`
- `WH_SUCCESS_9AM_ENGLISH`
- `VOICE_FAILED_ENGLISH`
- many similar channel/status/time/language combinations
- `RISK`

Important: this monthly table is still month-by-month.  
The new `3`-month logic is applied inside the CatBoost training/inference script on top of this table.

## 4. How the 3-Month Rolling Input Is Built

The new logic lives in:

- `build_rolling_feature_windows(...)`

inside [train_next_month_strategy_model_catboost.py](/home/ubuntu/aiml/recommendation/scripts/train_next_month_strategy_model_catboost.py).

It does this:

1. Read the monthly feature rows.
2. Sort by `APAC_CARD_NUMBER` and month.
3. For each account, take the latest `N` observed months, where `N = 3`.
4. Sum the numeric features across that rolling window.
5. Use the provided `RISK`.
6. Keep the end month as the `SOURCE_MONTH`.

So for `APR-2026`, the feature row used for prediction is:

- `FEB-2026` features
- plus `MAR-2026` features
- plus `APR-2026` features

This combined row is then used to predict `MAY-2026`.

## 5. Concrete Example

Sample account:

- `MFLAPDSECUL000005079554`

### 5.1 Month-by-Month Source Features

From [strategy_monthly_features.csv](/home/ubuntu/aiml/recommendation/data/features/strategy_monthly_features.csv):

`FEB-2026`

- `RISK = LOW`
- `SMS_TOTAL_INTENSITY = 6`
- `WH_TOTAL_INTENSITY = 5`
- `VOICE_TOTAL_INTENSITY = 5`
- `SMS_SUCCESS_9AM_REGIONAL = 1`
- `SMS_SUCCESS_10AM_REGIONAL = 1`
- `SMS_SUCCESS_3PM_ENGLISH = 1`
- `SMS_SUCCESS_4PM_ENGLISH = 1`
- `SMS_SUCCESS_5PM_ENGLISH = 1`
- `SMS_SUCCESS_8AM_REGIONAL = 1`
- `VOICE_FAILED_ENGLISH = 3`
- `VOICE_FAILED_TELUGU = 2`
- `WH_SUCCESS_3PM_TELUGU = 1`
- `WH_SUCCESS_5PM_ENGLISH = 1`
- `WH_SUCCESS_5PM_TELUGU = 1`
- `WH_SUCCESS_9AM_ENGLISH = 1`
- `WH_SUCCESS_9AM_TELUGU = 1`

`MAR-2026`

- `RISK = LOW`
- `SMS_TOTAL_INTENSITY = 4`
- `WH_TOTAL_INTENSITY = 6`
- `VOICE_TOTAL_INTENSITY = 5`
- `SMS_SUCCESS_10AM_ENGLISH = 1`
- `SMS_SUCCESS_12PM_REGIONAL = 1`
- `SMS_SUCCESS_3PM_REGIONAL = 1`
- `SMS_SUCCESS_4PM_ENGLISH = 1`
- `VOICE_FAILED_ENGLISH = 2`
- `VOICE_FAILED_TELUGU = 3`
- `WH_SUCCESS_1PM_ENGLISH = 1`
- `WH_SUCCESS_2PM_ENGLISH = 1`
- `WH_SUCCESS_3PM_ENGLISH = 1`
- `WH_SUCCESS_5PM_ENGLISH = 1`
- `WH_SUCCESS_8AM_TELUGU = 1`
- `WH_SUCCESS_9AM_ENGLISH = 1`

`APR-2026`

- `RISK = LOW`
- `SMS_TOTAL_INTENSITY = 1`
- `WH_TOTAL_INTENSITY = 0`
- `VOICE_TOTAL_INTENSITY = 0`
- `SMS_SUCCESS_10AM_REGIONAL = 1`

### 5.2 Rolled 3-Month Input for `APR-2026`

The model’s rolled input for `APR-2026` becomes:

- `SMS_TOTAL_INTENSITY = 11`
- `WH_TOTAL_INTENSITY = 11`
- `VOICE_TOTAL_INTENSITY = 10`
- `SMS_SUCCESS_10AM_REGIONAL = 2`
- `SMS_SUCCESS_10AM_ENGLISH = 1`
- `SMS_SUCCESS_12PM_REGIONAL = 1`
- `SMS_SUCCESS_3PM_ENGLISH = 1`
- `SMS_SUCCESS_3PM_REGIONAL = 1`
- `SMS_SUCCESS_4PM_ENGLISH = 2`
- `SMS_SUCCESS_5PM_ENGLISH = 1`
- `SMS_SUCCESS_8AM_REGIONAL = 1`
- `SMS_SUCCESS_9AM_REGIONAL = 1`
- `VOICE_FAILED_ENGLISH = 5`
- `VOICE_FAILED_TELUGU = 5`
- `WH_SUCCESS_1PM_ENGLISH = 1`
- `WH_SUCCESS_2PM_ENGLISH = 1`
- `WH_SUCCESS_3PM_ENGLISH = 1`
- `WH_SUCCESS_3PM_TELUGU = 1`
- `WH_SUCCESS_5PM_ENGLISH = 2`
- `WH_SUCCESS_5PM_TELUGU = 1`
- `WH_SUCCESS_8AM_TELUGU = 1`
- `WH_SUCCESS_9AM_ENGLISH = 2`
- `WH_SUCCESS_9AM_TELUGU = 1`

This is much richer than using only April’s single-month row.

### 5.3 Target Row for Training

For this same account, the training target comes from the next month schedule row in [strategy_schedule_dataset_all_months.csv](/home/ubuntu/aiml/recommendation/data/schedules/strategy_schedule_dataset_all_months.csv).

Examples from history:

`FEB-2026` schedule

- `D-5 = SMS-9AM-REGIONAL`
- `D-3 = WH-9AM-TELUGU`
- `D+1 = SMS-4PM-ENGLISH`
- `D+2 = SMS-8AM-REGIONAL`
- `D+3 = SMS-5PM-ENGLISH`
- `D+4 = SMS-10AM-REGIONAL`
- `D+5 = SMS-3PM-ENGLISH`

`MAR-2026` schedule

- `D+1 = WH-2PM-ENGLISH`
- `D+2 = SMS-4PM-ENGLISH`
- `D+3 = SMS-12PM-REGIONAL`
- `D+4 = SMS-10AM-ENGLISH`
- `D+5 = SMS-3PM-REGIONAL`

`APR-2026` schedule

- `D-4 = SMS-10AM-REGIONAL`

Training pairing example:

- source row ending in `FEB-2026` predicts target schedule `MAR-2026`
- source row ending in `MAR-2026` predicts target schedule `APR-2026`
- source row ending in `APR-2026` predicts target schedule `MAY-2026` during inference

## 6. Exact Training Flow

This is what the script does:

1. Read `strategy_monthly_features.csv`
2. Read `strategy_schedule_dataset_all_months.csv`
3. Build 3-month rolled source features per account
4. Convert source month to `SOURCE_MONTH_PERIOD`
5. Shift by `target_offset_months = 1` to get `TARGET_MONTH_PERIOD`
6. Join the rolled feature row to the next month’s schedule row
7. Split by source month:
   - train: `NOV-2025`, `DEC-2025`, `JAN-2026`
   - validation: `FEB-2026`
   - prediction source: `APR-2026`
8. Convert categorical columns into dummies:
   - `SOURCE_MONTH_*`
   - `RISK_*`
9. Train one CatBoost model per target day:
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
10. Save the model bundle and metrics
11. Score the prediction source month rows

## 7. Why There Are 10 Models

The output is a 10-column schedule, not one single label.

So the training script does:

- one classifier for `D-5`
- one classifier for `D-4`
- ...
- one classifier for `D+5`

Each classifier learns:

- given the rolled 3-month feature row
- what is the most likely strategy label for that day position

## 8. What the Final Model Input Matrix Looks Like

After feature preparation, each training row contains:

- rolled numeric feature totals across the last `3` months
- one-hot encoded source month
- one-hot encoded risk

Examples of final feature columns:

- `SMS_TOTAL_INTENSITY`
- `WH_TOTAL_INTENSITY`
- `VOICE_TOTAL_INTENSITY`
- `SMS_SUCCESS_10AM_REGIONAL`
- `VOICE_FAILED_ENGLISH`
- `WH_SUCCESS_9AM_ENGLISH`
- `SOURCE_MONTH_FEB-2026`
- `RISK_LOW`

Target columns are the 10 schedule day labels.

## 9. Why Blank Predictions Still Happen

Even with 3-month history, blank predictions can still happen when:

- the rolled history is weak or inconsistent
- the account historically had many blank/no-contact months
- the strongest learned class for a day is `-`

Blank prediction does not mean the row was missing.  
It usually means:

- the model received a valid feature row
- but the most likely label for that day was still `-`

## 10. Important Limitation to Remember

This model is still learning from historical strategy assignment, not directly from business outcome like:

- payment made
- cure
- response success

So it is mainly learning:

- what strategy was historically used next

not necessarily:

- what strategy is truly optimal next

That is good for replicating past campaign behavior, but it is not yet a response-optimization model.

## 11. Current Code Paths

Training:

- [train_next_month_strategy_model_catboost.py](/home/ubuntu/aiml/recommendation/scripts/train_next_month_strategy_model_catboost.py)

Inference-only:

- [predict_next_month_strategy_catboost.py](/home/ubuntu/aiml/recommendation/scripts/predict_next_month_strategy_catboost.py)

Key parameter:

- `--history-window-months 3`

## 12. Practical Summary

The new 3-month design works like this:

- start from monthly engineered features
- roll the latest `3` months together per account
- use that rolled row as the source feature vector
- join it to the next month schedule during training
- train 10 CatBoost day-wise models
- use the same 3-month rolling logic during inference

So for `APR-2026 -> MAY-2026`, the model does not look only at April anymore.  
It looks at the latest `3` months available for that account ending in April, then predicts the May schedule.
