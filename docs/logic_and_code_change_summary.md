# Logic and Code Change Summary

Last updated: August 10, 2026

## Purpose

This document summarizes the major logic and code changes implemented in the campaign recommendation project so far. It is intended as a working implementation summary for engineering, QA, deployment, and stakeholder handoff.

## 1. API and Service Layer Changes

### 1.1 API workflow and contract alignment
- Standardized authentication, training, inference, health, and transaction-status API flows.
- Updated bad-request and processing responses to return business-readable JSON payloads instead of low-level Python/subprocess stack fragments.
- Improved transaction lifecycle handling so Digital can create transaction IDs externally and AIML updates the same transaction state instead of assuming it owns the insert flow.
- Added safer wait-and-retry behavior around `ai_configurations` updates to handle timing gaps between Digital inserts and AIML updates.

Primary files:
- `src/campaign_recommendation/api_service.py`
- `scripts/run_api_service.py`

### 1.2 Audit and status tracking
- API status updates were aligned with `ai_configurations` and `api_audit_log` usage.
- Added support for maintaining API audit entries without changing the core AIML transaction update flow.
- Improved stored status/message handling so completion and failure payloads are persisted in a consistent JSON-oriented format.
- Added better logging around training and inference subprocess steps.

### 1.3 Model version reporting
- Training and inference API responses were updated so model versions are tied to training lifecycle instead of static env-only values.
- Accepted training requests can continue to report the currently active version, while completed successful training updates the effective model version.

## 2. Data Fetching and Month Selection Changes

### 2.1 Communication extraction improvements
- Communication fetch logic was updated to support multiple EMI cycle dates from `data_config` instead of a single static day.
- Communication CSV naming was changed to include month and EMI cycle day so files for different cycles do not overwrite each other.
- The pipeline now supports communication file discovery using both current and legacy naming patterns.
- Vertical column name for communication extraction was made configurable.

Primary files:
- `scripts/fetch_month_from_postgres.py`
- `scripts/prepare_training_window_from_postgres.py`
- `.env.example`

### 2.2 Cases extraction improvements
- Case extraction was updated to use configurable case-table and vertical-column configuration.
- Cases are now aligned to the EMI month derived from config-driven inference rules instead of earlier fixed assumptions.

Primary files:
- `scripts/fetch_cases_from_postgres.py`

### 2.3 Source/prediction month alignment
- Inference month logic was corrected so source month, prediction month, case month, and communication history month use a consistent rule based on configured EMI dates.
- History fetching was generalized from hardcoded 3-month behavior to configurable `history_window_months` behavior.

Primary files:
- `scripts/run_monthly_inference_pipeline.py`
- `scripts/pipeline_common.py`

## 3. Feature Engineering Changes

### 3.1 Day-window expansion
- The feature-generation window was expanded from `D-5` to `D+5` into `D-5` to `D+20`, excluding `D` for training targets.
- All downstream schedule, feature, prediction, and target-day handling were updated to support the expanded day range.

Primary files:
- `scripts/generate_strategy_dataset.py`
- `scripts/build_strategy_schedule_dataset.py`
- `scripts/build_monthly_feature_dataset.py`
- `scripts/pipeline_common.py`
- `scripts/train_next_month_strategy_model_catboost.py`
- `scripts/predict_next_month_strategy_catboost.py`
- `scripts/run_monthly_inference_pipeline.py`

### 3.2 Channel coverage expansion
- Added `VOICE_BOT` as a first-class supported communication channel in feature generation, prediction, scheduling, and vendor-selection paths.
- Added channel mappings for `VOICE_BOT` wherever channel-specific logic existed.

Primary files:
- `scripts/generate_strategy_dataset.py`
- `scripts/run_monthly_inference_pipeline.py`
- `scripts/export_mcollect_scheduler.py`
- `scripts/predict_next_month_strategy_catboost.py`

### 3.3 Dynamic vendor handling
- Vendor resolution was changed from static fallback behavior toward `data_config`-driven vendor selection by channel.
- Added support for `voice-bot.service.vendor-list` and other channel-specific vendor keys.
- Improved vendor parsing so names like `value-first` and `micro-*` are preserved correctly.

Primary files:
- `scripts/run_monthly_inference_pipeline.py`
- `scripts/export_mcollect_scheduler.py`

## 4. Training Pipeline Changes

### 4.1 Training-data preparation
- Training preparation now supports configurable vertical-column selection and updated communication file naming.
- Better logging was added for training preparation failures so user-facing messages are understandable.

Primary files:
- `scripts/prepare_training_window_from_postgres.py`
- `src/campaign_recommendation/api_service.py`

### 4.2 CatBoost training robustness
- Added fallback handling for missing day data within the `D-5` to `D+20` window.
- Added fallback handling for single-class day targets.
- Fallback behavior is now logged clearly at per-day model level.
- Added support for checkpoint reuse and stale-checkpoint invalidation by metadata.
- Added legacy joblib alias registration so saved fallback model classes continue to load correctly during training, inference, and drift checks.

Primary files:
- `scripts/train_next_month_strategy_model_catboost.py`
- `scripts/model_fallbacks.py`
- `scripts/predict_next_month_strategy_catboost.py`
- `scripts/run_monthly_inference_pipeline.py`

### 4.3 Month split behavior
- Auto-splitting of train/validation/test months was updated to reserve an actual test month when enough source history exists.
- Previous behavior could leave `test` empty even when enough months were available.

Primary files:
- `scripts/train_next_month_strategy_model_catboost.py`

### 4.4 Weighted training support
- Wired configurable status-based weighting into the active CatBoost path.
- Added env-driven toggle to enable or disable weighting without code changes.
- Weighted success signals now influence feature generation, predicted strategy selection, and CatBoost sample weights when enabled.
- Monthly aggregate feature tables explicitly avoid leaking target weight columns as predictor features.

Primary files:
- `scripts/generate_strategy_dataset.py`
- `scripts/build_strategy_schedule_dataset.py`
- `scripts/build_monthly_feature_dataset.py`
- `scripts/pipeline_common.py`
- `scripts/train_next_month_strategy_model_catboost.py`
- `.env.example`

## 5. Model Versioning and Artifact Management

### 5.1 Versioned artifacts
- Training and inference were updated toward explicit model-version-aware artifact naming.
- Added support for version-specific model, metrics, and checkpoint paths instead of relying only on generic static filenames.
- API and inference flows were aligned to load version-specific artifacts when supplied.

Primary files:
- `scripts/train_next_month_strategy_model_catboost.py`
- `scripts/run_monthly_inference_pipeline.py`
- `src/campaign_recommendation/api_service.py`

### 5.2 Environment-driven model resolution
- Added support for `MODEL_VERSION`, `MODEL_FILE`, and `METRICS_FILE` overrides in inference.
- Added `VENV_PYTHON` override support in the API service so different server layouts can be supported without hardcoding `/home/ubuntu/...` assumptions.

Primary files:
- `scripts/run_monthly_inference_pipeline.py`
- `src/campaign_recommendation/api_service.py`

## 6. Inference Pipeline Changes

### 6.1 Async and concurrency improvements
- Training and inference request handling was made asynchronous at API-service level so requests can be accepted quickly and processed in the background.
- Gunicorn/process-management changes were introduced to improve concurrent request handling and reduce timeouts under multiple requests.

Primary files:
- `src/campaign_recommendation/api_service.py`
- deployment/service configuration
- project dependency config

### 6.2 Prediction population and baseline handling
- Inference population selection was aligned to use the latest valid feature rows up to the source month.
- Blank-history rows are now tracked separately for reporting.

Primary files:
- `scripts/predict_next_month_strategy_catboost.py`
- `scripts/run_monthly_inference_pipeline.py`

### 6.3 Prediction reason improvements
- Business-facing reasoning text was made more readable and less technical.
- “No campaign” narratives were rewritten in business language for pre-due and post-due stages.
- Prediction evidence generation was introduced to support business validation of why recommendations were made.

Primary files:
- `scripts/predict_next_month_strategy_catboost.py`
- `scripts/generate_prediction_evidence.py`
- `docs/examples/prediction_evidence_sample.csv`
- `docs/examples/prediction_evidence_sample.md`

## 7. Prediction Evidence and Explainability Changes

### 7.1 Evidence CSV generation
- Added a standalone evidence-generation path and integrated it into inference.
- Evidence file includes top recommendations, success signal columns, narrative text, and historical support features.
- Added point-in-time logic and later adjustments for same-day/non-cumulative evidence interpretation.

Primary files:
- `scripts/generate_prediction_evidence.py`
- `scripts/predict_next_month_strategy_catboost.py`
- `scripts/run_monthly_inference_pipeline.py`

### 7.2 Business-readable explanations
- Added human-readable action text and business narratives.
- Added support-percentage and feature-name references to help campaign stakeholders validate recommended strategies.

## 8. Drift Logic Changes

### 8.1 Drift calculation stabilization
- Identified and fixed multiple drift-comparison bugs:
  - stale baseline month selection
  - chronological ordering issues in baseline month resolution
  - target-only baseline restriction that excluded the active source month
  - use of different account populations for baseline vs inference
  - inclusion of `SOURCE_MONTH_*` dummy columns in drift PSI
- Drift baseline now uses a fixed stabilized logic: latest source month, matched account population, and exclusion of `SOURCE_MONTH_*` dummy features.
- After these fixes, previously inflated drift values were reduced to near-zero in like-for-like validation.

Primary files:
- `scripts/run_monthly_inference_pipeline.py`
- `.env.example`

### 8.2 Drift simplification
- Intermediate drift env configurability was later removed after the comparison logic was standardized in code.

## 9. Export and Scheduler Changes

### 9.1 Scheduler and dataset handling
- Separated local export generation from SFTP destination handling.
- Added support for separate remote dataset and scheduler paths while keeping local export path single.
- Improved scheduler naming and operational integration with MCollect export flow.

Primary files:
- `scripts/export_recommendation_workbooks.py`
- `scripts/export_mcollect_scheduler.py`
- `deploy/systemd/recommendation.service`

### 9.2 Schema updates
- Pipeline/export logic was adapted to newer campaign recommendation and mapping table structures, including support for added ID columns.

## 10. Auth, Error Handling, and Logging Improvements

### 10.1 Authentication responses
- Wrong-credential and auth-failure handling was improved so client apps receive meaningful JSON messages.
- Unknown endpoints and route handling were adjusted to avoid misleading unauthorized responses where 404 should apply.

### 10.2 Logging quality
- Added more detailed step logs around training preparation, inference execution, child-script failures, and drift computation.
- Improved propagation of actual subprocess errors into readable status messages.

## 11. Documentation Added or Updated

New or significantly updated documentation includes:
- `docs/aiml_api_contract.md`
- `docs/aiml_integration_api.md`
- `docs/digital_team_api_guide.md`
- `docs/implementation_and_deployment_guide.md`
- `docs/client_pitch_campaign_recommendation.md`
- `docs/recommendation_model_end_to_end_data_flow.md`
- `docs/three_month_catboost_training_explainer.md`
- evidence examples under `docs/examples/`

## 12. Test and Validation Changes

- Added or updated tests for API behavior and monthly pipeline behavior.
- Added validation probes during development for drift, month alignment, feature generation, and evidence-file correctness.
- Added compile/import smoke validation for touched scripts during implementation.

Primary files:
- `tests/test_api_service.py`
- `tests/test_monthly_pipeline.py`

## 14. Removed or Simplified Logic and Environment Variables

The following logic/config items were removed or simplified because they were either unnecessary, confusing, or no longer required after pipeline stabilization.

### 14.1 Removed drift env configurability
- Removed env-driven drift knobs from `.env.example` and standardized drift logic directly in code.
- Removed the need to manage:
  - `DRIFT_BASELINE_MODE`
  - `DRIFT_EXCLUDE_FEATURE_PREFIXES`
  - `DRIFT_EXCLUDE_FEATURE_COLUMNS`
- Drift now always follows the fixed supported logic:
  - compare against the latest source month used by inference
  - match baseline and inference on the same account population
  - exclude `SOURCE_MONTH_*` dummy features from PSI

### 14.2 Removed obsolete drift comparison paths
- Removed practical dependence on stale training-month-driven drift baseline behavior.
- Removed dependence on target-only baseline restriction for drift.
- Removed configuration-driven branching for drift feature exclusion because it was not needed operationally.

### 14.3 Simplified environment surface
- Cleaned the env template so only operationally needed variables remain documented.
- Standardized the serving-mode example key to `MODEL_SERVING` instead of the older lowercase `modelserving` example.
- Removed unused MLflow template variables that were not consumed by the active pipeline:
  - `MLFLOW_EXPERIMENT_NAME`
  - `MLFLOW_RUN_NAME`
- Drift behavior is now code-standardized rather than server-configurable.

## 13. Remaining Notes

- This summary reflects the implemented code changes present in the working repository state as of August 10, 2026.
- Generated artifacts, local backups, and environment secrets are intentionally excluded from this summary except where they affected code or logic behavior.
- Some changes were architectural corrections, while others were business-rule changes requested for Digital and campaign-team workflows.
