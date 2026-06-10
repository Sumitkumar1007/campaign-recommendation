# Legacy Archive

This folder contains scripts that are not used by the API-based server runtime.

Archived under `legacy/api_unused_scripts/`:
- `benchmark_next_month_models.py`
- `explain_strategy_prediction.py`
- `generate_stakeholder_ppt.py`
- `mlflow_catboost_strategy_model.py`
- `register_catboost_model_mlflow.py`
- `train_strategy_model.py`
- `train_strategy_model_low_ram.py`
- `write_target_db_outputs.py`

Not archived yet:
- `scripts/export_mcollect_scheduler.py`: conditional API dependency when export is enabled.
- `scripts/quartz_job_data.py`: still imported by export code and tests.
- `src/campaign_recommendation/recommend.py` and related legacy package modules: still referenced by tests and the package entry point.
