# Campaign Recommendation

ML-based EMI campaign recommendation engine for day-wise communication strategy planning.

This project predicts the best communication strategy across the EMI window `D-5` to `D+5`, excluding `D`, and applies business quotas **per day**:

- `low` risk: up to `1` communication per day
- `medium` risk: up to `2` communications per day
- `high` risk: up to `3` communications per day

If previous-month signal is unavailable for a given day for a customer, the system returns `-` for that day.

## What This Project Does

- filters training data to EMI date `5th` only
- builds channel-aware communication-success targets
- supports weighted success scoring where `CLICKED` and `READ` are stronger than `DELIVERED`
- creates previous-month customer history and day-availability features
- trains and evaluates models using a month-based time split
- benchmarks multiple model families
- uses `extra_trees` as the current production default
- generates day-by-day strategy calendars, bucket summaries, and account-level explanations

## Current Production Model

The current default production model is `ExtraTreesClassifier` with probability calibration.

Why it was selected:

- it outperformed the legacy baseline in the benchmark experiments
- it achieved stronger ranking and calibrated probability quality on the held-out test month

Benchmark references:

- [Legacy vs Extra Trees](reports/legacy_vs_extra_trees.md)
- [Model Benchmark Experiments](reports/model_benchmark_experiments.md)

## Project Structure

- `src/campaign_recommendation/`
  Core package for training, recommendation generation, explainability, and benchmarking.
- `config/`
  Project configuration.
- `training-data/`
  Raw input dataset tracked with Git LFS.
- `outputs/`
  Generated model artifacts, recommendations, summaries, and account explanations.
- `reports/`
  Benchmark and comparison reports.
- `docs/`
  Technical handoff and deployment documentation.

## Key Outputs

- `outputs/model/model.joblib`
  Serialized trained model artifact.
- `outputs/model/metrics.json`
  Validation and test metrics for the current production model.
- `outputs/recommendations.csv`
  Day-wise strategy output across the EMI window.
- `outputs/bucket_counts.csv`
  Overall count by `risk + strategy label`.
- `outputs/bucket_counts_by_day.csv`
  Count by `risk + day + strategy label`.

## Commands

Train the current production model:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation train --max-rows 80000
```

Generate recommendations:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation recommend --model-dir outputs\model
```

Explain one account:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation explain --model-dir outputs\model --account-id MFLAPDSECUL000005010246
```

Run model benchmarks:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation benchmark-models --max-rows 80000
```

## Documentation

Additional CatBoost 3-month inference utilities are available under `scripts/`:

- `scripts/train_next_month_strategy_model_catboost.py`
  Trains the next-month CatBoost bundle with rolling 3-month account history.
- `scripts/predict_next_month_strategy_catboost.py`
  Runs inference only from a saved CatBoost bundle.
- `scripts/run_monthly_inference_pipeline.py`
  Fetches one source month from Postgres, refreshes processed local data, runs inference, and stores snapshots.
- `docs/three_month_catboost_training_explainer.md`
  Explains the 3-month feature window and model input shape.
- `docs/application_logging.md`
  Describes application logging and operational checks.
- `artifacts/analysis/may_2026_all_blank_27_apac_analysis.md`
  Reviews the 27 May 2026 all-blank APAC predictions.

For Postgres-backed monthly inference, pass credentials through environment variables or deployment secrets. Do not commit database passwords.

Technical handoff guide:

- [Implementation And Deployment Guide (Markdown)](docs/implementation_and_deployment_guide.md)
- [Implementation And Deployment Guide (PDF)](docs/implementation_and_deployment_guide.pdf)

Business summary:

- [Stakeholder Summary](docs/stakeholder_summary.md)

## Notes

- the current target is communication success, not direct payment conversion
- `VOICE` is included as a channel, but payment linkage for IVR is not available in the source data
- validation and testing are month-based, not random-row based
- generated outputs are not committed by default

## GitHub Repo Description

Suggested repo description:

`ML-based EMI campaign recommendation engine with day-wise strategy generation, benchmarking, explainability, and production-ready batch inference.`

Suggested repo tags:

`machine-learning`, `recommendation-system`, `campaign-optimization`, `collections`, `python`, `scikit-learn`
