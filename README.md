# Campaign Recommendation

Production-oriented first version of an ML-based campaign suggestion engine for EMI communications.

The v1 objective is to predict `communication success` for candidate outreach slots and recommend:

- `low` risk: top 1 communication
- `medium` risk: top 2 communications
- `high` risk: top 3 communications

Recommendations are generated for the EMI window `D-5` to `D+5`, excluding `D`.

## What v1 does

- reads raw communication-history CSV data in chunks
- keeps only accounts with `emi_date` on the 5th
- derives previous-month communication and payment history features for each customer
- derives a channel-aware binary success label from `comm_status`
- trains a baseline ML model to predict success probability
- splits train, validation, and test by `emi_date` month
- saves model artifacts and training metadata
- scores candidate campaign slots for a base population
- returns top-N recommendations by risk band

## Project Layout

- `src/campaign_recommendation/`: application package
- `config/`: configuration files
- `outputs/`: generated reports and recommendations
- `training-data/`: source training data

## Quick Start

Create or activate your Python environment, then from this folder run:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation train --max-rows 200000
```

Generate recommendations from the saved model:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation recommend --model-dir outputs\model
```

Explain one loan account and export all 10 day-level predictions:

```powershell
$env:PYTHONPATH="src"
d:\aiml\venv\Scripts\python.exe -m campaign_recommendation explain --model-dir outputs\model --account-id MFLAPDSECUL000005003456
```

## Notes

- v1 is a production-style baseline, not the final model.
- the current target is `communication success`, not payment conversion
- validation and test are time-based, not random-row based
- the model is trained from prior-month history plus current EMI context
- `VOICE` is included, but because IVR payment outcomes are unavailable, the model optimizes for communication success only
- training writes a `validation_slice_report.csv` to support production validation by risk, channel, and day offset
- account-level explanation exports the best recommendation for each day in the 10-day window plus feature contribution details for the final selected option
- the recommendation engine uses candidate slot generation plus model-based ranking, then applies business caps by risk
