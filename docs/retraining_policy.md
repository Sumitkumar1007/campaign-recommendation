# Model Retraining Policy

## Current Inference Contract

Current production model is a next-month model.

- Use `SOURCE_MONTH` as latest completed month with communication data.
- Use `PREDICT_MONTH` as exactly next month.
- Example: predict June 2026 with `SOURCE_MONTH=2026-05` and `PREDICT_MONTH=2026-06`.

Monthly inference does not need retraining every month. It loads existing model bundle from:

```text
artifacts/models/next_month_strategy_catboost_3m.joblib
```

Then it scores latest source-month features.

## When June Prediction Needs Retraining

For June 2026 prediction, retraining is not required if:

- trained model bundle already exists,
- feature schema produced by pipeline matches model bundle,
- source data shape and business rules did not change,
- recent model monitoring is healthy.

Run:

```bash
python scripts/run_monthly_inference_pipeline.py \
  --source-month 2026-05 \
  --predict-month 2026-06 \
  --model catboost_3m
```

## When To Retrain

Retrain when any trigger below happens:

- Scheduled retrain window arrives, usually monthly or quarterly after actual outcomes for latest month are available.
- Model quality drops versus validation/backtest threshold.
- Data drift is visible: risk mix, channel mix, language mix, success rate, or customer population changes materially.
- Feature schema changes.
- Business rules change, such as risk quota, allowed channels, success statuses, or D-window.
- New source table logic changes, such as date filter, status mapping, or source column semantics.
- Model becomes stale, for example trained on months too far behind current operation.

## Recommended Production Cadence

Use two separate jobs:

1. Monthly inference job
   - Runs when latest source month is complete and reconciled.
   - Does not retrain.
   - Writes feature snapshots and prediction snapshots.

2. Retraining job
   - Runs after target-month outcomes are available.
   - Backtests candidate model on future months.
   - Registers/promotes only if metrics pass acceptance gates.

## 3-Month History Note

`catboost_3m` rolls latest 3 months of features per account. For best inference quality, local/processed feature input should include the needed history window, not only one isolated month.

For June 2026 using a 3-month model, source history should include March, April, and May 2026 where available. If only May exists for an account, pipeline can still score with partial history, but it is weaker than intended.

## Operational Rule

Do not retrain just because calendar moved to next month. Retrain because new labeled outcomes are available and model monitoring says a new candidate is better or current model is stale.
