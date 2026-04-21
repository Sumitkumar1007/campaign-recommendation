# Campaign Recommendation Stakeholder Summary

## What This Project Is

This project recommends EMI communication strategies for loan accounts across the due-date window.

The system decides which communication type and send time should be used on each day from `D-5` to `D+5`, excluding `D`.

## Business Rule Applied

The recommendation limit is applied **per day**:

- `low` risk: up to `1` communication
- `medium` risk: up to `2` communications
- `high` risk: up to `3` communications

If a customer does not have prior signal for a particular day, that day is returned as `-`.

## What The Model Learns From

The model uses:

- latest 3 months of account-level communication history
- channel intensity by SMS, WhatsApp, and IVR/VOICE
- channel/time/language success and failure signals
- source risk bucket (provided)
- source month, used to predict exactly the next month

## What “Success” Means

The current target is a **next-month communication schedule**, not direct payment conversion.

Targets are built from prior successful communication patterns:

- `SMS` and `WhatsApp` use delivered/read/clicked signals
- `VOICE` is exposed as `IVR` in recommendation output and uses connection-based signals

## Current Best Model

The current production model is the 3-month CatBoost next-month strategy model:

```text
campaign_next_month_catboost_3m
```

It produces one schedule row per loan account for `D-5` through `D+5`, with `D` forced to `-`.

Current validation metrics should be read as schedule guidance quality, not payment-lift measurement. The latest saved CatBoost 3-month metrics use November 2025 through January 2026 as training source months and February 2026 as validation source month; no separate held-out test month is saved in the current metrics artifact.

## Main Deliverables

The project currently produces:

- a trained production model artifact
- day-wise recommendation calendar
- bucket count summaries
- account-level explanation outputs
- benchmark reports
- technical implementation and deployment guide

## What This Enables Operationally

The collections team can use the output to:

- decide which channels to use on each EMI-relative day
- control communication intensity by risk bucket
- identify blank days where no prior day-level signal exists
- view strategy mix counts across risk buckets

## Current Status

This is now in a production-oriented batch-inference form and can be extended further with:

- more account features like `POS` and `bucket`
- faster precomputed feature serving
- API-based inference
- richer business report exports
