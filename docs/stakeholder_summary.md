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

- EMI date
- risk
- collectable amount / outstanding balance proxy
- campaign type
- previous-month customer communication history
- previous-month day-level activity availability

## What “Success” Means

The current target is **communication success**, not direct payment conversion.

It is channel-aware:

- `SMS` and `WhatsApp` treat `READ` and `CLICKED` as stronger signals than `DELIVERED`
- `VOICE` uses connection-based positive outcomes

## Current Best Model

The project benchmarked multiple approaches and selected `Extra Trees` as the current production default because it performed better than the legacy baseline.

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
