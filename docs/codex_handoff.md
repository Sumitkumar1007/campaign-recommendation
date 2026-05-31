# Codex Handoff

## Project
Campaign recommendation ML pipeline with MCollect Digital scheduler integration.

## Current Branch
`feat/integration-with-digital`

## Operating Rules
- GitHub only. Do not push to or operate on the GitLab remote unless explicitly asked.
- Prefer targeted `rg` searches and focused file reads; avoid broad repo analysis unless needed.
- Use `./venv/bin/python` for project scripts/tests.
- Keep generated data, model files, logs, predictions, and `.env` out of commits.

## Key Entry Points
- Monthly inference: `scripts/run_monthly_inference_pipeline.py`
- MCollect export: `scripts/export_mcollect_scheduler.py`
- Quartz job data serializer: `scripts/quartz_job_data.py`
- Main tests: `tests/test_monthly_pipeline.py`

## Current Flow
1. Run monthly inference for a source month and next prediction month.
2. Pipeline writes staging/output rows to Postgres:
   - `digital_collections.ai_ml_campaign_recommendations`
   - `digital_collections.ai_ml_campaign_mapping`
3. Export script converts staging rows into MCollect Digital tables:
   - `dataset`
   - `qrtz_job_details`
   - `qrtz_triggers`
   - `qrtz_cron_triggers`
4. `digital_rules` templates are managed manually as a one-time MCollect setup and are only referenced by template name from campaign rows.

## Important Decisions
- Naming format now uses:
  - Scheduler: `PREDUE_AIML_SMS_LAP_ENGLISH_MR_5TH_KALEYRA_200526_2`
  - Template: `PREDUE_AIML_SMS_ENGLISH`
  - Dataset: `PREDUE AIML SMS LAP ENGLISH MR EMI 5TH [D-4,D-2] 09`
- `qrtz_job_details.job_data` is Java serialized Quartz `JobDataMap`, not JSON/text.
- Direct Quartz DB writing is temporary. Replace `DirectDatabaseMcollectPublisher` if/when MCollect provides an API.
- `scripts/export_mcollect_scheduler.py` is dry-run by default; actual writes require `--write`.
- `digital_rules` rows are not created by this repo. Templates/verbiages are predefined and created manually once in MCollect using names like `PREDUE_AIML_SMS_ENGLISH`.
- Use `--trigger-state PAUSED` first for MCollect review before live scheduler execution.
- Latest known monthly run: `APR-2026 -> MAY-2026`, model `catboost_3m`.

## Known Risks
- Confirm MCollect job class mapping before production writes:
  - `SMS -> com.company.api_test.jobs.BatchSMSJob`
  - `WHATSAPP -> com.company.api_test.jobs.BatchWhatsappJob`
  - `VOICE -> com.company.api_test.jobs.BatchVoiceJob`
- Quartz cron dates are generated from EMI-relative labels like `D-5`, `D-1`, `D+5` using `emi_cycle`.

## Useful Commands
```bash
./venv/bin/python scripts/run_monthly_inference_pipeline.py \
  --source-month 2026-04 \
  --predict-month 2026-05 \
  --model catboost_3m

./venv/bin/python scripts/export_mcollect_scheduler.py \
  --source-month 2026-04 \
  --prediction-month 2026-05 \
  --model catboost_3m

./venv/bin/python scripts/export_mcollect_scheduler.py \
  --source-month 2026-04 \
  --prediction-month 2026-05 \
  --model catboost_3m \
  --trigger-state PAUSED \
  --write

./venv/bin/python -m pytest -q tests/test_monthly_pipeline.py
```

## Last Context Snapshot
- Monthly inference for `APR-2026 -> MAY-2026` completed successfully with `1` prediction row for `MOB-TEST-Sumit`.
- Current staged campaign output for that run is `14` campaigns and `14` mappings.
- Monthly inference stored `1` prediction snapshot, `14` campaign recommendations, `14` campaign mappings, and one `api_audit_log` row in Postgres.
- Export script now writes only `dataset`, `qrtz_job_details`, `qrtz_triggers`, and `qrtz_cron_triggers`. Existing `digital_rules` templates are referenced by name and remain manually managed.
- Current export dry run for `APR-2026 -> MAY-2026`: `14` campaigns, `7` datasets, `4` template refs, `14` jobs, `14` triggers.
