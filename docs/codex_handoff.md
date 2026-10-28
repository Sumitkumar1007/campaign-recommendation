# Codex Handoff

## Project
Campaign recommendation ML pipeline with MCollect Digital scheduler integration.

## Current Branch
`Production/csb-v1.1`

## Operating Rules
- GitHub only. Do not push to or operate on the GitLab remote unless explicitly asked.
- Prefer targeted `rg` searches and focused file reads; avoid broad repo analysis unless needed.
- Use `./venv/bin/python` for project scripts/tests. Install repo from `pyproject.toml` via `pip install -e .`; no `requirements.txt`.
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
- Payment weighting distinguishes between Link Payments (`c.payment_unique_id == p.reference_number`, default `5.0x` multiplier) and External Payments (account payment on communication date, default `2.5x` multiplier). Multipliers are configurable in `config/weights_config.json` (`payment_multipliers`) or via `.env` (`STRATEGY_LINK_PAYMENT_MULTIPLIER`, `STRATEGY_EXTERNAL_PAYMENT_MULTIPLIER`).
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

./venv/bin/python -m pytest -q

```

## Last Context Snapshot
- Monthly inference for `APR-2026 -> MAY-2026` completed successfully with `1` prediction row for `MOB-TEST-Sumit`.
- Current staged campaign output for that run is `14` campaigns and `14` mappings.
- Monthly inference stored `1` prediction snapshot, `14` campaign recommendations, `14` campaign mappings, and updated one `ai_configurations` row in Postgres.
- Export script now writes only `dataset`, `qrtz_job_details`, `qrtz_triggers`, and `qrtz_cron_triggers`. Existing `digital_rules` templates are referenced by name and remain manually managed.
- Current export dry run for `APR-2026 -> MAY-2026`: `14` campaigns, `7` datasets, `4` template refs, `14` jobs, `14` triggers.
- Production inference run for `SEP-2026 -> OCT-2026` completed with 8,127 prediction rows, 405 campaigns, 38,154 mappings, and 100% template resolution across 80,112 touchpoints.
- Fixed `--dry-run` AttributeError in `run_monthly_inference_pipeline.py`.
- Added `--auto-class-weights` and class distribution reporting in `train_next_month_strategy_model_catboost.py`.
- Overhauled recommendation reasoning: eliminated `"Reason not available..."` placeholders, added structured JSON cold-start fallback reasons, loan key normalization, and bank holiday suppression explanations. Chat history saved to `docs/chat_history_2026_10_19.md`.
- Multi-mode post-due prediction (`predict_distinct_modes_by_risk`) with anti-collision scheduling, channel-aligned reasoning (`_align_reason_to_mode`), and cold-start support.
- Model probability-ranked pre-due candidate day selection: ranks candidate days ($D-4 \dots D-1$) by model probability rather than positional order, preventing $D-1$ from being mechanically dropped.
- 2nd-best fallback protection: active post-due models select the highest probability non-blank class when top prediction is `"-"`.
- Model-driven $D-5$ baseline: selects highest probability non-dash candidate from model probabilities for 100% base reach.
- Strict regional language template protection: regional predictions (Marathi, Hindi, Tamil, Telugu, etc.) never fall back to DEFAULT or ENGLISH templates in `digital_rules`.
- Missing communication data resolved: retained `UNKNOWN` risk communication events in `generate_strategy_dataset.py`, with safe mapping to `LR` (Low Risk) in campaign grouping.
- Two-stage prediction outputs: `<stem>_raw.csv` (unfiltered) and `<stem>.csv` (business-filtered).
- Dynamic `send_hour_window.step_hours` prediction snapping: Keeps `<stem>_raw.csv` untouched with original 1-hour steps for auditability, while `<stem>.csv`, cold-start fallbacks, reasons, and schedulers are snapped cleanly to candidate step hours (e.g., 3-hour steps: 9AM, 12PM, 3PM, 6PM).
- Comprehensive branch-wise logic and changelog maintained in `docs/csb_client_branch_and_postdue_logic_documentation.md` (Version 1.1, 205 passing tests).

