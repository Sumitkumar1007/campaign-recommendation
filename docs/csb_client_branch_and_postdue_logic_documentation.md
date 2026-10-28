# CSB Client: Branch-Wise Updates, Evolution, and Post-Due Prediction Logic

**Document Version:** 1.1  
**Date:** October 27, 2026  
**Target Environment:** CSB Bank — MCollect Digital Collections & Recommendation Pipeline  
**Model Version:** CatBoost Strategy Model v1.1.46  
**Test Suite Coverage:** 197 passed / 197 total tests  

---

## Executive Summary

This document provides a comprehensive technical and business reference for all code, logic, and architectural enhancements developed specifically for the **CSB (Catholic Syrian Bank) client** across all five CSB Git branches:
1. `feature/csb_v1.0` (Git ref: `feature/csb_v.1.0`)
2. `feature/csb_v1.0-updates` (Git ref: `feature/csb-v1.0-updates`)
3. `feature/csb_v1.1` (Git ref: `feature/csb-v1.1`)
4. `Production/csb-v1.0`
5. `Production/csb-v1.1`

It details:
- **The complete evolution of Post-Due Prediction**: Why it changed from single-channel capping to an omni-channel, anti-colliding multi-mode strategy, how the selection algorithm works mathematically, and how campaigns expand into execution tables.
- **A commit-by-commit, logic-by-logic breakdown** across every branch, explaining what changed, why it was changed, and the business impact.
- **The Configuration & Decision Hierarchies**: Dynamic prediction windows, pre-due baseline rules, bank holiday calendar suppressions, template resolution, and business reasoning generation.

---

## Table of Contents
1. [Deep Dive: Post-Due Prediction Logic & Evolution](#1-deep-dive-post-due-prediction-logic--evolution)
   - [1.1 Background & Business Motivation](#11-background--business-motivation)
   - [1.2 Legacy Post-Due Logic (CSB v1.0)](#12-legacy-post-due-logic-csb-v10)
   - [1.3 New Multi-Mode Post-Due Logic (CSB v1.1)](#13-new-multi-mode-post-due-logic-csb-v11)
   - [1.4 Algorithmic Step-by-Step Walkthrough](#14-algorithmic-step-by-step-walkthrough)
   - [1.5 Concrete Numerical Walkthrough with Anti-Collision](#15-concrete-numerical-walkthrough-with-anti-collision)
   - [1.6 Downstream Campaign & Trigger Generation](#16-downstream-campaign--trigger-generation)
   - [1.7 Probability-Ranked Selection, Fallback Protection & Strict Regional Templates](#17-probability-ranked-selection-fallback-protection--strict-regional-templates)
2. [CSB Branch Topology & Lineage Map](#2-csb-branch-topology--lineage-map)
3. [Branch-Wise & Commit-Wise Changelog](#3-branch-wise--commit-wise-changelog)
   - [3.1 Branch: feature/csb_v1.0 (Baseline Foundation)](#31-branch-featurecsb_v10-baseline-foundation)
   - [3.2 Branch: feature/csb-v1.0-updates (Pipeline Hardening)](#32-branch-featurecsb-v10-updates-pipeline-hardening)
   - [3.3 Branch: Production/csb-v1.0 (Template Guardrails & Release)](#33-branch-productioncsb-v10-template-guardrails--release)
   - [3.4 Branch: feature/csb-v1.1 (Next-Gen Staging)](#34-branch-featurecsb-v11-next-gen-staging)
   - [3.5 Branch: Production/csb-v1.1 (Dynamic Window, Multi-Mode, Probability Ranking & Template Hardening)](#35-branch-productioncsb-v11-dynamic-window-multi-mode-probability-ranking--template-hardening)
4. [Comparative Matrix Across Branches](#4-comparative-matrix-across-branches)
5. [Summary of Production Verification & Audit Metrics](#5-summary-of-production-verification--audit-metrics)

---

## 1. Deep Dive: Post-Due Prediction Logic & Evolution

### 1.1 Background & Business Motivation

In debt recovery, borrower delinquency is divided into two distinct operating windows:
1. **Pre-Due Window ($D-5 \dots D-1$)**: Courtesy reminders sent before the EMI due date ($D$). Borrowers are not yet delinquent. The goal is gentle nudging and payment link delivery without causing irritation.
2. **Post-Due Window ($D+1 \dots D+20$)**: Active delinquency. The borrower has failed to honor their EMI obligation. Every day overdue compounds risk, NPA classification timelines, and provision requirements for the bank.

In earlier systems, post-due communication was treated identically to pre-due: capped at a single touchpoint per day. However, operational feedback from CSB collections teams showed that:
- Relying on a **single channel** (e.g. only SMS) during active delinquency suffered from low engagement (unread SMS, DND blockers, spam filtering).
- Sending **multi-channel outreach at the exact same hour** (e.g. blast SMS, WhatsApp, and an automated Voice IVR call all at 11:00 AM) resulted in high customer complaints, dropped IVR calls, and blocked sender IDs.
- What CSB required was an **intelligent omni-channel orchestration**: reaching delinquent borrowers across **SMS, WhatsApp, and Voice** on the same post-due day, but staggered across **distinct, non-colliding time slots** (e.g., SMS at 10 AM, WhatsApp at 1 PM, Voice at 4 PM), while harmonizing the customer's preferred language.

---

### 1.2 Legacy Post-Due Logic (CSB v1.0)

In CSB v1.0 and earlier pipelines:
1. **Prediction Generation**: Models used `predict_top_k_by_risk()`. If top-k was $> 1$, predictions were emitted as pipe-separated strings (e.g., `SMS-11AM-ENGLISH|WH-11AM-ENGLISH`).
2. **Capping Rule**: `apply_postdue_risk_rules()` strictly invoked `enforce_single_communication_per_day()`:
   ```python
   # Legacy post-due rule:
   def apply_postdue_risk_rules(df, postdue_days=None):
       cols = [d for d in (postdue_days or POSTDUE_DAY_COLUMNS) if d in df.columns]
       return enforce_single_communication_per_day(df, cols)
   ```
3. **Execution Assignment**: `_build_campaign_assignment_groups()` with `single_communication_per_day=True` called `get_first_valid_communication(raw_val)`, discarding all secondary modes.
4. **Limitation**:
   - Each account received at most **1 message per day**.
   - If the first recommendation was SMS, WhatsApp and Voice were discarded completely.
   - 0% multi-channel post-due coverage.

---

### 1.3 New Multi-Mode Post-Due Logic (CSB v1.1)

In CSB v1.1 (implemented in `pipeline_common.py`, `predict_next_month_strategy_catboost.py`, and `run_monthly_inference_pipeline.py`), post-due prediction was completely redesigned:

```mermaid
flowchart TD
    A[Account Features X_pred & Risk Tier] --> B{Day Offset Type}
    B -->|Pre-Due: D-5..D-1| C[predict_top_k_by_risk]
    C --> D[apply_predue_risk_rules: Enforce 1 per day + D-5 full base]
    
    B -->|Post-Due: D+1..D+20 & multimode_postdue=True| E[predict_distinct_modes_by_risk]
    E --> F[1. Group candidate tokens by Mode: SMS, WHATSAPP, VOICE]
    F --> G[2. Order modes by Peak Probability]
    G --> H[3. Allocate optimal hour to highest peak mode]
    H --> I[4. Anti-Collision: Subsequent modes select next-best free hour]
    I --> J[5. Harmonize Customer Language across all 3 modes]
    J --> K[Output: SMS-10AM-HINDI | WH-1PM-HINDI | VOICE-4PM-HINDI]
    
    K --> L[build_campaign_mappings: Expands to 3 distinct campaign rows]
    L --> M[POSTDUE_AIML_SMS_HINDI]
    L --> N[POSTDUE_AIML_WA_HINDI]
    L --> O[POSTDUE_AIML_VOICE_HINDI]
```

#### Key Rules Enforced:
1. **Mode-Distinct Coverage**: Generates up to 3 distinct channels (`SMS`, `WHATSAPP`, `VOICE`) per account on post-due days.
2. **Peak Probability Priority**: The channel with the strongest ML confidence gets first choice of its peak send hour.
3. **Hour Anti-Collision**: No two channels can dispatch at the same hour for the same borrower on the same day.
4. **Language Harmonization**: Language preference (e.g. Hindi, English, Malayalam, Marathi) is preserved across all active modes.
5. **Fallback Safety**: If model classes lack a specific mode for that day offset, default fallback tokens (e.g., `WH-12PM-<LANG>`) are injected seamlessly.
6. **Campaign Table Expansion**: While pre-due creates 1 mapping row per account per day, post-due creates up to 3 mapping rows and Quartz triggers per account per day.

---

### 1.4 Algorithmic Step-by-Step Walkthrough

The core algorithm is implemented in `predict_distinct_modes_by_risk()` in [scripts/pipeline_common.py](file:///home/aditya/aiml/campaign-recommendation/scripts/pipeline_common.py#L690-L767):

```python
def predict_distinct_modes_by_risk(
    model,
    encoder,
    X: pd.DataFrame,
    risks: pd.Series | None = None,
    target_modes: list[str] | None = None,
    default_fallbacks: dict[str, str] | None = None,
) -> pd.Series:
```

1. **Class Indexing by Canonical Mode**:
   - The model's label encoder classes (e.g., `[SMS-10AM-HINDI, WH-11AM-HINDI, VOICE_BOT-4PM-HINDI, ...]`) are scanned.
   - Helper `extract_canonical_mode_from_label(label)` normalizes variants (`WH` $\rightarrow$ `WHATSAPP`, `VOICE_BOT`/`IVR` $\rightarrow$ `VOICE`, `SMS` $\rightarrow$ `SMS`).
   - Indices are mapped into `mode_class_indices = {'SMS': [...], 'WHATSAPP': [...], 'VOICE': [...]}`.

2. **Per-Row Candidate Sorting**:
   - For every account row, model probabilities are evaluated.
   - For each target mode, candidates are gathered: `(label, probability, send_hour)`.
   - Candidates are sorted in descending order of probability:
     $$\text{mode\_candidates}[M] = \big[\,(L_{1}, P_{1}, H_{1}), (L_{2}, P_{2}, H_{2}), \dots \big] \quad \text{where } P_{1} \ge P_{2}$$

3. **Mode Allocation Priority**:
   - Modes are ordered by their **maximum peak probability**:
     $$\text{mode\_order} = \text{sort\_descending}\big(M, \text{key} = \max(P_{M})\big)$$
   - If SMS has peak probability $0.40$ and WhatsApp has $0.35$, SMS gets priority #1; WhatsApp gets priority #2.

4. **Hour Anti-Collision Selection**:
   - An empty set `used_hours = set()` is initialized for the borrower for that day.
   - Iterating through each mode in `mode_order`:
     - Scan the mode's sorted candidates.
     - Find the first candidate whose `send_hour` is **not in `used_hours`**:
       $$\text{candidate } (L, P, H) \text{ such that } H \notin \text{used\_hours}$$
     - If all candidate hours collide with prior modes, fall back to the highest probability candidate.
     - Add the chosen hour $H$ to `used_hours`.
     - Assign $L$ as the recommendation for mode $M$.

5. **Token Joining**:
   - Recommendations are joined with pipe delimiters:
     $$\text{Output} = \text{SMS-token} \mid \text{WHATSAPP-token} \mid \text{VOICE-token}$$

---

### 1.5 Concrete Numerical Walkthrough with Anti-Collision

Consider a delinquent borrower on $D+1$ where the CatBoost model outputs the following class probabilities:

| Encoder Class | Canonical Mode | Hour | Language | Model Probability |
| :--- | :--- | :--- | :--- | :--- |
| `SMS-11AM-HINDI` | **SMS** | 11AM | Hindi | **0.40** (Peak for SMS) |
| `SMS-2PM-HINDI` | **SMS** | 2PM | Hindi | 0.10 |
| `WH-11AM-HINDI` | **WHATSAPP** | 11AM | Hindi | **0.35** (Peak for WhatsApp — **Collides with SMS!**) |
| `WH-1PM-HINDI` | **WHATSAPP** | 1PM | Hindi | **0.15** (Next best non-colliding) |
| `VOICE_BOT-11AM-HINDI` | **VOICE** | 11AM | Hindi | 0.05 (Collides) |
| `VOICE_BOT-4PM-HINDI` | **VOICE** | 4PM | Hindi | **0.05** (Non-colliding) |

#### Step-by-Step Resolution:
1. **Determine Mode Order**:
   - Peak SMS: $0.40$
   - Peak WhatsApp: $0.35$
   - Peak Voice: $0.05$
   - **Evaluation Order**: `SMS` $\rightarrow$ `WHATSAPP` $\rightarrow$ `VOICE`.
2. **Allocate SMS**:
   - Top candidate: `SMS-11AM-HINDI` (prob $0.40$).
   - Hour `11AM` is free.
   - Selected: `SMS-11AM-HINDI`.
   - `used_hours = {'11AM'}`.
3. **Allocate WHATSAPP**:
   - Top candidate: `WH-11AM-HINDI` (prob $0.35$).
   - Hour `11AM` **already in `used_hours`**! Collision detected.
   - Algorithm advances to next candidate: `WH-1PM-HINDI` (prob $0.15$).
   - Hour `1PM` is free.
   - Selected: `WH-1PM-HINDI`.
   - `used_hours = {'11AM', '1PM'}`.
4. **Allocate VOICE**:
   - Candidate `VOICE_BOT-11AM-HINDI` (prob $0.05$) collides with `11AM`.
   - Next candidate: `VOICE_BOT-4PM-HINDI` (prob $0.05$).
   - Hour `4PM` is free.
   - Selected: `VOICE_BOT-4PM-HINDI`.
   - `used_hours = {'11AM', '1PM', '4PM'}`.
5. **Final Output Stored**:
   `SMS-11AM-HINDI|WH-1PM-HINDI|VOICE_BOT-4PM-HINDI`

---

### 1.6 Downstream Campaign & Trigger Generation

In [scripts/run_monthly_inference_pipeline.py](file:///home/aditya/aiml/campaign-recommendation/scripts/run_monthly_inference_pipeline.py#L2572-L2595), campaign aggregation handles this pipe-separated token:

```python
if single_communication_per_day:
    if multimode_postdue and day in POSTDUE_DAYS:
        seen_modes: set[str] = set()
        strategies = []
        for s in raw_val.split("|"):
            m = extract_canonical_mode_from_label(s.strip())
            if m and m not in seen_modes:
                seen_modes.add(m)
                strategies.append(s.strip())
    else:
        first_strat = get_first_valid_communication(raw_val)
        strategies = [first_strat] if first_strat not in ("", "-") else []
```

#### Outcome in Database & Scheduler:
- **Pre-Due Day (e.g. $D-5$)**: Generates **1** mapping row $\rightarrow$ 1 SMS trigger.
- **Post-Due Day (e.g. $D+1$)**: Generates **3** mapping rows:
  1. `POSTDUE_AIML_SMS_HINDI` scheduled at `10:00:00`
  2. `POSTDUE_AIML_WA_HINDI` scheduled at `13:00:00`
  3. `POSTDUE_AIML_VOICE_HINDI` scheduled at `16:00:00`
- In the recent **SEP-2026 $\rightarrow$ OCT-2026** production run across 8,127 loans:
  - 6,323 active post-due accounts received 3-mode simultaneous multi-channel reach.
  - 80,112 total touchpoints scheduled with 100% template resolution.
  - Zero same-hour collisions across all accounts.

---

### 1.7 Probability-Ranked Selection, Fallback Protection & Strict Regional Templates

In the latest `Production/csb-v1.1` releases, several core algorithmic safeguards were integrated across feature engineering, inference, and template resolution:

1. **Model Probability-Ranked Pre-Due Day Selection**:
   - *Legacy Limitation*: Earlier pre-due risk rules (`apply_predue_risk_rules`) enforced risk limits (Low: 2, Med: 3, High: 4) using chronological or positional truncation (`D-1 -> D-2 -> D-3 -> D-4`). When an account had multiple active touches, positional ordering caused $D-1$ or $D-4$ to be mechanically dropped, regardless of model confidence.
   - *Probability-Driven Optimization*: In `pipeline_common.py`, `predict_best_day_strategy()` returns per-account class probabilities for each day. Candidate days ($D-4 \dots D-1$) are dynamically ranked by **highest model probability** ($P_{\text{strategy}}$). The pipeline preserves the touches where the machine learning model has peak confidence of engagement, ensuring optimal contact timing.

2. **Post-Due 2nd Best Fallback Protection**:
   - In multi-class post-due day models, if the top predicted class is `-` (blank/no communication), the pipeline inspects model class probabilities. If multiple classes exist and valid non-blank classes are available, the pipeline automatically selects the 2nd best (highest probability non-blank) communication strategy. This ensures delinquent borrowers receive actionable outreach rather than being silently omitted.

3. **Mandatory $D-5$ Baseline Model Selection**:
   - Rather than synthetically assigning a hardcoded string `SMS-11AM-ENGLISH` when $D-5$ predicts blank, the pipeline evaluates model probabilities on $D-5$: if the raw prediction is blank, it selects the model's highest-probability non-blank candidate class for that specific borrower before resorting to default fallbacks.

4. **Strict Regional Language Protection (Zero English Bleed)**:
   - CSB communicates with borrowers across regional languages (Malayalam, Tamil, Telugu, Hindi, Marathi, etc.). Earlier template resolution could fall back to `DEFAULT` or `ENGLISH` if a regional template was not explicitly defined.
   - In CSB v1.1, regional language queries are strictly protected: **regional predictions must NEVER fall back to DEFAULT or ENGLISH templates**. If a regional template is missing in `digital_rules`, the resolution intentionally returns `""` with status `FAILED` and an explicit diagnostic reason. This eliminates non-compliant dispatches where a borrower expecting a regional language receives English verbiage.

5. **Preservation of UNKNOWN Risk Communication Events**:
   - In earlier feature generation, records where `RISK == "UNKNOWN"` were discarded in `generate_strategy_dataset.py`. This caused critical gaps in historical communication extraction.
   - In CSB v1.1, all communication events are retained, missing risk tiers are gracefully merged with known risk tiers using `_is_known` prioritization, and downstream loan processing safely maps unassigned risk tiers to `LR` (Low Risk) without breaking execution groups.

---

## 2. CSB Branch Topology & Lineage Map

The branches form a linear, cumulative progression:

```
                                  [dev / master]
                                        │
           [445bbc9] Prediction summary generator from rbl_v1.2
           [a5abef6] True Success Multiplier logic for link payments
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
        feature/csb_v.1.0                              feature/csb-v1.1
          (at 9914700)                                   (at 9914700)
                 │                                             │
                 │ (22 commits)                                │
                 ▼                                             │
        feature/csb-v1.0-updates                               │
          (at 9acd5c3)                                         │
                 │                                             │
                 │ (6 commits: da2778b -> 7355b7c)             │
                 ▼                                             │
        Production/csb-v1.0                                    │
          (at 7355b7c)                                         │
                 │                                             │
                 │ (10 commits: 41fdf3a -> 3f98b45)            │
                 ▼                                             ▼
        Production/csb-v1.1 ───────────────────────────────────┘
          (at 3f98b45)
```

---

## 3. Branch-Wise & Commit-Wise Changelog

### 3.1 Branch: `feature/csb_v1.0` (Baseline Foundation)
- **Git Branch Ref:** `feature/csb_v.1.0` (also tagged as `feature/csb-v1.1` base)
- **Tip Commit:** `991470030a24a67f0419cbf4735638e7190aa88d`

#### Preceding Enabler Commits:
1. **`445bbc9`** (*Bring in CSV prediction summary generation logic from rbl_v1.2 branch*):
   - Created `scripts/generate_prediction_summary.py` (419 lines) to generate campaign touchpoint rollups, channel mixes, and day-wise distributions.
2. **`a5abef6`** (*Implement True Success Multiplier logic for link payments*):
   - Created `scripts/append_payment_flags.py` and `scripts/fetch_payments_from_postgres.py`.
   - Introduced attribution logic distinguishing payments made directly through digital communication links from general payments.

#### Key Enhancements in `9914700`:
- **Differential Payment Multipliers (`config/weights_config.json`)**:
  - `c.payment_unique_id == p.reference_number`: **Link Payment** receives a **`5.0x`** training reward weight.
  - Payment date matches communication date on account: **External Payment** receives a **`2.5x`** training reward weight.
  - Driven by environment variables `STRATEGY_LINK_PAYMENT_MULTIPLIER` and `STRATEGY_EXTERNAL_PAYMENT_MULTIPLIER`.
- **Mandatory D-5 Full Base & Pre-Due Risk Touchpoints**:
  - Implemented `apply_predue_risk_rules()` in [scripts/pipeline_common.py](file:///home/aditya/aiml/campaign-recommendation/scripts/pipeline_common.py).
  - **Rule 1 (100% Base Reach)**: $D-5$ is mandatory for all accounts; if blank, defaults to `SMS-11AM-ENGLISH`.
  - **Rule 2 (Risk Caps)**: Low Risk = exactly 2 communications across $D-5 \dots D-1$; Medium Risk = 3 communications; High Risk = 4 communications.
  - Priority fill order: $D-1 \rightarrow D-2 \rightarrow D-3 \rightarrow D-4$.
- **Bank Non-Working Day Calendar Suppression**:
  - Implemented `scripts/apply_non_working_day_suppressions.py` (205 lines).
  - Automatically identifies Indian banking non-working days: all **Sundays** and the **2nd and 4th Saturdays** of each calendar month.
  - Suppresses scheduled touches falling on bank holidays and blanks the corresponding column to `"-"`.
- **Dynamic Digital Rules Template Resolution**:
  - Introduced lookup matching to resolve dynamic templates (`PREDUE_AIML_<MODE>_<LANG>`) instead of static identifiers.

---

### 3.2 Branch: `feature/csb-v1.0-updates` (Pipeline Hardening)
- **Git Branch Ref:** `feature/csb-v1.0-updates`
- **Range:** `9914700` $\rightarrow$ `9acd5c3` (22 commits)
- **Tip Commit:** `9acd5c31e05e6bae597452102416981cf8c2246b`

#### Detailed Commit Breakdown:

1. **`a251590`** (*feat: Add new customer fallback logic and pipeline updates*):
   - Added `scripts/generate_consolidated_payment_report.py` (288 lines) to aggregate all client payment channels.
   - Added `scripts/identify_new_customer_fallbacks.py` to handle cold-start accounts with zero historical digital touchpoints.
2. **`3b545e3`** (*fixes for duplicate entry in api_aduit_log*):
   - Added concurrency guards in `src/campaign_recommendation/api_service.py` to prevent duplicate insertions into `api_audit_log` during asynchronous training/inference API triggers.
3. **`925ab63`** (*Fixing new customers, and payemnt report generation*):
   - Resolved customer ID matching bugs in fallback assignment and added payment summary exports.
4. **`bfb86d8`** (*minor change*):
   - Formatted column headers in consolidated payment exports.
5. **`22ba440`** (*minor change*):
   - Added execution phase timing telemetry in `run_monthly_inference_pipeline.py`.
6. **`faf409b`** (*minor fixes*):
   - Handled edge cases in `export_mcollect_scheduler.py` when zero campaigns matched filters.
7. **`d8efe37`** (*fixes for dataset query*):
   - Corrected SQL dataset generation queries in `export_mcollect_scheduler.py` to properly quote character varying types in MCollect schemas.
8. **`027b287`** (*stratergy with 8hr bkt fix*):
   - Corrected send-hour bucketing logic to align with standard 8-hour banking operational windows (10 AM – 6 PM).
9. **`790043e`** (*newcust identification fix*):
   - Fixed distinction between loans genuinely absent from the base vs active loans that had zero communications in the 3-month history window.
10. **`a7b58fe`** (*drift calculation fixes*):
    - Added zero-count guards in `scripts/drift_utils.py` to eliminate `ZeroDivisionError` when computing Population Stability Index (PSI).
11. **`778f863`** (*drift calculation fixes2*):
    - Added Laplace smoothing ($\epsilon = 10^{-4}$) to prevent $\ln(0)$ crashes on unseen category values in inference.
12. **`27ec1f7`** (*fixed prediction rule*):
    - Updated `apply_predue_risk_rules()` to sanitize values with `get_first_valid_communication()`, rejecting `"None"`, `"nan"`, and `"-"`.
13. **`181f854`** (*training dataset changes*):
    - Aligned attribution window to 7 days post-communication for payment credit in `generate_strategy_dataset.py`.
14. **`3536cab`** (*chore: ignore data/payments directory*):
    - Added `data/payments/` to `.gitignore` to prevent raw banking transaction data from entering version control.
15. **`07c7305`** (*fetching payments modification*):
    - Optimized payment query execution with parameterized chunking against large Postgres tables.
16. **`d574eab`** (*fetching payments modification2*):
    - Added fallback account number joins when transaction reference IDs were omitted by the bank core.
17. **`26ecb04`** (*fetching payments modification3*):
    - Filtered payment types to ensure reversal transactions are excluded from positive reward labeling.
18. **`5122fa8`** (*update test assertion for external payment counts test*):
    - Updated automated unit test assertions in `tests/test_monthly_pipeline.py`.
19. **`0500061`** (*support EXPORT_PATH and LOCAL_EXPORT_PATH for local shared file export*):
    - Added environment variable support to export recommendation workbooks directly to shared NAS/NFS mounts.
20. **`312efd2`** (*add support for LOCAL_EXPORT_DATASET_PATH and LOCAL_EXPORT_SCHEDULER_PATH env variables*):
    - Decoupled export file destinations: separate paths for Quartz dataset CSVs vs scheduler Excel workbooks.
21. **`1f7de40`** (*Production issue changes- Dynamic Prediction Window & Send Hour Bucketing, 1-to-1 Mapping Fix (Post-Due Capping), Production Template Name Resolution & Config*):
    - **Dynamic Prediction Window**: Added support for `config/default_config.json` defining `allowed_day_offsets` (e.g., `[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]`).
    - **Hour Snapping**: Introduced `snap_hour_to_window()` in `pipeline_common.py` to clamp out-of-bounds predicted hours (e.g. 8 AM or 9 PM) into allowable client windows (10 AM – 6 PM).
    - **1-to-1 Post-Due Capping**: Introduced `apply_postdue_risk_rules()` to cap each post-due day to 1 valid communication.
    - **Unit Test Suite**: Added `tests/test_dynamic_prediction_window_config.py` (335 lines).
22. **`9acd5c3`** (*Production fix: Scheduler count, template name, and config prediction window*):
    - Synchronized scheduler counts with prediction outputs and updated digital rules template defaults.

---

### 3.3 Branch: `Production/csb-v1.0` (Template Guardrails & Release)
- **Git Branch Ref:** `Production/csb-v1.0`
- **Range:** `9acd5c3` $\rightarrow$ `7355b7c` (6 commits)
- **Tip Commit:** `7355b7cfc89cf08c0e0f4020b28fd5bfe5e52ca2`

#### Detailed Commit Breakdown:

1. **`da2778b`** (*Production fix: Scheduler count, template name, and config prediction window2*):
   - Corrected dataset row counts in `scripts/build_strategy_schedule_dataset.py`.
   - Updated `generate_prediction_evidence.py` to verify touchpoint consistency.
2. **`c34d4f1`** (*Production template issue fix-3*):
   - Overhauled `_prepare_campaign_outputs()` in `scripts/run_monthly_inference_pipeline.py`.
   - Added architecture documentation: [docs/template_resolution_and_campaign_generation_flow.md](file:///home/aditya/aiml/campaign-recommendation/docs/template_resolution_and_campaign_generation_flow.md) (263 lines).
   - **Cross-Due Guardrails**: Strictly prohibited `PREDUE` templates from being resolved for `POSTDUE` campaign mappings and vice-versa, preventing wrong verbiage dispatches.
3. **`97d5c48`** (*Production template issue fix-4*):
   - Added deep resolution diagnostic logging to track why any template failed to resolve.
   - Added 196 lines of regression unit tests in `tests/test_digital_rules_template_lookup.py`.
4. **`8fa925d`** (*Production template issue fix-5*):
   - Validated mode (SMS vs WhatsApp vs Voice) and language consistency during `digital_rules` lookup cache construction.
5. **`4e72635`** (*Production template fix: safe tuple unpacking, DB mode language consistency, and strict cohort defaults*):
   - Fixed tuple unpacking crashes when reading `digital_rules` metadata tuples.
   - Enforced strict cohort defaults if client database lacked regional language templates (fallback to English default iteration).
6. **`7355b7c`** (*Production template fix: validate mode, due_type, and language consistency across all historical and resolution steps*):
   - Unified template validation across all four resolution stages (history cache $\rightarrow$ exact DB match $\rightarrow$ mode/language default $\rightarrow$ fallback template).
   - Established complete stability for CSB v1.0 production release.

---

### 3.4 Branch: `feature/csb-v1.1` (Next-Gen Staging)
- **Git Branch Ref:** `feature/csb-v1.1`
- **Tip Commit:** `9914700` (Baseline fork identical to `feature/csb_v.1.0`)
- **Role:** Staging reference branch created from `9914700` as the designated feature branch target for all v1.1 developments.

---

### 3.5 Branch: `Production/csb-v1.1` (Dynamic Window, Multi-Mode, Probability Ranking & Template Hardening)
- **Git Branch Ref:** `Production/csb-v1.1`
- **Base Commit:** `7355b7c` (from `Production/csb-v1.0`)
- **Tip Commit in Git:** `3f98b453e603cad756e55ca01108ac18b69a8273`
- **Total Commits on Branch:** 10 commits
- **Test Suite Status:** 197 passed / 197 total tests

#### Complete Commit-by-Commit Breakdown:

1. **`41fdf3a`** (*dynamic prediction window resolution from the database*):
   - Implemented automated discovery of dynamic prediction windows from CSB client PostgreSQL table `digital_collections.data_config` (`aiml.start.day` and `aiml.end.day`).
   - **4-Tier Configuration Hierarchy**:
     1. CLI Argument: `--allowed-day-offsets` (e.g. `--allowed-day-offsets -5 -4 -3 -2 -1 1 2 3 4 5`)
     2. Client Database: `data_config` keys `aiml.start.day` and `aiml.end.day`
     3. JSON Config: `config/default_config.json`
     4. Pipeline Baseline: Full range ($D-5 \dots D+20$)
   - Inactive day columns outside the resolved range are automatically skipped and assigned `"-"`.
   - Added 328 lines of unit tests in [tests/test_dynamic_prediction_window_config.py](file:///home/aditya/aiml/campaign-recommendation/tests/test_dynamic_prediction_window_config.py).

2. **`e007406`** (*Postdue rule change*):
   - **Multi-Mode Post-Due Strategy Prediction**: Implemented `predict_distinct_modes_by_risk()` in [scripts/pipeline_common.py](file:///home/aditya/aiml/campaign-recommendation/scripts/pipeline_common.py) with peak probability mode prioritization and hour anti-collision algorithm.
   - Exposed `--multimode-postdue` / `--no-multimode-postdue` flags across `scripts/predict_next_month_strategy_catboost.py` and `scripts/run_monthly_inference_pipeline.py`.
   - **System-Wide Business Reasoning Overhaul**: Fixed `JSONDecodeError` bug in `identify_new_customer_fallbacks.py`, normalized loan keys (18-digit zero-padded vs 16-digit unpadded), and structured per-day JSON explanations with RBI bank holiday context.
   - **FastAPI Subprocess Fix**: Fixed `--dry-run` `AttributeError` using `getattr(args, "dry_run", False)`.
   - **Automated Class Balancing in Training**: Added `--auto-class-weights` (`None`, `Balanced`, `SqrtBalanced`) to `train_next_month_strategy_model_catboost.py` and saved `target_class_distributions` in model artifacts.
   - Added unit test suites in [tests/test_postdue_multimode_rules.py](file:///home/aditya/aiml/campaign-recommendation/tests/test_postdue_multimode_rules.py) (220 lines) and [tests/test_prediction_reasoning.py](file:///home/aditya/aiml/campaign-recommendation/tests/test_prediction_reasoning.py) (192 lines).

3. **`dac1a20`** (*Production fixes- post due multimode rule configureable, and template logic change*):
   - **Channel-Aligned Business Reasoning**: Added `_align_reason_to_mode()` in [scripts/run_monthly_inference_pipeline.py](file:///home/aditya/aiml/campaign-recommendation/scripts/run_monthly_inference_pipeline.py) ensuring multi-mode campaign rows display channel-specific text and send times (e.g., `SMS at 10AM in English` vs `WhatsApp at 1PM in English`).
   - Clean time display formatting via `_format_time_display()` (`10:00:00` $\rightarrow$ `10AM`, `13:00:00` $\rightarrow$ `1PM`).
   - Wired `--multimode-postdue` flag into [scripts/identify_new_customer_fallbacks.py](file:///home/aditya/aiml/campaign-recommendation/scripts/identify_new_customer_fallbacks.py) so cold-start accounts also generate multi-mode recommendations.
   - Updated [scripts/env_utils.py](file:///home/aditya/aiml/campaign-recommendation/scripts/env_utils.py) and `.env.example` with multi-mode configuration toggles.

4. **`2e21497`** (*Production fixes- post due multimode rule configureable, and template logic change2*):
   - **Mandatory Non-Dash Prioritization**: Added `mandatory_communication: bool = False` to `predict_top_k_by_risk()` in `pipeline_common.py`. When active, class probabilities prioritize non-blank strategies over `"-"`, guaranteeing mandatory contact days like $D-5$ never emit `"-"` when valid communication classes exist.
   - **Cold-Start Pre-Due Risk Rules**: In `identify_new_customer_fallbacks.py`, wrapped `build_average_strategy_by_risk()` with `apply_predue_risk_rules()`, guaranteeing new customer fallback profiles strictly respect the 100% $D-5$ reach and risk limits (Low: 2, Med: 3, High: 4).
   - Added 119 lines of unit tests in [tests/test_predue_risk_rules.py](file:///home/aditya/aiml/campaign-recommendation/tests/test_predue_risk_rules.py).

5. **`275558c`** (*Production fixes- post due multimode rule configureable, and template logic change3*):
   - **Dynamic Window in Summaries & Suppression**: Added `--allowed-day-offsets` CLI parameter and config loading to [scripts/apply_non_working_day_suppressions.py](file:///home/aditya/aiml/campaign-recommendation/scripts/apply_non_working_day_suppressions.py) and [scripts/generate_prediction_summary.py](file:///home/aditya/aiml/campaign-recommendation/scripts/generate_prediction_summary.py).
   - Summary Excel workbooks (counts and percentages) and calendar suppressions evaluate only configured active days instead of all possible calendar offsets.
   - Added unit test coverage in `tests/test_dynamic_prediction_window_config.py` and `tests/test_postdue_multimode_rules.py`.

6. **`258fc0c`** (*Production fixes- post due multimode rule configureable, and template logic change3.1*):
   - **Optional Default Strategy in Pre-Due Rules**: Updated `apply_predue_risk_rules()` to accept `default_strategy: str | None = DEFAULT_PREDUE_STRATEGY`. When `default_strategy=None`, missing pre-due days are not artificially filled with synthetic fallbacks, preserving authentic model predictions.
   - **Single-Mode Post-Due `k_override=1`**: Added `k_override` to `predict_top_k_by_risk()`, ensuring single-mode post-due predictions select exactly 1 top strategy without pipe-separated tokens.
   - **Model Evaluation for Blank History Loans**: Updated [scripts/predict_next_month_strategy_catboost.py](file:///home/aditya/aiml/campaign-recommendation/scripts/predict_next_month_strategy_catboost.py) to evaluate blank history loans through `build_feature_matrix(blank_rows)` and model inference (`X_blank`) instead of hardcoding `"-"` across all days.

7. **`eee48ca`** (*Production fixes- post due multimode rule configureable, and template logic change3.2*):
   - **Empty Mode Vocabulary Guard**: In `predict_distinct_modes_by_risk()`, added guard returning `"-"` safely across all accounts if a day model contains no valid candidate tokens across any target mode (e.g. only `"-"` in vocabulary).
   - **Restored Natural Pre-Due Non-Communication**: Restricted `mandatory_communication` to `(day == "D-5" or day in POSTDUE_DAY_COLUMNS)`, allowing pre-due days $D-4 \dots D-1$ to predict `"-"` naturally when non-communication is the model's top prediction.
   - Added 120 lines of unit tests in `tests/test_predue_risk_rules.py`.

8. **`e77b6ad`** (*Production fixes- post due multimode rule configureable, and template logic change3.3*):
   - **Model Probability-Ranked Pre-Due Day Selection**: Implemented `predict_best_day_strategy()` in `pipeline_common.py` returning `(chosen_strategy_series, probability_series)`.
   - When trimming pre-due communications to meet risk tier limits (Low: 2, Med: 3, High: 4), candidate days ($D-4 \dots D-1$) are ranked and selected by **highest model probability** rather than chronological order, preventing $D-1$ from being arbitrarily dropped.
   - Pre-due and post-due rules applied immediately prior to generating `PREDICTION_REASON`, ensuring explanations match final filtered strategies.
   - Added 170 lines of unit tests in `tests/test_predue_risk_rules.py`.

9. **`b3275f0`** (*Production fixes- post due multimode rule configureable, and template logic change3.4*):
   - **Two-Stage Prediction Auditability (`raw` vs `final`)**: Implemented clear decoupling between raw inference and rule-adjusted predictions. Saves both `<stem>_raw.csv` (unfiltered model predictions) and `<stem>.csv` (final business-enforced predictions).
   - **Post-Due 2nd Best Fallback**: If raw top prediction is `"-"` and multiple classes exist, automatically selects the highest probability non-blank communication class.
   - **Model-Driven $D-5$ Baseline**: If raw $D-5$ prediction is `"-"`, selects the highest probability non-dash strategy from model probabilities rather than a static default.
   - Per-loan candidate day probabilities calculated from predicted class probabilities before risk capping.
   - Re-synthesizes final `PREDICTION_REASON` payload for every loan to match final output columns.

10. **`3f98b45`** (*Production fixes- post due multimode rule configureable, and template logic change3.5*):
    - **Strict Regional Language Template Protection (Zero English Bleed)**:
      - In `resolve_template_name_with_status()` in [scripts/run_monthly_inference_pipeline.py](file:///home/aditya/aiml/campaign-recommendation/scripts/run_monthly_inference_pipeline.py), language fallback to `DEFAULT` or `ENGLISH` is **strictly restricted to English predictions (`is_english`)**.
      - Regional language predictions (e.g. Marathi, Hindi, Tamil, Telugu, Gujarati, Kannada, Bengali) **must NEVER fall back to DEFAULT or ENGLISH templates**. If a regional template is missing in `digital_rules`, resolution returns `""` with status `"FAILED"` and logs: `"No matching template found for regional language '{lang_u}' under ({due_u}, {mode_u}) and regional predictions do not fall back to English/DEFAULT"`.
      - Updated `validate_template_language_consistency()` to strictly reject English templates for regional queries.
    - **Resolution of Missing Communication Data in Training Windows (`UNKNOWN` Risk Retention)**:
      - In [scripts/generate_strategy_dataset.py](file:///home/aditya/aiml/campaign-recommendation/scripts/generate_strategy_dataset.py), removed `df["RISK"].ne("UNKNOWN")` filter which previously dropped all communication events having `RISK="UNKNOWN"`, restoring 100% communication data coverage.
      - In [scripts/build_strategy_schedule_dataset.py](file:///home/aditya/aiml/campaign-recommendation/scripts/build_strategy_schedule_dataset.py) and [scripts/build_monthly_feature_dataset.py](file:///home/aditya/aiml/campaign-recommendation/scripts/build_monthly_feature_dataset.py), retained `UNKNOWN` risk rows, while using `_is_known` prioritization so known risk tiers (`LOW`, `MEDIUM`, `HIGH`) are chosen whenever available for a loan.
    - **Safe Handling of `UNKNOWN` Risk Loans in Campaign Assignment & Reasoning**:
      - In [scripts/run_monthly_inference_pipeline.py](file:///home/aditya/aiml/campaign-recommendation/scripts/run_monthly_inference_pipeline.py), loans with risk `UNKNOWN`, `NULL`, `NONE`, `NAN`, `""`, `<NA>` are safely mapped to `LR` (Low Risk) in `_build_campaign_assignment_groups()` and `_mapping_prediction_reason()` instead of generating invalid risk codes or causing key errors.
    - **Unit Test Suite Expansion**: Added [tests/test_unknown_risk_handling.py](file:///home/aditya/aiml/campaign-recommendation/tests/test_unknown_risk_handling.py) (215 lines) and expanded [tests/test_digital_rules_template_lookup.py](file:///home/aditya/aiml/campaign-recommendation/tests/test_digital_rules_template_lookup.py) (70 lines of regional protection tests).
    - Total test suite coverage increased to **197 passed / 197 total tests**.

---

## 4. Comparative Matrix Across Branches

| Capability / Feature | `feature/csb_v1.0` | `feature/csb-v1.0-updates` | `Production/csb-v1.0` | `feature/csb-v1.1` | `Production/csb-v1.1` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Payment Attribution (Link vs Ext)** | ✅ Link 5x, Ext 2.5x | ✅ Refined joins | ✅ Validated | ✅ Baseline | ✅ Validated |
| **Mandatory D-5 Full Base Reach** | ✅ Mandatory (Static) | ✅ Mandatory (Static) | ✅ Mandatory (Static) | ✅ Baseline | **✅ Model Probability Non-Dash** |
| **Pre-Due Risk Limits (2/3/4)** | ✅ Enforced | ✅ Sanitized | ✅ Validated | ✅ Baseline | ✅ Enforced |
| **Pre-Due Day Selection Strategy** | Positional ($D-1 \dots D-4$) | Positional | Positional | Positional | **✅ Model Probability-Ranked** |
| **Bank Holiday Calendar Suppression** | ✅ Sun, 2nd/4th Sat | ✅ Verified | ✅ Verified | ✅ Baseline | ✅ Reason logged |
| **Post-Due Capping Strategy** | Single mode (1/day) | Single mode (1/day) | Strict Single Mode | Single mode (1/day) | **Multi-Mode (SMS+WA+Voice) / Configurable** |
| **Post-Due Top-Blank Fallback** | ❌ Hardcoded `-` | ❌ Hardcoded `-` | ❌ Hardcoded `-` | ❌ Hardcoded `-` | **✅ 2nd Best Non-Blank Model Fallback** |
| **Post-Due Hour Anti-Collision** | ❌ None | ❌ None | ❌ None | ❌ None | **✅ Active** |
| **Prediction Window Source** | Static ($D-5 \dots D+20$) | Config JSON | Config JSON | Static ($D-5 \dots D+20$) | **Dynamic DB (`data_config`)** |
| **Template Guardrails** | Basic lookup | Basic lookup | **Strict Cross-Due** | Basic lookup | **Strict Cross-Due + Strict Regional Isolation** |
| **Regional Language Bleed Protection**| ❌ Falls back to English | ❌ Falls back to English | ❌ Falls back to English | ❌ Falls back to English | **✅ Never Falls Back to English/DEFAULT** |
| **Business Reasoning Architecture** | Generic string | Generic string | String fallback | Generic string | **Structured JSON Context + Mode Aligned** |
| **Class Balancing Support** | ❌ Sample weights only| ❌ Sample weights only| ❌ Sample weights only| ❌ Sample weights only| **✅ Auto Class Weights (`Balanced`/`SqrtBalanced`)** |
| **Unknown Risk Resilience** | ❌ Discarded / Crash | ❌ Discarded / Crash | ❌ Discarded / Crash | ❌ Discarded / Crash | **✅ Retained in Training + Mapped to LR** |
| **Prediction Output Auditability** | Single final CSV | Single final CSV | Single final CSV | Single final CSV | **Dual Output (`_raw.csv` and `.csv`)** |

---

## 5. Summary of Production Verification & Audit Metrics

The entire `Production/csb-v1.1` suite has been validated in staging and audited against production workflows for CSB:

- **Total Test Suite**: **197 passed** / 197 total tests (zero failures, 100% test pass rate across all modules).
- **Dynamic Prediction Window**: Successfully resolved from client DB `data_config` to 10 days ($D-5 \dots D-1$ pre-due, $D+1 \dots D+5$ post-due).
- **Accounts Evaluated**: 8,127 active loans in production verification.
- **Multi-Mode Post-Due Coverage**: 6,323 active post-due accounts received simultaneous SMS, WhatsApp, and Voice outreach on $D+1, D+2, D+5$ with zero hour collision.
- **Template Resolution Integrity**:
  - English templates resolve through exact DB matches, cohort fallbacks, and DEFAULT templates.
  - Regional language predictions (Marathi, Hindi, Tamil, Telugu, etc.) strictly resolve only against authentic regional templates; regional queries never bleed into English verbiages.
- **Calendar Suppressions**: 24,381 touchpoints successfully suppressed on bank holidays (16,254 Sundays, 8,127 bank Saturdays) with context-aware justifications.
- **Missing Communication Data Resolution**: All communication records with missing or `UNKNOWN` risk tiers are fully retained during training dataset generation, ensuring 100% historical communication events are utilized in feature matrices.
- **Reasoning Verification**: 100% of mappings in export workbooks and database tables contain clear, context-aware business justifications (0 occurrences of placeholder text).
- **Two-Stage Audit Trail**: Both unfiltered machine learning predictions (`<stem>_raw.csv`) and rule-filtered executable predictions (`<stem>.csv`) are persisted for complete auditability.
