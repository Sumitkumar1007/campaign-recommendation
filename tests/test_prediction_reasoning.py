from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_common import parse_strategy_parts, readable_strategy
from identify_new_customer_fallbacks import build_fallback_rows, apply_fallbacks_to_prediction_file
from run_monthly_inference_pipeline import (
    _load_prediction_context_lookup,
    _mapping_prediction_reason,
    build_campaign_mappings,
)
from apply_non_working_day_suppressions import suppress_non_working_days_in_dataframe
from export_recommendation_workbooks import build_scheduler_rows


def test_readable_strategy_single_and_multimode():
    assert readable_strategy("SMS-11AM-ENGLISH") == "SMS at 11AM in English"
    assert readable_strategy("WH-2PM-REGIONAL") == "WhatsApp at 2PM in Regional language"
    assert readable_strategy("VOICE-4PM-ENGLISH") == "voice call at 4PM in English"
    assert readable_strategy("VOICE_BOT-10AM-HINDI") == "voice bot at 10AM in Hindi"
    assert readable_strategy("-") == "no communication"
    assert readable_strategy("") == "no communication"

    multimode = "SMS-11AM-ENGLISH|WH-12PM-ENGLISH|VOICE-4PM-ENGLISH"
    expected = "SMS at 11AM in English, WhatsApp at 12PM in English, voice call at 4PM in English"
    assert readable_strategy(multimode) == expected


def test_identify_new_customer_fallbacks_writes_json_reasons(tmp_path: Path):
    new_cases = pd.DataFrame([
        {
            "apacCardNumber": "009700354078721801",
            "originalRisk": "HIGH",
            "vertical": "LAP",
        }
    ])
    fallback_by_risk = {
        "HIGH": {
            "D-5": "WH-11AM-ENGLISH",
            "D-4": "WH-4PM-ENGLISH",
            "D-1": "WH-3PM-ENGLISH",
            "D+1": "SMS-5PM-ENGLISH",
            "D+2": "SMS-12PM-ENGLISH",
        }
    }
    fallback_rows = build_fallback_rows(new_cases, fallback_by_risk, allowed_day_labels={"D-5", "D-4", "D-1", "D+1", "D+2"})
    assert not fallback_rows.empty

    d5_row = fallback_rows[fallback_rows["day"] == "D-5"].iloc[0]
    assert "Cold-start fallback" in d5_row["reason"]
    assert "WhatsApp at 11AM in English" in d5_row["reason"]

    # Test applying to prediction file
    pred_file = tmp_path / "predictions.csv"
    pd.DataFrame([
        {
            "Loan_number": "009700354078721801",
            "SOURCE_RISK": "HIGH",
            "SOURCE_VERTICAL": "LAP",
            "MONTH": "OCT-2026",
            "SOURCE_MONTH_USED": "SEP-2026",
            "D-5": "-",
            "D-4": "-",
            "D-1": "-",
            "D+1": "-",
            "D+2": "-",
            "PREDICTION_REASON": "",
        }
    ]).to_csv(pred_file, index=False)

    updated = apply_fallbacks_to_prediction_file(pred_file, fallback_rows)
    assert updated == 1

    updated_df = pd.read_csv(pred_file, dtype=str)
    raw_reason = updated_df.loc[0, "PREDICTION_REASON"]
    parsed_reason = json.loads(raw_reason)
    assert isinstance(parsed_reason, dict)
    assert "D-5" in parsed_reason
    assert "Cold-start fallback" in parsed_reason["D-5"]
    assert "WhatsApp at 11AM in English" in parsed_reason["D-5"]
    assert updated_df.loc[0, "IS_NEW_CUSTOMER"] == "True"


def test_mapping_prediction_reason_resolves_cold_start_fallback(tmp_path: Path):
    pred_file = tmp_path / "predictions.csv"
    reasons = {
        "D-5": "Cold-start fallback: WhatsApp at 11AM in English is recommended based on HIGH-risk historical average.",
    }
    pd.DataFrame([
        {
            "Loan_number": "9700354078721801",  # Stripped leading zeroes
            "SOURCE_RISK": "HIGH",
            "SOURCE_VERTICAL": "LAP",
            "MONTH": "OCT-2026",
            "SOURCE_MONTH_USED": "SEP-2026",
            "IS_NEW_CUSTOMER": "True",
            "D-5": "WH-11AM-ENGLISH",
            "PREDICTION_REASON": json.dumps(reasons),
        }
    ]).to_csv(pred_file, index=False)

    context_lookup = _load_prediction_context_lookup(pred_file, "OCT-2026")
    # Verified normalized loan lookup with zero padding
    assert "009700354078721801" in context_lookup
    assert "9700354078721801" in context_lookup

    mapping_row = pd.Series({
        "loan_number": "009700354078721801",
        "date": "D-5",
        "mode": "WHATSAPP",
        "time": "11AM",
        "language": "ENGLISH",
        "risk": "HIGH",
        "due_type": "PREDUE",
    })
    reason = _mapping_prediction_reason(mapping_row, context_lookup)
    assert "Cold-start fallback" in reason
    assert "WhatsApp at 11AM in English" in reason


def test_mapping_prediction_reason_fallback_when_day_not_in_reasons():
    context_lookup = {
        "12345": {
            "reasons": {},
            "is_new_customer": True,
            "risk": "MEDIUM",
        }
    }
    mapping_row = pd.Series({
        "loan_number": "12345",
        "date": "D-2",
        "mode": "SMS",
        "time": "12PM",
        "language": "ENGLISH",
        "risk": "MEDIUM",
        "due_type": "PREDUE",
    })
    reason = _mapping_prediction_reason(mapping_row, context_lookup)
    assert "Cold-start fallback" in reason
    assert "SMS" in reason
    assert "MEDIUM" in reason
    assert "Reason not available" not in reason


def test_suppress_non_working_days_updates_prediction_reason():
    df = pd.DataFrame([
        {
            "EMI_DATE": "07/10/2026",  # 07-Oct-2026 is Wednesday
            # 07-Oct - 3 days = 04-Oct-2026 which is Sunday
            "D-3": "SMS-11AM-ENGLISH",
            "PREDICTION_REASON": json.dumps({"D-3": "SMS at 11AM in English is recommended..."}),
        }
    ])
    suppressed_df, stats = suppress_non_working_days_in_dataframe(df, saturday_mode="bank_saturdays", suppress_sundays=True)
    assert stats["sunday_suppressions"] == 1
    assert suppressed_df.loc[0, "D-3"] == "Sunday"

    updated_reasons = json.loads(suppressed_df.loc[0, "PREDICTION_REASON"])
    assert "Communication suppressed on D-3" in updated_reasons["D-3"]
    assert "Sunday" in updated_reasons["D-3"]


def test_build_scheduler_rows_provides_informative_fallback_reason():
    campaigns = pd.DataFrame([
        {
            "name": "PREDUE_AIML_SMS_LAP_ENGLISH_HR_7TH_PRUTECH_191026",
            "mode": "SMS",
            "date": "D-5",
            "time": "11AM",
            "template_name": "PREDUE_SMS",
            "dataset_name": "PREDUE|SMS|LAP|ENGLISH|HR|7",
            "vendor": "prutech",
            "active": "T",
            "due_type": "PREDUE",
            "risk": "HIGH",
        }
    ])
    # Empty reason dictionary simulating unmapped campaign
    rows = build_scheduler_rows(campaigns, {})
    assert len(rows) == 1
    scheduler_reason = rows[0][8]
    assert "PREDUE SMS at 11AM campaign scheduled on D-5" in scheduler_reason
    assert "HIGH risk strategy cohort patterns" in scheduler_reason
    assert "Reason not available" not in scheduler_reason
