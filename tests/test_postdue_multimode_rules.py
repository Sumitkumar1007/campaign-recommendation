from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_common import (
    extract_canonical_mode_from_label,
    extract_hour_from_label,
    apply_postdue_risk_rules,
    predict_distinct_modes_by_risk,
)
from run_monthly_inference_pipeline import build_campaign_mappings, build_campaign_recommendations


class DummyEncoder:
    def __init__(self, classes: list[str]):
        self.classes_ = np.array(classes)


class DummyModel:
    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self._probs[np.newaxis, :], len(X), axis=0)


def test_extract_canonical_mode_from_label():
    assert extract_canonical_mode_from_label("SMS-11AM-ENGLISH") == "SMS"
    assert extract_canonical_mode_from_label("WH-12PM-ENGLISH") == "WHATSAPP"
    assert extract_canonical_mode_from_label("WHATSAPP-2PM-HINDI") == "WHATSAPP"
    assert extract_canonical_mode_from_label("VOICE-3PM-ENGLISH") == "VOICE"
    assert extract_canonical_mode_from_label("VOICE_BOT-4PM-TELUGU") == "VOICE"
    assert extract_canonical_mode_from_label("IVR-10AM-HINDI") == "VOICE"
    assert extract_canonical_mode_from_label("-") is None
    assert extract_canonical_mode_from_label("") is None


def test_extract_hour_from_label():
    assert extract_hour_from_label("SMS-11AM-ENGLISH") == "11AM"
    assert extract_hour_from_label("WH-12PM-HINDI") == "12PM"
    assert extract_hour_from_label("VOICE_BOT-4PM-REGIONAL") == "4PM"
    assert extract_hour_from_label("-") is None


def test_predict_distinct_modes_all_three_modes_and_repeated_language():
    # Classes contain multiple hours and languages for SMS, WhatsApp, and Voice
    classes = [
        "-",
        "SMS-10AM-HINDI",      # prob 0.30
        "SMS-11AM-HINDI",      # prob 0.20
        "WH-11AM-HINDI",       # prob 0.25 (collides with SMS if 11AM chosen)
        "WH-1PM-HINDI",        # prob 0.15
        "VOICE_BOT-11AM-HINDI",# prob 0.05 (collides)
        "VOICE_BOT-4PM-HINDI", # prob 0.05
    ]
    probs = np.array([0.00, 0.30, 0.20, 0.25, 0.15, 0.05, 0.05])
    model = DummyModel(probs)
    encoder = DummyEncoder(classes)
    X = pd.DataFrame({"f1": [1.0, 2.0]})

    res = predict_distinct_modes_by_risk(model, encoder, X)
    assert len(res) == 2

    # Check for row 0
    row_val = res.iloc[0]
    parts = row_val.split("|")
    assert len(parts) == 3

    modes = [extract_canonical_mode_from_label(p) for p in parts]
    assert modes == ["SMS", "WHATSAPP", "VOICE"]

    # Language is repeated (HINDI across all 3 modes)
    assert all("HINDI" in p for p in parts)

    # Check hours: SMS took 10AM (0.30), WH took 11AM (0.25), VOICE took 4PM (0.05)
    hours = [extract_hour_from_label(p) for p in parts]
    assert len(set(hours)) == 3  # All hours are non-colliding
    assert hours == ["10AM", "11AM", "4PM"]


def test_predict_distinct_modes_hour_anti_collision():
    # Both SMS and WH have their peak probability at 11AM
    classes = [
        "SMS-11AM-ENGLISH",    # prob 0.40 (higher peak than WH)
        "SMS-2PM-ENGLISH",     # prob 0.10
        "WH-11AM-ENGLISH",     # prob 0.35 (collides with SMS-11AM)
        "WH-1PM-ENGLISH",      # prob 0.10 (next best for WH)
        "VOICE-4PM-ENGLISH",   # prob 0.05
    ]
    probs = np.array([0.40, 0.10, 0.35, 0.10, 0.05])
    model = DummyModel(probs)
    encoder = DummyEncoder(classes)
    X = pd.DataFrame({"f1": [1.0]})

    res = predict_distinct_modes_by_risk(model, encoder, X)
    parts = res.iloc[0].split("|")
    assert len(parts) == 3

    # SMS gets 11AM (peak prob 0.40)
    assert parts[0] == "SMS-11AM-ENGLISH"
    # WH shifts to next best non-colliding hour 1PM (instead of colliding at 11AM)
    assert parts[1] == "WH-1PM-ENGLISH"
    # Voice gets 4PM
    assert parts[2] == "VOICE-4PM-ENGLISH"


def test_predict_distinct_modes_fallback_when_mode_missing():
    # Only SMS and Voice exist in encoder classes; WhatsApp is absent
    classes = [
        "SMS-10AM-ENGLISH",
        "VOICE-3PM-ENGLISH",
    ]
    probs = np.array([0.60, 0.40])
    model = DummyModel(probs)
    encoder = DummyEncoder(classes)
    X = pd.DataFrame({"f1": [1.0]})

    res = predict_distinct_modes_by_risk(model, encoder, X)
    parts = res.iloc[0].split("|")
    assert len(parts) == 3
    assert parts[0] == "SMS-10AM-ENGLISH"
    assert parts[1] == "WH-12PM-ENGLISH"  # Fallback for WhatsApp
    assert parts[2] == "VOICE-3PM-ENGLISH"


def test_apply_postdue_risk_rules_multimode():
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_01",
                # Duplicate SMS, plus WH and Voice
                "D+1": "SMS-10AM-ENGLISH|SMS-11AM-ENGLISH|WH-1PM-ENGLISH|VOICE-4PM-ENGLISH",
                # Only 1 mode
                "D+2": "SMS-11AM-ENGLISH",
            }
        ]
    )

    # When allow_multimode=True, preserves up to 1 per mode (drops 2nd SMS)
    res_multi = apply_postdue_risk_rules(df, allow_multimode=True)
    d1_parts = res_multi.iloc[0]["D+1"].split("|")
    assert len(d1_parts) == 3
    assert d1_parts == ["SMS-10AM-ENGLISH", "WH-1PM-ENGLISH", "VOICE-4PM-ENGLISH"]
    assert res_multi.iloc[0]["D+2"] == "SMS-11AM-ENGLISH"

    # When allow_multimode=False, caps to 1st strategy overall
    res_single = apply_postdue_risk_rules(df, allow_multimode=False)
    assert res_single.iloc[0]["D+1"] == "SMS-10AM-ENGLISH"
    assert res_single.iloc[0]["D+2"] == "SMS-11AM-ENGLISH"


def test_build_campaign_mappings_multimode_postdue(tmp_path: Path):
    prediction_file = tmp_path / "predictions_postdue_multimode.csv"
    pd.DataFrame(
        {
            "Loan_number": ["0781109469306"],
            "MONTH": ["MAY-2026"],
            "SOURCE_RISK": ["HIGH"],
            "VERTICAL": ["LAP"],
            "emi_date": ["05/05/2026"],
            # Pre-due D-5 has 2 strategies, should be capped to 1
            "D-5": ["SMS-11AM-ENGLISH|WH-12PM-ENGLISH"],
            # Post-due D+1 has 3 distinct modes, should expand to 3 mapping rows
            "D+1": ["SMS-10AM-ENGLISH|WH-1PM-ENGLISH|VOICE-4PM-ENGLISH"],
        }
    ).to_csv(prediction_file, index=False)

    mappings = build_campaign_mappings(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
        single_communication_per_day=True,
        multimode_postdue=True,
    )

    # D-5 pre-due must have exactly 1 mapping row (single-communication per day)
    d_minus_5 = mappings[mappings["date"] == "D-5"]
    assert len(d_minus_5) == 1
    assert d_minus_5.iloc[0]["mode"] == "SMS"

    # D+1 post-due must have exactly 3 mapping rows (1 SMS, 1 WHATSAPP, 1 VOICE)
    d_plus_1 = mappings[mappings["date"] == "D+1"]
    assert len(d_plus_1) == 3
    assert set(d_plus_1["mode"]) == {"SMS", "WHATSAPP", "VOICE"}
    assert set(d_plus_1["time"]) == {"10:00:00", "13:00:00", "16:00:00"}

    # Also verify campaign recommendations have the correct templates for all 3 modes
    campaigns = build_campaign_recommendations(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
        single_communication_per_day=True,
        multimode_postdue=True,
    )
    d_plus_1_c = campaigns[campaigns["date"] == "D+1"]
    assert len(d_plus_1_c) == 3
    assert set(d_plus_1_c["mode"]) == {"SMS", "WHATSAPP", "VOICE"}
    assert set(d_plus_1_c["template_name"]) == {
        "POSTDUE_AIML_SMS_ENGLISH",
        "POSTDUE_AIML_WA_ENGLISH",
        "POSTDUE_AIML_VOICE_ENGLISH",
    }
