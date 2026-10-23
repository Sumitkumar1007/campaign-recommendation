from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

from pipeline_common import apply_predue_risk_rules, predict_top_k_by_risk


def test_d5_full_base_mandatory():
    # Account with all blank pre-due days
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_BLANK_01",
                "SOURCE_RISK": "LOW",
                "D-5": "-",
                "D-4": "-",
                "D-3": "-",
                "D-2": "-",
                "D-1": "-",
            }
        ]
    )

    res = apply_predue_risk_rules(df)
    # D-5 must be populated for 100% base
    assert res.iloc[0]["D-5"] == "SMS-11AM-ENGLISH"
    # Low risk requires total 2 pre-due communications (D-5 + D-1 priority fill)
    assert res.iloc[0]["D-1"] == "SMS-11AM-ENGLISH"
    active_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[0][col] != "-"]
    assert len(active_predue) == 2


def test_low_risk_capped_at_2_communications():
    # Account with all 5 pre-due days predicted
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_LOW_01",
                "SOURCE_RISK": "LOW",
                "D-5": "SMS-11AM-ENGLISH",
                "D-4": "WH-4PM-ENGLISH",
                "D-3": "VOICE-2PM-ENGLISH",
                "D-2": "SMS-9AM-ENGLISH",
                "D-1": "SMS-5PM-ENGLISH",
            }
        ]
    )

    res = apply_predue_risk_rules(df)
    active_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[0][col] != "-"]
    assert len(active_predue) == 2
    assert "D-5" in active_predue


def test_medium_risk_3_to_4_communications():
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_MED_01",
                "SOURCE_RISK": "MEDIUM",
                "D-5": "-",  # Blank D-5
                "D-4": "-",
                "D-3": "-",
                "D-2": "-",
                "D-1": "-",
            }
        ]
    )

    res = apply_predue_risk_rules(df)
    active_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[0][col] != "-"]
    assert len(active_predue) == 3
    assert res.iloc[0]["D-5"] == "SMS-11AM-ENGLISH"


def test_high_risk_4_communications():
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_HIGH_01",
                "SOURCE_RISK": "HIGH",
                "D-5": "SMS-11AM-ENGLISH",
                "D-4": "WH-4PM-ENGLISH",
                "D-3": "VOICE-2PM-ENGLISH",
                "D-2": "SMS-9AM-ENGLISH",
                "D-1": "SMS-5PM-ENGLISH",
            }
        ]
    )

    res = apply_predue_risk_rules(df)
    active_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[0][col] != "-"]
    assert len(active_predue) == 4
    assert "D-5" in active_predue


def test_predict_top_k_by_risk_picks_second_best_when_top_is_dash_for_mandatory():
    class DummyModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.array(
                [
                    [0.70, 0.20, 0.10],  # Top is '-', 2nd best is WH-4PM-ENGLISH
                    [0.70, 0.20, 0.10],  # Top is '-', 2nd best is WH-4PM-ENGLISH
                    [0.70, 0.20, 0.10],  # Top is '-', 2nd best is WH-4PM-ENGLISH
                ]
            )

    class DummyEncoder:
        classes_ = np.array(["-", "WH-4PM-ENGLISH", "SMS-11AM-ENGLISH"])

    X = pd.DataFrame({"x": [1, 2, 3]})
    risks = pd.Series(["LOW", "MEDIUM", "HIGH"])

    # 1. Non-mandatory: LOW outputs '-', MEDIUM outputs '-|WH...', HIGH outputs '-|WH...|SMS...'
    non_mandatory_output = predict_top_k_by_risk(
        DummyModel(),
        DummyEncoder(),
        X,
        risks,
        mandatory_communication=False,
    )
    assert non_mandatory_output.iloc[0] == "-"

    # 2. Mandatory (e.g. D-5): picks 2nd best (WH-4PM-ENGLISH) instead of '-' for all risk tiers!
    mandatory_output = predict_top_k_by_risk(
        DummyModel(),
        DummyEncoder(),
        X,
        risks,
        mandatory_communication=True,
    )
    assert mandatory_output.iloc[0] == "WH-4PM-ENGLISH"  # LOW risk k=1 picks 2nd best, NOT '-'!
    assert mandatory_output.iloc[1].startswith("WH-4PM-ENGLISH|")  # MEDIUM risk has 2nd best as first option
    assert mandatory_output.iloc[2].startswith("WH-4PM-ENGLISH|")  # HIGH risk has 2nd best as first option


def test_new_customers_receive_predue_risk_rules(tmp_path: Path):
    from identify_new_customer_fallbacks import (
        build_average_strategy_by_risk,
        build_fallback_rows,
        apply_fallbacks_to_prediction_file,
    )

    # Historical schedule has all blank pre-due days
    schedule_csv = tmp_path / "schedule.csv"
    pd.DataFrame({
        "RISK": ["LOW", "MEDIUM", "HIGH"],
        "D-5": ["-", "-", "-"],
        "D-4": ["-", "-", "-"],
        "D-3": ["-", "-", "-"],
        "D-2": ["-", "-", "-"],
        "D-1": ["-", "-", "-"],
        "D": ["-", "-", "-"],
        "D+1": ["SMS-10AM-ENGLISH", "SMS-10AM-ENGLISH", "SMS-10AM-ENGLISH"],
    }).to_csv(schedule_csv, index=False)

    fallback_by_risk = build_average_strategy_by_risk(schedule_csv, apply_predue_rules=True)

    # 1. High risk new customers must have exactly 4 pre-due communications and mandatory D-5
    high_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if fallback_by_risk["HIGH"][col] != "-"]
    assert len(high_predue) == 4
    assert fallback_by_risk["HIGH"]["D-5"] == "SMS-11AM-ENGLISH"

    # 2. Medium risk new customers must have exactly 3 pre-due communications and mandatory D-5
    med_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if fallback_by_risk["MEDIUM"][col] != "-"]
    assert len(med_predue) == 3
    assert fallback_by_risk["MEDIUM"]["D-5"] == "SMS-11AM-ENGLISH"

    # 3. Low risk new customers must have exactly 2 pre-due communications and mandatory D-5
    low_predue = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if fallback_by_risk["LOW"][col] != "-"]
    assert len(low_predue) == 2
    assert fallback_by_risk["LOW"]["D-5"] == "SMS-11AM-ENGLISH"

    # 4. End-to-end new customer fallbacks written to prediction file
    pred_csv = tmp_path / "predictions.csv"
    pd.DataFrame({
        "Loan_number": ["ACC_NEW_HIGH", "ACC_NEW_LOW"],
        "SOURCE_RISK": ["HIGH", "LOW"],
        "SOURCE_VERTICAL": ["LAP", "LAP"],
        "MONTH": ["OCT-2026", "OCT-2026"],
        "SOURCE_MONTH_USED": ["SEP-2026", "SEP-2026"],
        "D-5": ["-", "-"],
        "D-4": ["-", "-"],
        "D-3": ["-", "-"],
        "D-2": ["-", "-"],
        "D-1": ["-", "-"],
        "D": ["-", "-"],
        "D+1": ["-", "-"],
        "IS_NEW_CUSTOMER": ["False", "False"],
    }).to_csv(pred_csv, index=False)

    new_cases = pd.DataFrame([
        {"apacCardNumber": "ACC_NEW_HIGH", "originalRisk": "HIGH", "vertical": "LAP"},
        {"apacCardNumber": "ACC_NEW_LOW", "originalRisk": "LOW", "vertical": "LAP"},
    ])
    fallback_rows = build_fallback_rows(new_cases, fallback_by_risk)
    apply_fallbacks_to_prediction_file(pred_csv, fallback_rows)

    updated_pred = pd.read_csv(pred_csv, dtype=str)
    high_row = updated_pred[updated_pred["Loan_number"] == "ACC_NEW_HIGH"].iloc[0]
    high_active = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if high_row[col] != "-"]
    assert len(high_active) == 4
    assert high_row["D-5"] != "-"

    low_row = updated_pred[updated_pred["Loan_number"] == "ACC_NEW_LOW"].iloc[0]
    low_active = [col for col in ["D-5", "D-4", "D-3", "D-2", "D-1"] if low_row[col] != "-"]
    assert len(low_active) == 2
    assert low_row["D-5"] != "-"


def test_three_apacs_old_and_new_distinction_and_postdue_single_class():
    """
    Validates:
    1. Pre-due: Only D-5 is mandatory (forces 2nd-best if top is '-').
       Other pre-due days (D-4..D-1) allow '-' so genuine feature distinctions are preserved.
    2. New customer gets D-5 from model, and pre-due risk rules fill missing touchpoints to meet risk quota.
    3. Post-due: Days with only 1 class ('-') predict '-'.
       Days with non-dash classes pick 2nd-best by risk quota instead of '-'.
    """
    class MockModel:
        def __init__(self, probs):
            self._probs = np.array(probs)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return self._probs

    class MockEncoder:
        def __init__(self, classes):
            self.classes_ = np.array(classes)

    # 3 accounts:
    # Row 0: NEW customer (LOW risk) -> no features, top is '-' on all days
    # Row 1: OLD customer 1 (LOW risk) -> features trigger D-2 as top
    # Row 2: OLD customer 2 (HIGH risk) -> features trigger D-4 as top
    X = pd.DataFrame({"feat": [0, 1, 2]})
    risks = pd.Series(["LOW", "LOW", "HIGH"])

    # 1. D-5: mandatory_communication=True for all
    enc_d5 = MockEncoder(["-", "SMS-2PM-ENGLISH", "WH-11AM-ENGLISH"])
    model_d5 = MockModel([
        [0.80, 0.15, 0.05],  # Row 0 (New): top is '-', picks 2nd best SMS-2PM-ENGLISH
        [0.80, 0.15, 0.05],  # Row 1 (Old 1): top is '-', picks 2nd best SMS-2PM-ENGLISH
        [0.80, 0.05, 0.15],  # Row 2 (Old 2): top is '-', picks 2nd best WH-11AM-ENGLISH
    ])
    d5_pred = predict_top_k_by_risk(model_d5, enc_d5, X, risks, mandatory_communication=True)
    assert d5_pred.iloc[0] == "SMS-2PM-ENGLISH"
    assert d5_pred.iloc[1] == "SMS-2PM-ENGLISH"
    assert d5_pred.iloc[2].startswith("WH-11AM-ENGLISH")

    # 2. D-4: mandatory_communication=False (reverted)
    enc_d4 = MockEncoder(["-", "SMS-4PM-ENGLISH"])
    model_d4 = MockModel([
        [0.99, 0.01],  # Row 0: top is '-', stays '-'
        [0.99, 0.01],  # Row 1: top is '-', stays '-'
        [0.10, 0.90],  # Row 2 (Old 2): features trigger SMS-4PM-ENGLISH!
    ])
    d4_pred = predict_top_k_by_risk(model_d4, enc_d4, X, risks, mandatory_communication=False)
    assert d4_pred.iloc[0] == "-"
    assert d4_pred.iloc[1] == "-"
    assert d4_pred.iloc[2].startswith("SMS-4PM-ENGLISH")

    # 3. D-2: mandatory_communication=False (reverted)
    enc_d2 = MockEncoder(["-", "WH-10AM-ENGLISH"])
    model_d2 = MockModel([
        [0.95, 0.05],  # Row 0: top is '-', stays '-'
        [0.20, 0.80],  # Row 1 (Old 1): features trigger WH-10AM-ENGLISH!
        [0.95, 0.05],  # Row 2
    ])
    d2_pred = predict_top_k_by_risk(model_d2, enc_d2, X, risks, mandatory_communication=False)
    # For Row 2, overwrite to '-' as it had no D-2 signal
    d2_pred.iloc[2] = "-"
    assert d2_pred.iloc[0] == "-"
    assert d2_pred.iloc[1] == "WH-10AM-ENGLISH"
    assert d2_pred.iloc[2] == "-"

    # Combine into DataFrame and apply pre-due risk rules
    pred_df = pd.DataFrame({
        "Loan_number": ["ACC_NEW", "ACC_OLD_1", "ACC_OLD_2"],
        "SOURCE_RISK": ["LOW", "LOW", "HIGH"],
        "D-5": d5_pred,
        "D-4": d4_pred,
        "D-3": ["-", "-", "-"],
        "D-2": d2_pred,
        "D-1": ["-", "-", "-"],
    })
    res = apply_predue_risk_rules(pred_df, predue_days=["D-5", "D-4", "D-3", "D-2", "D-1"], default_strategy=None)

    # Account NEW (LOW risk): has D-5 from model, and no hardcoded strategy injected on missing days!
    new_active = [c for c in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[0][c] != "-"]
    assert len(new_active) == 1
    assert res.iloc[0]["D-5"] == "SMS-2PM-ENGLISH"
    assert res.iloc[0]["D-1"] == "-"  # NOT filled with fake SMS-11AM-ENGLISH!

    # Account OLD 1 (LOW risk): has D-5 and feature-triggered D-2 (NOT identical to NEW!)
    old1_active = [c for c in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[1][c] != "-"]
    assert len(old1_active) == 2
    assert res.iloc[1]["D-5"] == "SMS-2PM-ENGLISH"
    assert res.iloc[1]["D-2"] == "WH-10AM-ENGLISH"
    assert res.iloc[1]["D-1"] == "-"

    # Account OLD 2 (HIGH risk): has D-5 and feature-triggered D-4
    old2_active = [c for c in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.iloc[2][c] != "-"]
    assert len(old2_active) == 2
    assert res.iloc[2]["D-5"] == "WH-11AM-ENGLISH"
    assert res.iloc[2]["D-4"] == "SMS-4PM-ENGLISH"

    # 4. Post-due: single class '-' vs multi-class with 2nd best by risk
    # D+3 has single class '-'
    enc_dplus3 = MockEncoder(["-"])
    model_dplus3 = MockModel([[1.0], [1.0], [1.0]])
    dplus3_pred = predict_top_k_by_risk(model_dplus3, enc_dplus3, X, risks, mandatory_communication=True)
    assert dplus3_pred.iloc[0] == "-"
    assert dplus3_pred.iloc[1] == "-"
    assert dplus3_pred.iloc[2] == "-"

    # D+1 has classes ['-', 'SMS-5PM-ENGLISH', 'WH-3PM-ENGLISH']
    enc_dplus1 = MockEncoder(["-", "SMS-5PM-ENGLISH", "WH-3PM-ENGLISH"])
    model_dplus1 = MockModel([
        [0.85, 0.10, 0.05],
        [0.85, 0.10, 0.05],
        [0.85, 0.10, 0.05],
    ])
    dplus1_pred = predict_top_k_by_risk(model_dplus1, enc_dplus1, X, risks, mandatory_communication=True)
    # LOW risk (k=1) picks 2nd-best
    assert dplus1_pred.iloc[0] == "SMS-5PM-ENGLISH"
    # HIGH risk (k=3) shows recommendations by risk
    assert dplus1_pred.iloc[2] == "SMS-5PM-ENGLISH|WH-3PM-ENGLISH|-"


def test_predue_highest_model_probability_selection_keeps_d1():
    from pipeline_common import apply_predue_risk_rules

    # 3 accounts:
    # Account 0 (HIGH risk): D-1 has highest probability (0.45) > D-2 (0.35) > D-4 (0.25) > D-3 (0.10)
    # Target exact is 4 (1 on D-5 + 3 from D-4..D-1). Top 3 by prob are D-1, D-2, D-4. D-3 must be trimmed to '-'!
    # Account 1 (MEDIUM risk): D-1 (0.50) > D-2 (0.40) > D-4 (0.20) > D-3 (0.10)
    # Target exact is 3 (1 on D-5 + 2 from D-4..D-1). Top 2 by prob are D-1, D-2. D-4 and D-3 must be '-'!
    # Account 2 (LOW risk): D-1 (0.60) > D-4 (0.20) > D-2 (0.15) > D-3 (0.05)
    # Target exact is 2 (1 on D-5 + 1 from D-4..D-1). Top 1 by prob is D-1. D-4, D-3, D-2 must be '-'!
    df = pd.DataFrame({
        "Loan_number": ["ACC_HIGH", "ACC_MED", "ACC_LOW"],
        "SOURCE_RISK": ["HIGH", "MEDIUM", "LOW"],
        "D-5": ["WH-2PM-ENGLISH", "WH-2PM-ENGLISH", "WH-2PM-ENGLISH"],
        "D-4": ["SMS-4PM-ENGLISH", "SMS-4PM-ENGLISH", "SMS-4PM-ENGLISH"],
        "D-3": ["WH-1PM-ENGLISH", "WH-1PM-ENGLISH", "WH-1PM-ENGLISH"],
        "D-2": ["SMS-12PM-ENGLISH", "SMS-12PM-ENGLISH", "SMS-12PM-ENGLISH"],
        "D-1": ["WH-4PM-ENGLISH", "WH-4PM-ENGLISH", "WH-4PM-ENGLISH"],
    })
    day_probs = {
        "D-4": pd.Series([0.25, 0.20, 0.20]),
        "D-3": pd.Series([0.10, 0.10, 0.05]),
        "D-2": pd.Series([0.35, 0.40, 0.15]),
        "D-1": pd.Series([0.45, 0.50, 0.60]),
    }

    res = apply_predue_risk_rules(
        df,
        predue_days=["D-5", "D-4", "D-3", "D-2", "D-1"],
        default_strategy=None,
        day_probabilities=day_probs,
    )

    # Account 0 (HIGH risk): D-1, D-2, D-4 kept; D-3 trimmed to '-'! D-1 is NOT blank!
    assert res.loc[0, "D-5"] == "WH-2PM-ENGLISH"
    assert res.loc[0, "D-4"] == "SMS-4PM-ENGLISH"
    assert res.loc[0, "D-3"] == "-"
    assert res.loc[0, "D-2"] == "SMS-12PM-ENGLISH"
    assert res.loc[0, "D-1"] == "WH-4PM-ENGLISH"
    active_high = [c for c in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.loc[0, c] != "-"]
    assert len(active_high) == 4

    # Account 1 (MEDIUM risk): D-1, D-2 kept; D-4, D-3 trimmed to '-'
    assert res.loc[1, "D-5"] == "WH-2PM-ENGLISH"
    assert res.loc[1, "D-4"] == "-"
    assert res.loc[1, "D-3"] == "-"
    assert res.loc[1, "D-2"] == "SMS-12PM-ENGLISH"
    assert res.loc[1, "D-1"] == "WH-4PM-ENGLISH"
    active_med = [c for c in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.loc[1, c] != "-"]
    assert len(active_med) == 3

    # Account 2 (LOW risk): only D-1 kept; D-4, D-3, D-2 trimmed to '-'
    assert res.loc[2, "D-5"] == "WH-2PM-ENGLISH"
    assert res.loc[2, "D-4"] == "-"
    assert res.loc[2, "D-3"] == "-"
    assert res.loc[2, "D-2"] == "-"
    assert res.loc[2, "D-1"] == "WH-4PM-ENGLISH"
    active_low = [c for c in ["D-5", "D-4", "D-3", "D-2", "D-1"] if res.loc[2, c] != "-"]
    assert len(active_low) == 2


def test_predict_best_day_strategy_handles_postdue_cases():
    from pipeline_common import predict_best_day_strategy

    class MockEncoder:
        def __init__(self, classes: list[str]):
            self.classes_ = np.array(classes)

    class MockModel:
        def __init__(self, probs: list[list[float]]):
            self._probs = np.array(probs)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return self._probs

    X = pd.DataFrame({"x": [1, 2, 3]})

    # Case 1: Multiple classes, top is '-' -> picks 2nd best (SMS-4PM-ENGLISH)
    enc1 = MockEncoder(["-", "SMS-4PM-ENGLISH", "WH-2PM-ENGLISH"])
    mod1 = MockModel([
        [0.70, 0.20, 0.10],
        [0.60, 0.35, 0.05],
        [0.80, 0.15, 0.05],
    ])
    strat1, prob1 = predict_best_day_strategy(mod1, enc1, X)
    assert list(strat1) == ["SMS-4PM-ENGLISH", "SMS-4PM-ENGLISH", "SMS-4PM-ENGLISH"]
    assert list(prob1) == [0.20, 0.35, 0.15]

    # Case 2: Top is non-blank (WH-2PM-ENGLISH has highest prob) -> picks best (1st)
    enc2 = MockEncoder(["-", "SMS-4PM-ENGLISH", "WH-2PM-ENGLISH"])
    mod2 = MockModel([
        [0.30, 0.10, 0.60],
        [0.20, 0.30, 0.50],
        [0.40, 0.10, 0.50],
    ])
    strat2, prob2 = predict_best_day_strategy(mod2, enc2, X)
    assert list(strat2) == ["WH-2PM-ENGLISH", "WH-2PM-ENGLISH", "WH-2PM-ENGLISH"]
    assert list(prob2) == [0.60, 0.50, 0.50]

    # Case 3: Only 1 class ('-') -> outputs '-' and prob 0.0
    enc3 = MockEncoder(["-"])
    mod3 = MockModel([
        [1.0],
        [1.0],
        [1.0],
    ])
    strat3, prob3 = predict_best_day_strategy(mod3, enc3, X)
    assert list(strat3) == ["-", "-", "-"]
    assert list(prob3) == [0.0, 0.0, 0.0]


def test_reasoning_matches_column_recommendations():
    import json
    from predict_next_month_strategy_catboost import build_prediction_reason

    # Row has recommendations on D-5, D-4, D-2, and suppressions/no campaign on D-3, D-1
    pred_row = pd.Series({
        "SOURCE_RISK": "HIGH",
        "D-5": "WH-2PM-ENGLISH",
        "D-4": "SMS-4PM-ENGLISH",
        "D-3": "Sunday",
        "D-2": "SMS-12PM-ENGLISH",
        "D-1": "-",
        "D": "-",
        "D+7": "SMS-5PM-ENGLISH",
        "D+10": "SMS-4PM-ENGLISH",
        "D+1": "-",
    })
    source_row = pd.Series({
        "RISK": "HIGH",
        "SMS_TOTAL_INTENSITY": 5,
        "WH_TOTAL_INTENSITY": 2,
    })
    fallback_config = {
        "D+1": {"fallback_reason": "single_unique_value", "fallback_value": "-"},
    }

    reason_json = build_prediction_reason(
        prediction_row=pred_row,
        source_row=source_row,
        day_fallback_config=fallback_config,
        target_days=["D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+7", "D+10"],
    )
    reasons = json.loads(reason_json)

    # 1. D-5: WhatsApp recommended
    assert "WhatsApp at 2PM in English is recommended" in reasons["D-5"]

    # 2. D-4: SMS recommended - MUST NOT say "At this stage, no campaign is recommended"
    assert "SMS at 4PM in English is recommended" in reasons["D-4"]
    assert "no campaign is recommended" not in reasons["D-4"].lower()

    # 3. D-3: Non-working bank day Sunday
    assert "Communication suppressed on D-3 because the scheduled contact date falls on a non-working bank day (Sunday)" in reasons["D-3"]

    # 4. D-2: SMS recommended
    assert "SMS at 12PM in English is recommended" in reasons["D-2"]
    assert "no campaign is recommended" not in reasons["D-2"].lower()

    # 5. D-1: No campaign
    assert "At this stage, no campaign is recommended to prevent excessive communication" in reasons["D-1"]

    # 6. D+7: SMS recommended
    assert "SMS at 5PM in English is recommended" in reasons["D+7"]

    # 7. D+1: Single unique outcome fallback
    assert "contained only one unique outcome" in reasons["D+1"]


def test_regional_language_preserved_in_reason_and_strategy():
    import json
    from pipeline_common import readable_strategy
    from predict_next_month_strategy_catboost import build_prediction_reason

    # Verify readable_strategy preserves REGIONAL without converting to English
    assert readable_strategy("SMS-11AM-REGIONAL") == "SMS at 11AM in Regional language"
    assert readable_strategy("WH-2PM-REGIONAL") == "WhatsApp at 2PM in Regional language"
    assert "English" not in readable_strategy("SMS-11AM-REGIONAL")

    pred_row = pd.Series({
        "SOURCE_RISK": "MEDIUM",
        "D+14": "SMS-11AM-REGIONAL",
    })
    source_row = pd.Series({
        "RISK": "MEDIUM",
        "SMS_TOTAL_INTENSITY": 3,
    })
    reason_json = build_prediction_reason(
        prediction_row=pred_row,
        source_row=source_row,
        target_days=["D+14"],
    )
    reasons = json.loads(reason_json)
    assert "Regional language" in reasons["D+14"]
    assert "English" not in reasons["D+14"]




