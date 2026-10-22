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

