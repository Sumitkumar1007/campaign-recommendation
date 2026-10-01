from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_common import apply_predue_risk_rules


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
