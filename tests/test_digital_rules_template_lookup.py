from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_mcollect_scheduler import dataset_query_for
from run_monthly_inference_pipeline import (
    _template_name,
    build_latest_digital_rule_map,
    resolve_template_name,
)


def test_resolve_template_name_with_digital_rules():
    # Setup digital_rules lookup: (digital_rule_id, mode, language) -> iteration
    digital_rules_lookup = {
        ("101", "SMS", "ENGLISH"): "CUSTOM_SMS_RULE_101_ENG",
        ("102", "WHATSAPP", "HINDI"): "CUSTOM_WHATSAPP_RULE_102_HIN",
    }

    # Setup history digital_rule_id map: (loan_number, mode, language) -> digital_rule_id
    digital_rule_id_map = {
        ("ACC001", "SMS", "ENGLISH"): "101",
        ("ACC002", "WHATSAPP", "HINDI"): "102",
    }

    # Test resolution with valid digital_rule_id match
    name1 = resolve_template_name(
        loan_number="ACC001",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert name1 == "CUSTOM_SMS_RULE_101_ENG"

    # Test resolution for unknown loan -> fallback to standard _template_name
    name_fallback = resolve_template_name(
        loan_number="ACC_UNKNOWN",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert name_fallback == _template_name(due_type="PREDUE", mode="SMS", language="ENGLISH")


def test_build_latest_digital_rule_map(tmp_path):
    comm_file = tmp_path / "comm_sample.csv"
    df_comm = pd.DataFrame(
        [
            {
                "apac_card_number": "ACC100",
                "communication_type": "SMS",
                "verbiage_language": "ENGLISH",
                "digital_rule_id": "45",
                "created_date": "2026-04-01 10:00:00",
            },
            {
                "apac_card_number": "ACC100",
                "communication_type": "SMS",
                "verbiage_language": "ENGLISH",
                "digital_rule_id": "99",
                "created_date": "2026-04-05 12:00:00",  # Latest
            },
        ]
    )
    df_comm.to_csv(comm_file, index=False)

    rule_map = build_latest_digital_rule_map([comm_file])
    assert rule_map.get(("ACC100", "SMS", "ENGLISH")) == "99"


def test_dataset_query_decoupled_from_template_name():
    row = pd.Series(
        {
            "mode": "SMS",
            "vertical": "LAP",
            "template_name": "CUSTOM_ITERATION_NAME_WITHOUT_LANG_SUFFIX",
            "language": "ENGLISH",
            "risk": "LR",
            "emi_cycle": 5,
            "date": "01,02",
            "time": "11:00:00",
            "source_month": "APR-2026",
            "prediction_month": "MAY-2026",
        }
    )

    query = dataset_query_for(
        row,
        schema="digital_collections",
        campaign_table="ai_ml_campaign_recommendations",
        mapping_table="ai_ml_campaign_mapping",
    )

    assert "amcm.language = 'ENGLISH'" in query
    assert "amcm.mode = 'SMS'" in query
    assert "amcm.vertical = 'LAP'" in query


def test_dataset_query_strips_template_name_noise():
    row = pd.Series(
        {
            "mode": "SMS",
            "vertical": "EL",
            "template_name": "English-Pre Due Date Reminder",
            "risk": "LR",
            "emi_cycle": 7,
            "date": "D-5,D-1",
            "time": "11:00:00",
            "source_month": "SEP-2026",
            "prediction_month": "OCT-2026",
        }
    )

    query = dataset_query_for(
        row,
        schema="digital_collections",
        campaign_table="ai_ml_campaign_recommendations",
        mapping_table="ai_ml_campaign_mapping",
    )

    assert "amcm.language = 'ENGLISH'" in query
    assert "ENGLISH-PRE DUE DATE REMINDER" not in query
    assert "English-Pre Due Date Reminder" not in query

