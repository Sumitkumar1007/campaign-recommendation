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
    _normalize_campaign_identifier_due_type,
    build_latest_digital_rule_map,
    extract_active_digital_rule_ids,
    fetch_digital_rules_lookup,
    resolve_template_name,
)


def test_resolve_template_name_with_digital_rules():
    # Setup digital_rules lookup with cohort-level and rule-id mappings
    digital_rules_lookup = {
        ("PREDUE", "SMS", "ENGLISH"): "CUSTOM_SMS_RULE_101_ENG",
        ("PREDUE", "SMS", "DEFAULT"): "CUSTOM_SMS_RULE_101_ENG",
        ("PREDUE", "WHATSAPP", "HINDI"): "CUSTOM_WHATSAPP_RULE_102_HIN",
        ("PREDUE", "WHATSAPP", "DEFAULT"): "CUSTOM_WHATSAPP_RULE_102_HIN",
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

    # Test resolution for unknown loan -> cohort lookup from digital_rules_lookup
    name_cohort = resolve_template_name(
        loan_number="ACC_UNKNOWN",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert name_cohort == "CUSTOM_SMS_RULE_101_ENG"

    # Test language fallback in digital_rules_lookup (e.g. TELUGU falls back to DEFAULT)
    name_lang_fallback = resolve_template_name(
        loan_number="ACC_UNKNOWN",
        due_type="PREDUE",
        mode="WHATSAPP",
        language="TELUGU",
        digital_rules_lookup=digital_rules_lookup,
    )
    assert name_lang_fallback == "CUSTOM_WHATSAPP_RULE_102_HIN"

    # When no rule exists in digital_rules_lookup, synthesized fallback is commented out -> returns ""
    name_empty = resolve_template_name(
        loan_number="ACC_UNKNOWN",
        due_type="POSTDUE",
        mode="VOICE",
        language="ENGLISH",
        digital_rules_lookup=digital_rules_lookup,
    )
    assert name_empty == ""


def test_normalize_campaign_identifier_due_type():
    assert _normalize_campaign_identifier_due_type("PRE_DUE_CAMPAIGN") == "PREDUE"
    assert _normalize_campaign_identifier_due_type("PREDUE") == "PREDUE"
    assert _normalize_campaign_identifier_due_type("POST_DUE_CAMPAIGN") == "POSTDUE"
    assert _normalize_campaign_identifier_due_type("POSTDUE") == "POSTDUE"
    assert _normalize_campaign_identifier_due_type("DUE_DATE") == "DUEDATE"
    # Fallback to inspecting iteration
    assert _normalize_campaign_identifier_due_type(None, iteration="PREDUE_SMS") == "PREDUE"
    assert _normalize_campaign_identifier_due_type(None, iteration="POSTDUE-VOICE") == "POSTDUE"


def test_extract_active_digital_rule_ids(tmp_path: Path):
    file1 = tmp_path / "comm_old.csv"
    pd.DataFrame({
        "apac_card_number": ["1", "2"],
        "digital_rule_id": ["10", "20"],
    }).to_csv(file1, index=False)

    file2 = tmp_path / "comm_latest.csv"
    pd.DataFrame({
        "apac_card_number": ["3", "4", "5"],
        "digital_rule_id": ["101", "102", "101"],
    }).to_csv(file2, index=False)

    # Full scan extracts rule IDs across all historical communication files
    rule_ids = extract_active_digital_rule_ids([file1, file2])
    assert rule_ids == ["10", "20", "101", "102"]


def test_fetch_digital_rules_lookup_with_active_rule_ids():
    class DummyCursor:
        def __init__(self):
            self.last_query = None
            self.last_params = None

        def execute(self, query, params=None):
            self.last_query = str(query)
            self.last_params = params

        def fetchall(self):
            return [
                (50, "SMS", "ENGLISH", "PREDUE", "OLD_PREDUE_SMS_V1"),
                (101, "SMS", "ENGLISH", "PREDUE", "PREDUE_SMS"),
                (102, "WHATSAPP", "ENGLISH", "POSTDUE", "POSTDUE-WHATSAPP"),
            ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class DummyConn:
        def __init__(self):
            self.cursor_obj = DummyCursor()

        def cursor(self):
            return self.cursor_obj

    conn = DummyConn()
    active_ids = ["50", "101", "102"]
    lookup = fetch_digital_rules_lookup(conn, schema="digital_collections", active_rule_ids=active_ids)

    # Verified query contains AND id = ANY(%s) and ORDER BY id ASC
    assert "AND id = ANY" in conn.cursor_obj.last_query
    assert "ORDER BY id ASC" in conn.cursor_obj.last_query
    assert conn.cursor_obj.last_params == [[50, 101, 102]]

    # Verified newer rule 101 took precedence over older rule 50 for (PREDUE, SMS, ENGLISH)
    assert lookup.get(("PREDUE", "SMS", "ENGLISH")) == "PREDUE_SMS"
    assert lookup.get(("PREDUE", "SMS", "DEFAULT")) == "PREDUE_SMS"
    assert lookup.get(("POSTDUE", "WHATSAPP", "ENGLISH")) == "POSTDUE-WHATSAPP"



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

