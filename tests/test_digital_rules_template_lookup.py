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
    emit_template_resolution_audit,
    extract_active_digital_rule_ids,
    fetch_digital_rules_lookup,
    get_template_logger,
    resolve_template_name,
    resolve_template_name_with_status,
    validate_template_due_type_consistency,
    validate_template_language_consistency,
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


def test_validate_template_due_type_consistency():
    # Pre-due should reject pure post-due templates
    assert validate_template_due_type_consistency("ENGLISH POST TEST WHATSAPP", "PREDUE") is False
    assert validate_template_due_type_consistency("POSTDUE_REMINDER_SMS", "PREDUE") is False
    assert validate_template_due_type_consistency("ENGLISH PRE TEST WHATSAPP", "PREDUE") is True
    assert validate_template_due_type_consistency("PREDUE_SMS", "PREDUE") is True

    # Post-due should reject pure pre-due templates
    assert validate_template_due_type_consistency("ENGLISH PRE TEST WHATSAPP", "POSTDUE") is False
    assert validate_template_due_type_consistency("PREDUE_REMINDER_SMS", "POSTDUE") is False
    assert validate_template_due_type_consistency("ENGLISH POST TEST WHATSAPP", "POSTDUE") is True
    assert validate_template_due_type_consistency("POSTDUE_SMS", "POSTDUE") is True

    # Empty / neutral templates
    assert validate_template_due_type_consistency("", "PREDUE") is True
    assert validate_template_due_type_consistency("PAYMENT_LINK_SMS", "PREDUE") is True
    assert validate_template_due_type_consistency("PAYMENT_LINK_SMS", "POSTDUE") is True


def test_build_latest_digital_rule_map_rejects_inconsistent_due_type(tmp_path: Path):
    comm_file = tmp_path / "comm_test.csv"
    # Row 1: date < emi_date -> PREDUE, but digital_rule_id=200 has POSTDUE metadata
    # Row 2: date < emi_date -> PREDUE, digital_rule_id=201 has PREDUE metadata
    df = pd.DataFrame([
        {
            "apac_card_number": "ACC_BAD",
            "communication_type": "WHATSAPP",
            "verbiage_language": "ENGLISH",
            "digital_rule_id": "200",
            "date": "2026-09-01",
            "emi_date": "2026-09-05",
            "created_date": "2026-09-01 10:00:00",
        },
        {
            "apac_card_number": "ACC_GOOD",
            "communication_type": "WHATSAPP",
            "verbiage_language": "ENGLISH",
            "digital_rule_id": "201",
            "date": "2026-09-01",
            "emi_date": "2026-09-05",
            "created_date": "2026-09-01 11:00:00",
        },
    ])
    df.to_csv(comm_file, index=False)

    rules_lookup = {
        ("RULE_META", "200"): {
            "due_type": "POSTDUE",
            "mode": "WHATSAPP",
            "language": "ENGLISH",
            "iteration": "ENGLISH POST TEST WHATSAPP",
        },
        ("RULE_META", "201"): {
            "due_type": "PREDUE",
            "mode": "WHATSAPP",
            "language": "ENGLISH",
            "iteration": "ENGLISH PRE TEST WHATSAPP",
        },
    }

    rule_map = build_latest_digital_rule_map([comm_file], digital_rules_lookup=rules_lookup)

    # ACC_BAD inconsistent rule should NOT be registered for PREDUE
    assert ("ACC_BAD", "PREDUE", "WHATSAPP", "ENGLISH") not in rule_map

    # ACC_GOOD consistent rule SHOULD be registered
    assert rule_map.get(("ACC_GOOD", "PREDUE", "WHATSAPP", "ENGLISH")) == ("201", "ENGLISH PRE TEST WHATSAPP")

    # COMM_COHORT should have the consistent pre-due template
    assert rule_map.get(("COMM_COHORT", "PREDUE", "WHATSAPP", "ENGLISH")) == "ENGLISH PRE TEST WHATSAPP"
    assert rule_map.get(("COMM_COHORT", "PREDUE", "WHATSAPP", "DEFAULT")) == "ENGLISH PRE TEST WHATSAPP"


def test_resolve_template_name_priority_order():
    # Setup digital_rule_id_map with:
    # 1. Historical account rule for ACC_HIST
    # 2. COMM_COHORT latest communication templates
    digital_rule_id_map = {
        ("ACC_HIST", "PREDUE", "WHATSAPP", "ENGLISH"): ("100", "HIST_PREDUE_WHATSAPP_ENG"),
        ("COMM_COHORT", "PREDUE", "WHATSAPP", "ENGLISH"): "COMM_PREDUE_WHATSAPP_ENG",
        ("COMM_COHORT", "PREDUE", "WHATSAPP", "DEFAULT"): "COMM_PREDUE_WHATSAPP_ENG",
        ("COMM_COHORT", "PREDUE", "SMS", "ENGLISH"): "COMM_PREDUE_SMS_ENG",
        ("COMM_COHORT", "PREDUE", "SMS", "DEFAULT"): "COMM_PREDUE_SMS_ENG",
    }

    digital_rules_lookup = {
        ("PREDUE", "VOICE", "ENGLISH"): "DB_PREDUE_VOICE_ENG",
        ("PREDUE", "VOICE", "DEFAULT"): "DB_PREDUE_VOICE_ENG",
    }

    config_defaults = {
        "POSTDUE": {
            "SMS": {
                "DEFAULT": "CONFIG_POSTDUE_SMS_DEF",
            }
        }
    }

    # Priority 1: Existing account historical rule match
    t1, s1, _ = resolve_template_name_with_status(
        loan_number="ACC_HIST",
        due_type="PREDUE",
        mode="WHATSAPP",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert t1 == "HIST_PREDUE_WHATSAPP_ENG"
    assert s1 == "ACCOUNT_RULE_MATCH"

    # Priority 2: New account (no historical rule) resolves against latest COMM_COHORT
    t2, s2, _ = resolve_template_name_with_status(
        loan_number="ACC_NEW",
        due_type="PREDUE",
        mode="WHATSAPP",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert t2 == "COMM_PREDUE_WHATSAPP_ENG"
    assert s2 == "COMM_COHORT_MATCH"

    # Priority 3: Language fallback in COMM_COHORT (TELUGU -> DEFAULT)
    t3, s3, _ = resolve_template_name_with_status(
        loan_number="ACC_NEW",
        due_type="PREDUE",
        mode="WHATSAPP",
        language="TELUGU",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert t3 == "COMM_PREDUE_WHATSAPP_ENG"
    assert s3 == "COMM_LANG_FALLBACK"

    # Priority 4: Database digital_rules cohort fallback (mode not in comms)
    t4, s4, _ = resolve_template_name_with_status(
        loan_number="ACC_NEW",
        due_type="PREDUE",
        mode="VOICE",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert t4 == "DB_PREDUE_VOICE_ENG"
    assert s4 == "DB_COHORT_MATCH"

    # Priority 5: Config defaults fallback
    t5, s5, _ = resolve_template_name_with_status(
        loan_number="ACC_NEW",
        due_type="POSTDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
        default_templates=config_defaults,
    )
    assert t5 == "CONFIG_POSTDUE_SMS_DEF"
    assert s5 == "CONFIG_FALLBACK"

    # Priority 6: Unmatched / Missing -> Return empty string and FAILED status
    t6, s6, reason6 = resolve_template_name_with_status(
        loan_number="ACC_NEW",
        due_type="POSTDUE",
        mode="TELECALLING",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert t6 == ""
    assert s6 == "FAILED"
    assert "No communication template or digital rule found for mode 'TELECALLING'" in reason6


def test_template_standalone_logger_isolation(tmp_path: Path):
    log_file = tmp_path / "test_template_resolution.log"
    logger = get_template_logger(log_file=log_file)

    assert logger.name == "template_resolution"
    assert logger.propagate is False

    test_msg = "STANDALONE_TEMPLATE_AUDIT_LOG_CHECK"
    logger.info(test_msg)

    # Flush handlers
    for h in logger.handlers:
        h.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content


def test_validate_template_language_consistency():
    # English prediction rejects regional template names
    assert validate_template_language_consistency("Marathi_Backup_Field_agency_WA_Template", "ENGLISH") is False
    assert validate_template_language_consistency("Malayalam_Backup_Field_agency_Template", "ENGLISH") is False
    assert validate_template_language_consistency("PREDUE_AIML_SMS_HINDI", "ENGLISH") is False

    # English prediction accepts valid English / neutral templates
    assert validate_template_language_consistency("English-Pre Due Date Reminder", "ENGLISH") is True
    assert validate_template_language_consistency("ENGLISH TEST WHATSAPP", "ENGLISH") is True
    assert validate_template_language_consistency("PAYMENT_LINK_SMS", "ENGLISH") is True

    # Regional language accepts its own template
    assert validate_template_language_consistency("Marathi_Backup_Field_agency_WA_Template", "MARATHI") is True
    assert validate_template_language_consistency("Malayalam_Backup_Field_agency_Template", "MALAYALAM") is True

    # Regional language rejects conflicting regional template
    assert validate_template_language_consistency("Marathi_Backup_Field_agency_WA_Template", "HINDI") is False
    assert validate_template_language_consistency("PREDUE_AIML_SMS_HINDI", "MARATHI") is False


def test_resolve_template_name_rejects_incompatible_rule_language():
    """
    UAT Scenario:
    Account has historical rule 171 which in digital_rules is defined as MARATHI.
    The prediction strategy is English (WH-14-ENGLISH).
    Resolution must reject rule 171 and resolve the valid English cohort template.
    """
    digital_rule_id_map = {
        ("MOB-TEST-Aryan_AIML", "PREDUE", "WHATSAPP", "MARATHI"): ("171", "Marathi_Backup_Field_agency_WA_Template"),
        ("MOB-TEST-Aryan_AIML", "PREDUE", "WHATSAPP", ""): ("171", "Marathi_Backup_Field_agency_WA_Template"),
        ("COMM_COHORT", "PREDUE", "WHATSAPP", "ENGLISH"): "ENGLISH TEST WHATSAPP",
        ("COMM_COHORT", "PREDUE", "WHATSAPP", "DEFAULT"): "ENGLISH TEST WHATSAPP",
    }

    digital_rules_lookup = {
        ("PREDUE", "WHATSAPP", "ENGLISH"): "ENGLISH TEST WHATSAPP",
        ("PREDUE", "WHATSAPP", "DEFAULT"): "ENGLISH TEST WHATSAPP",
        ("PREDUE", "WHATSAPP", "MARATHI"): "Marathi_Backup_Field_agency_WA_Template",
        ("171", "WHATSAPP", "MARATHI"): "Marathi_Backup_Field_agency_WA_Template",
        ("171", "WHATSAPP", ""): "Marathi_Backup_Field_agency_WA_Template",
        ("RULE_META", "171"): {
            "due_type": "PREDUE",
            "mode": "WHATSAPP",
            "language": "MARATHI",
            "iteration": "Marathi_Backup_Field_agency_WA_Template",
        },
    }

    # When prediction is ENGLISH, rule 171 (Marathi) MUST be rejected and resolve to English cohort template!
    template_name, status, reason = resolve_template_name_with_status(
        loan_number="MOB-TEST-Aryan_AIML",
        due_type="PREDUE",
        mode="WHATSAPP",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert template_name == "ENGLISH TEST WHATSAPP"
    assert status == "COMM_COHORT_MATCH"
    assert "Marathi" not in template_name

    # Conversely, when prediction is MARATHI, rule 171 (Marathi) matches successfully!
    template_name_mar, status_mar, _ = resolve_template_name_with_status(
        loan_number="MOB-TEST-Aryan_AIML",
        due_type="PREDUE",
        mode="WHATSAPP",
        language="MARATHI",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
    )
    assert template_name_mar == "Marathi_Backup_Field_agency_WA_Template"
    assert status_mar == "ACCOUNT_RULE_MATCH"


def test_build_latest_digital_rule_map_rejects_language_mismatch(tmp_path: Path):
    """
    Verify that if comm_data has verbiage_language = ENGLISH, but the digital_rule_id
    is defined as MARATHI in digital_rules metadata, it is rejected as a mismatch.
    """
    comm_file = tmp_path / "comm_lang_test.csv"
    df = pd.DataFrame([
        {
            "apac_card_number": "MOB-TEST-Aryan_AIML",
            "communication_type": "WHATSAPP",
            "verbiage_language": "ENGLISH",
            "digital_rule_id": "171",
            "date": "2026-09-01",
            "emi_date": "2026-09-05",
            "created_date": "2026-09-01 10:00:00",
        }
    ])
    df.to_csv(comm_file, index=False)

    rules_lookup = {
        ("RULE_META", "171"): {
            "due_type": "PREDUE",
            "mode": "WHATSAPP",
            "language": "MARATHI",
            "iteration": "Marathi_Backup_Field_agency_WA_Template",
        },
    }

    rule_map = build_latest_digital_rule_map([comm_file], digital_rules_lookup=rules_lookup)

    # Inconsistent language: comm said ENGLISH, rule 171 in digital_rules is MARATHI
    # It must NOT be registered under ("MOB-TEST-Aryan_AIML", "PREDUE", "WHATSAPP", "ENGLISH")
    assert ("MOB-TEST-Aryan_AIML", "PREDUE", "WHATSAPP", "ENGLISH") not in rule_map
    assert ("COMM_COHORT", "PREDUE", "WHATSAPP", "ENGLISH") not in rule_map

