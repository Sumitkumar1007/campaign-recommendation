from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_common import (
    apply_predue_risk_rules,
    day_label_to_offset,
    offset_to_day_label,
    offsets_to_day_columns,
    snap_hour_to_window,
)
from run_monthly_inference_pipeline import (
    _format_scheduler_hour,
    build_campaign_recommendations,
)


def test_offsets_to_day_columns_conversion_and_sorting():
    # Mix of negative, zero, positive integers and strings
    raw_offsets = [-5, -2, 0, 3, 1, "-1", "D+5", 10]
    result = offsets_to_day_columns(raw_offsets)
    assert result == ["D-5", "D-2", "D-1", "D", "D+1", "D+3", "D+5", "D+10"]


def test_snap_hour_to_window_clamping_and_stepping():
    # Window 9 to 18 with 2-hour steps -> [9, 11, 13, 15, 17, 18]
    window = {"start_hour": 9, "end_hour": 18, "step_hours": 2}

    # Before start hour -> clamped to 9
    assert snap_hour_to_window(7, window) == 9
    # Exact start hour
    assert snap_hour_to_window(9, window) == 9
    # Hour 10 is between 9 and 11 -> snaps to 9 (tie-breaker)
    assert snap_hour_to_window(10, window) == 9
    # Hour 11 is exact candidate
    assert snap_hour_to_window(11, window) == 11
    # Hour 12 is between 11 and 13 -> snaps to 11
    assert snap_hour_to_window(12, window) == 11
    # Hour 17 is exact candidate
    assert snap_hour_to_window(17, window) == 17
    # Hour 18 is exact end candidate
    assert snap_hour_to_window(18, window) == 18
    # After end hour -> clamped to 18
    assert snap_hour_to_window(21, window) == 18


def test_format_scheduler_hour_with_window():
    window = {"start_hour": 10, "end_hour": 18, "step_hours": 2}
    # 11AM snaps to 10:00:00 (since candidates are 10, 12, 14, 16, 18)
    assert _format_scheduler_hour("11AM", send_hour_window=window) == "10:00:00"
    # 12PM is an exact match -> 12:00:00
    assert _format_scheduler_hour("12PM", send_hour_window=window) == "12:00:00"
    # 8AM is below start -> clamped to 10:00:00
    assert _format_scheduler_hour("8AM", send_hour_window=window) == "10:00:00"


def test_apply_predue_risk_rules_with_custom_days():
    # Custom window with only D-3, D-2, D-1 (no D-5)
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_01",
                "SOURCE_RISK": "LOW",
                "D-3": "-",
                "D-2": "-",
                "D-1": "-",
            }
        ]
    )
    res = apply_predue_risk_rules(df, predue_days=["D-3", "D-2", "D-1"])
    # Target exact for LOW is 2 touchpoints across available pre-due days
    active = [col for col in ["D-3", "D-2", "D-1"] if res.iloc[0][col] != "-"]
    assert len(active) == 2
    # Filled starting from latest pre-due day (D-1, D-2)
    assert res.iloc[0]["D-1"] == "SMS-11AM-ENGLISH"
    assert res.iloc[0]["D-2"] == "SMS-11AM-ENGLISH"
    assert res.iloc[0]["D-3"] == "-"


def test_build_campaign_recommendations_filters_to_allowed_offsets(tmp_path: Path):
    prediction_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "Loan_number": ["ACC_01"],
            "MONTH": ["MAY-2026"],
            "SOURCE_RISK": ["LOW"],
            "VERTICAL": ["LAP"],
            "emi_date": ["05/05/2026"],
            "D-5": ["SMS-10AM-HINDI"],
            "D-3": ["SMS-11AM-HINDI"],
            "D-1": ["SMS-12PM-HINDI"],
            "D+1": ["SMS-2PM-HINDI"],
            "D+5": ["SMS-4PM-HINDI"],
        }
    ).to_csv(prediction_file, index=False)

    # Restrict prediction window to only D-5 and D+1
    output = build_campaign_recommendations(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
        allowed_day_offsets=[-5, 1],
    )

    # Campaign recommendations should only contain D-5 and D+1
    dates = set(output["date"].tolist())
    assert "D-5" in dates
    assert "D+1" in dates
    # D-3, D-1, and D+5 must not be included
    assert "D-3" not in dates
    assert "D-1" not in dates
    assert "D+5" not in dates


def test_build_campaign_recommendations_snaps_hours(tmp_path: Path):
    prediction_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "Loan_number": ["ACC_01"],
            "MONTH": ["MAY-2026"],
            "SOURCE_RISK": ["LOW"],
            "VERTICAL": ["LAP"],
            "emi_date": ["05/05/2026"],
            "D-5": ["SMS-11AM-HINDI"],
        }
    ).to_csv(prediction_file, index=False)

    output = build_campaign_recommendations(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
        send_hour_window={"start_hour": 10, "end_hour": 18, "step_hours": 2},
    )

    # 11AM should have been snapped to 10:00:00
    assert output.iloc[0]["time"] == "10:00:00"


def test_apply_postdue_risk_rules_caps_to_single_prediction_per_day():
    from pipeline_common import apply_postdue_risk_rules

    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_0781",
                "D+1": "VOICE-3PM-ENGLISH|WH-1PM-ENGLISH",
                "D+2": "VOICE-1PM-ENGLISH|WH-12PM-ENGLISH",
            }
        ]
    )
    res = apply_postdue_risk_rules(df)
    assert res.iloc[0]["D+1"] == "VOICE-3PM-ENGLISH"
    assert res.iloc[0]["D+2"] == "VOICE-1PM-ENGLISH"


def test_build_campaign_mappings_single_communication_per_day(tmp_path: Path):
    from run_monthly_inference_pipeline import build_campaign_mappings

    prediction_file = tmp_path / "predictions_postdue.csv"
    pd.DataFrame(
        {
            "Loan_number": ["0781109469306"],
            "MONTH": ["MAY-2026"],
            "SOURCE_RISK": ["HIGH"],
            "VERTICAL": ["CE"],
            "emi_date": ["07/05/2026"],
            "D+1": ["VOICE-3PM-ENGLISH|WH-1PM-ENGLISH"],
            "D+2": ["VOICE-1PM-ENGLISH|WH-12PM-ENGLISH"],
        }
    ).to_csv(prediction_file, index=False)

    mappings = build_campaign_mappings(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[7],
        vertical="CE",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
        single_communication_per_day=True,
    )

    # When single_communication_per_day=True, each day has at most 1 communication row
    d_plus_1_mappings = mappings[mappings["date"] == "D+1"]
    d_plus_2_mappings = mappings[mappings["date"] == "D+2"]
    assert len(d_plus_1_mappings) == 1
    assert d_plus_1_mappings.iloc[0]["mode"] == "VOICE"
    assert len(d_plus_2_mappings) == 1
    assert d_plus_2_mappings.iloc[0]["mode"] == "VOICE"


def test_get_configured_default_template_string_and_dict_formats():
    from pipeline_common import get_configured_default_template

    config_strings = {
        "PREDUE": {
            "WHATSAPP": "PREDUE WHATSAPP ENGLISH",
            "SMS": "PREDUE_SMS",
        },
        "POSTDUE": {
            "WHATSAPP": "POSTDUE-WHATSAPP",
            "SMS": "POSTDUE-SMS",
            "VOICE": "POSTDUE-VOICE",
        },
    }
    # Direct lookups
    assert get_configured_default_template(config_strings, due_type="PREDUE", mode="WHATSAPP") == "PREDUE WHATSAPP ENGLISH"
    assert get_configured_default_template(config_strings, due_type="PRE", mode="WH") == "PREDUE WHATSAPP ENGLISH"
    assert get_configured_default_template(config_strings, due_type="PREDUE", mode="SMS") == "PREDUE_SMS"
    assert get_configured_default_template(config_strings, due_type="POSTDUE", mode="SMS") == "POSTDUE-SMS"
    assert get_configured_default_template(config_strings, due_type="POST", mode="VOICE") == "POSTDUE-VOICE"
    assert get_configured_default_template(config_strings, due_type="POSTDUE", mode="VOICE_BOT") == "POSTDUE-VOICE"

    # Dict format with name, id, and language
    config_objects = {
        "PREDUE": {
            "WHATSAPP": {
                "name": "PREDUE WHATSAPP REGIONAL",
                "id": "101",
                "language": "HINDI",
            },
            "SMS": {
                "id": "202",
                "template_name": "PREDUE_SMS_CUSTOM",
            },
        }
    }
    assert get_configured_default_template(config_objects, due_type="PREDUE", mode="WHATSAPP") == "PREDUE WHATSAPP REGIONAL"
    assert get_configured_default_template(config_objects, due_type="PREDUE", mode="SMS") == "PREDUE_SMS_CUSTOM"


def test_resolve_template_name_priority():
    from run_monthly_inference_pipeline import resolve_template_name

    default_templates = {
        "PREDUE": {"SMS": "PREDUE_SMS", "WHATSAPP": "PREDUE WHATSAPP ENGLISH"},
        "POSTDUE": {"SMS": "POSTDUE-SMS"},
    }

    # 1. Fallback to default_templates when no history map
    res = resolve_template_name(
        loan_number="ACC_01",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        default_templates=default_templates,
    )
    assert res == "PREDUE_SMS"

    # 2. History map present and matching digital_rules_lookup takes precedence
    digital_rule_id_map = {("ACC_01", "SMS", "ENGLISH"): "RULE_99"}
    digital_rules_lookup = {("RULE_99", "SMS", "ENGLISH"): "MCOLLECT_EXISTING_ITERATION_1"}
    res_history = resolve_template_name(
        loan_number="ACC_01",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_lookup,
        default_templates=default_templates,
    )
    assert res_history == "MCOLLECT_EXISTING_ITERATION_1"

    # 3. Neither history nor default_templates -> synthesized fallback
    res_synthesized = resolve_template_name(
        loan_number="ACC_01",
        due_type="POSTDUE",
        mode="VOICE",
        language="ENGLISH",
        default_templates=default_templates,
    )
    assert res_synthesized == "POSTDUE_AIML_VOICE_ENGLISH"


def test_build_campaign_recommendations_uses_default_templates(tmp_path: Path):
    prediction_file = tmp_path / "pred_templates.csv"
    pd.DataFrame(
        {
            "Loan_number": ["ACC_01"],
            "MONTH": ["MAY-2026"],
            "SOURCE_RISK": ["LOW"],
            "VERTICAL": ["LAP"],
            "emi_date": ["05/05/2026"],
            "D-1": ["WH-10AM-ENGLISH"],
            "D+1": ["SMS-11AM-ENGLISH"],
        }
    ).to_csv(prediction_file, index=False)

    default_templates = {
        "PREDUE": {
            "WHATSAPP": "PREDUE WHATSAPP ENGLISH",
            "SMS": "PREDUE_SMS",
        },
        "POSTDUE": {
            "SMS": "POSTDUE-SMS",
        },
    }

    output = build_campaign_recommendations(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
        default_templates=default_templates,
    )

    predue_row = output[output["date"] == "D-1"].iloc[0]
    postdue_row = output[output["date"] == "D+1"].iloc[0]

    assert predue_row["template_name"] == "PREDUE WHATSAPP ENGLISH"
    assert postdue_row["template_name"] == "POSTDUE-SMS"

