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

    digital_rules_lookup = {
        ("PREDUE", "SMS", "ENGLISH"): "PREDUE_SMS",
        ("PREDUE", "SMS", "DEFAULT"): "PREDUE_SMS",
        ("POSTDUE", "SMS", "ENGLISH"): "POSTDUE-SMS",
        ("POSTDUE", "SMS", "DEFAULT"): "POSTDUE-SMS",
        ("RULE_99", "SMS", "ENGLISH"): "MCOLLECT_EXISTING_ITERATION_1",
    }

    # 1. Exact cohort match from digital_rules_lookup
    res = resolve_template_name(
        loan_number="ACC_01",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rules_lookup=digital_rules_lookup,
    )
    assert res == "PREDUE_SMS"

    # 2. Account-specific rule ID lookup (e.g. for a custom rule without cohort key)
    digital_rules_custom = {
        ("RULE_99", "SMS", "ENGLISH"): "MCOLLECT_EXISTING_ITERATION_1",
    }
    digital_rule_id_map = {("ACC_01", "SMS", "ENGLISH"): "RULE_99"}
    res_history = resolve_template_name(
        loan_number="ACC_01",
        due_type="PREDUE",
        mode="SMS",
        language="ENGLISH",
        digital_rule_id_map=digital_rule_id_map,
        digital_rules_lookup=digital_rules_custom,
    )
    assert res_history == "MCOLLECT_EXISTING_ITERATION_1"

    # 3. Mode not in digital_rules_lookup -> commented out fallbacks return ""
    res_none = resolve_template_name(
        loan_number="ACC_01",
        due_type="POSTDUE",
        mode="VOICE",
        language="ENGLISH",
        digital_rules_lookup=digital_rules_lookup,
    )
    assert res_none == ""


def test_build_campaign_recommendations_uses_digital_rules(tmp_path: Path):
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

    digital_rules_lookup = {
        ("PREDUE", "WHATSAPP", "ENGLISH"): "PREDUE WHATSAPP ENGLISH",
        ("POSTDUE", "SMS", "ENGLISH"): "POSTDUE-SMS",
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
        default_templates=None,
    )

    # With digital_rules_lookup passed into build_campaign_assignment_groups
    from run_monthly_inference_pipeline import _build_campaign_assignment_groups

    assignment_groups = _build_campaign_assignment_groups(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        run_token="080426",
        digital_rules_lookup=digital_rules_lookup,
    )

    predue_row = assignment_groups[assignment_groups["date"] == "D-1"].iloc[0]
    postdue_row = assignment_groups[assignment_groups["date"] == "D+1"].iloc[0]

    assert predue_row["template_name"] == "PREDUE WHATSAPP ENGLISH"
    assert postdue_row["template_name"] == "POSTDUE-SMS"


def test_default_config_json_is_valid_and_loadable():
    import json
    from generate_strategy_dataset import load_send_hour_window
    from pipeline_common import load_default_config

    repo_root = Path(__file__).resolve().parent.parent
    config_file = repo_root / "config" / "default_config.json"
    assert config_file.exists(), "config/default_config.json should exist"

    # Strict JSON validation
    with config_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "send_hour_window" in data
    assert "default_templates" in data
    assert "POSTDUE" in data["default_templates"]

    # Verify script loaders work cleanly with it
    window = load_send_hour_window(config_file)
    assert window["start_hour"] == 9
    assert window["end_hour"] == 18

    app_cfg = load_default_config(config_file)
    assert app_cfg["default_templates"]["POSTDUE"]["VOICE"] == "POSTDUE-VOICE"


def test_config_loaders_handle_commented_json(tmp_path: Path):
    from generate_strategy_dataset import load_send_hour_window
    from pipeline_common import load_default_config

    commented_json = tmp_path / "commented_config.json"
    commented_json.write_text(
        """{
        "send_hour_window": {
            "start_hour": 10,
            "end_hour": 17,
            "step_hours": 2
        },
        "default_templates": {
            "POSTDUE": {
                "SMS": "POSTDUE-SMS"
                // "VOICE": "COMMENTED-OUT"
            }
        }
    }""",
        encoding="utf-8",
    )

    window = load_send_hour_window(commented_json)
    assert window["start_hour"] == 10
    assert window["end_hour"] == 17
    assert window["step_hours"] == 2

    app_cfg = load_default_config(commented_json)
    assert app_cfg["default_templates"]["POSTDUE"]["SMS"] == "POSTDUE-SMS"


def test_load_allowed_day_offsets(tmp_path: Path):
    from generate_strategy_dataset import load_allowed_day_offsets

    cfg = tmp_path / "offsets.json"
    cfg.write_text('{"allowed_day_offsets": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]}')
    offsets = load_allowed_day_offsets(cfg)
    assert offsets == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def test_generate_strategy_dataset_filters_by_allowed_day_offsets():
    from generate_strategy_dataset import process_chunk

    raw_chunk = pd.DataFrame(
        {
            "apac_card_number": ["123", "123", "123"],
            "party_id": ["P1", "P1", "P1"],
            "risk": ["LOW", "LOW", "LOW"],
            "communication_type": ["SMS", "SMS", "SMS"],
            "comm_status": ["DELIVERED", "DELIVERED", "DELIVERED"],
            "emi_date": ["2026-05-05", "2026-05-05", "2026-05-05"],
            "date": ["2026-05-06", "2026-05-10", "2026-05-15"],  # offsets: +1, +5, +10
            "created_date": ["2026-05-06 10:00:00", "2026-05-10 10:00:00", "2026-05-15 10:00:00"],
            "vertical": ["LAP", "LAP", "LAP"],
            "verbiage_language": ["ENGLISH", "ENGLISH", "ENGLISH"],
        }
    )

    allowed = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    _, strat, _ = process_chunk(
        raw_chunk,
        allowed_day_offsets=allowed,
    )

    days_in_dataset = set(strat["DAY"].tolist())
    assert "D+1" in days_in_dataset
    assert "D+5" in days_in_dataset
    assert "D+10" not in days_in_dataset, "D+10 must be excluded when allowed_day_offsets max is 5"


def test_identify_new_customer_fallbacks_respects_allowed_day_labels(tmp_path: Path):
    from identify_new_customer_fallbacks import apply_fallbacks_to_prediction_file, build_fallback_rows
    from pipeline_common import DAY_COLUMNS, SCHEDULE_DAY_COLUMNS

    new_cases = pd.DataFrame(
        [
            {
                "apacCardNumber": "NEW_01",
                "originalRisk": "LOW",
                "vertical": "LAP",
            }
        ]
    )
    fallback_by_risk = {
        "LOW": {day: "SMS-11AM-ENGLISH" for day in DAY_COLUMNS}
    }
    allowed_labels = {"D-5", "D-1", "D+1", "D+5"}

    fallback_rows = build_fallback_rows(new_cases, fallback_by_risk, allowed_day_labels=allowed_labels)
    # Check that rows for disallowed days have "-"
    disallowed_row = fallback_rows[fallback_rows["day"] == "D+10"].iloc[0]
    assert disallowed_row["recommendedStrategy"] == "-"
    allowed_row = fallback_rows[fallback_rows["day"] == "D+1"].iloc[0]
    assert allowed_row["recommendedStrategy"] == "SMS-11AM-ENGLISH"

    pred_file = tmp_path / "pred.csv"
    initial_pred = pd.DataFrame(
        [
            {
                "Loan_number": "NEW_01",
                "SOURCE_MONTH_USED": "APR-2026",
                "MONTH": "MAY-2026",
                **{d: "-" for d in SCHEDULE_DAY_COLUMNS},
            }
        ]
    )
    initial_pred.to_csv(pred_file, index=False)

    apply_fallbacks_to_prediction_file(pred_file, fallback_rows, allowed_day_labels=allowed_labels)
    updated_pred = pd.read_csv(pred_file)
    assert updated_pred.iloc[0]["D+1"] == "SMS-11AM-ENGLISH"
    assert updated_pred.iloc[0]["D+10"] == "-"
    assert updated_pred.iloc[0]["D+20"] == "-"


def test_build_prediction_summary_dynamic_prediction_window(tmp_path: Path):
    from run_monthly_inference_pipeline import build_prediction_summary

    pred_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "MONTH": "MAY-2026",
                "SOURCE_RISK": "LOW",
                "D-5": "SMS-11AM-ENGLISH",
                "D-1": "SMS-11AM-ENGLISH",
                "D+1": "SMS-11AM-ENGLISH",
                "D+2": "SMS-11AM-ENGLISH",
                "D+3": "SMS-11AM-ENGLISH",
            }
        ]
    ).to_csv(pred_file, index=False)

    # 1. With explicit configured campaign_days (e.g. 5 days)
    configured_days = ["D-5", "D-1", "D+1", "D+2", "D+3"]
    summary = build_prediction_summary(
        prediction_file=pred_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        campaign_df=pd.DataFrame(),
        mapping_df=pd.DataFrame(),
        campaign_days=configured_days,
    )

    assert summary["prediction_days_count"] == 5
    assert summary["prediction_window"] == "5 days"
    assert summary["prediction_days"] == configured_days

    # 2. With 10 days
    ten_days = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
    summary_10 = build_prediction_summary(
        prediction_file=pred_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        campaign_df=pd.DataFrame(),
        mapping_df=pd.DataFrame(),
        campaign_days=ten_days,
    )

    assert summary_10["prediction_days_count"] == 10
    assert summary_10["prediction_window"] == "10 days"
    assert summary_10["prediction_days"] == ten_days



