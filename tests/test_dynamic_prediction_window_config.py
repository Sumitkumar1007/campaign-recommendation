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
    format_hour_label,
    offset_to_day_label,
    offsets_to_day_columns,
    parse_hour_to_int,
    snap_hour_to_window,
    snap_prediction_dataframe_to_window,
    snap_strategy_to_window,
    snap_strategy_token_to_window,
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


def test_predict_catboost_cli_log_file(monkeypatch):
    import sys
    from predict_next_month_strategy_catboost import parse_args

    test_args = [
        "predict_next_month_strategy_catboost.py",
        "--feature-file", "features.csv",
        "--schedule-file", "schedules.csv",
        "--model-file", "model.joblib",
        "--base-population-file", "cases.csv",
        "--prediction-file", "preds.csv",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    args = parse_args()
    assert hasattr(args, "log_file")
    assert args.log_file is None

    test_args_with_log = test_args + ["--log-file", "/tmp/test.log"]
    monkeypatch.setattr(sys, "argv", test_args_with_log)
    args_with_log = parse_args()
    assert args_with_log.log_file == "/tmp/test.log"


def test_build_strategy_schedule_dataset_dynamic_columns_and_training_full_columns(tmp_path: Path, monkeypatch):
    import sys
    from build_strategy_schedule_dataset import main
    from pipeline_common import SCHEDULE_DAY_COLUMNS

    input_csv = tmp_path / "training_data.csv"
    pd.DataFrame(
        [
            {
                "ENTITY_KEY": "ACC_01",
                "MONTH": "MAY-2026",
                "DAY": "D-5",
                "RISK": "LOW",
                "PREDICTED_STRATEGY": "SMS-11AM-ENGLISH",
            },
            {
                "ENTITY_KEY": "ACC_01",
                "MONTH": "MAY-2026",
                "DAY": "D+1",
                "RISK": "LOW",
                "PREDICTED_STRATEGY": "WH-2PM-ENGLISH",
            },
        ]
    ).to_csv(input_csv, index=False)

    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"allowed_day_offsets": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]}')

    # 1. With config-file (Inference mode): dynamic columns (only D-5..D+5)
    out_inference = tmp_path / "schedule_inference.csv"
    test_args_inf = [
        "build_strategy_schedule_dataset.py",
        "--input-file", str(input_csv),
        "--output-file", str(out_inference),
        "--config-file", str(cfg),
    ]
    monkeypatch.setattr(sys, "argv", test_args_inf)
    main()

    df_inf = pd.read_csv(out_inference)
    assert "D-5" in df_inf.columns
    assert "D+5" in df_inf.columns
    assert "D+6" not in df_inf.columns
    assert "D+20" not in df_inf.columns

    # 2. Without config-file (Training mode): all 26 columns preserved
    out_train = tmp_path / "schedule_train.csv"
    test_args_train = [
        "build_strategy_schedule_dataset.py",
        "--input-file", str(input_csv),
        "--output-file", str(out_train),
    ]
    monkeypatch.setattr(sys, "argv", test_args_train)
    main()

    df_train = pd.read_csv(out_train)
    for day in SCHEDULE_DAY_COLUMNS:
        assert day in df_train.columns


def test_generate_summary_workbooks_dynamic_columns(tmp_path: Path):
    from generate_prediction_summary import generate_summary_workbooks
    from pipeline_common import DAY_COLUMNS

    pred_data = {
        "MONTH": ["OCT-2026"],
        "APAC_NO": ["ACC_01"],
        "SOURCE_RISK": ["LOW"],
    }
    for day in DAY_COLUMNS:
        pred_data[day] = ["SMS-11AM-ENGLISH"]
    pred_df = pd.DataFrame(pred_data)

    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"allowed_day_offsets": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]}')

    # Test with config_file (dynamic columns D-5..D+5)
    counts_file, pct_file = generate_summary_workbooks(
        pred_df,
        tmp_path,
        "test_dynamic",
        config_file=cfg,
    )

    counts_xl = pd.read_excel(counts_file, sheet_name="Low Risk")
    assert "D-5" in counts_xl.columns
    assert "D+5" in counts_xl.columns
    assert "D+6" not in counts_xl.columns
    assert "D+20" not in counts_xl.columns

    pct_xl = pd.read_excel(pct_file, sheet_name="Low Risk")
    assert "D-5" in pct_xl.columns
    assert "D+5" in pct_xl.columns
    assert "D+6" not in pct_xl.columns
    assert "D+20" not in pct_xl.columns

    intensity_xl = pd.read_excel(counts_file, sheet_name="Communication Intensity", skiprows=1)
    assert "D-5" in intensity_xl.columns
    assert "D+5" in intensity_xl.columns
    assert "D+6" not in intensity_xl.columns
    assert "D+20" not in intensity_xl.columns


def test_build_prediction_summary_fallback_strategy_filtered(tmp_path: Path):
    from run_monthly_inference_pipeline import build_prediction_summary
    from pipeline_common import DAY_COLUMNS

    pred_file = tmp_path / "preds.csv"
    pd.DataFrame(
        [
            {
                "MONTH": "OCT-2026",
                "SOURCE_RISK": "LOW",
                "D-5": "SMS-11AM-ENGLISH",
                "D-1": "SMS-11AM-ENGLISH",
                "D+1": "SMS-11AM-ENGLISH",
                "D+2": "SMS-11AM-ENGLISH",
                "D+3": "SMS-11AM-ENGLISH",
            }
        ]
    ).to_csv(pred_file, index=False)

    sched_file = tmp_path / "sched.csv"
    sched_data = {
        "Loan_number": ["ACC_01"],
        "RISK": ["LOW"],
    }
    for day in DAY_COLUMNS:
        sched_data[day] = ["SMS-11AM-ENGLISH"]
    pd.DataFrame(sched_data).to_csv(sched_file, index=False)

    ten_days = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
    summary = build_prediction_summary(
        prediction_file=pred_file,
        source_month_label="SEP-2026",
        prediction_month_label="OCT-2026",
        model_name="catboost_3m",
        campaign_df=pd.DataFrame(),
        mapping_df=pd.DataFrame(),
        schedule_file=sched_file,
        campaign_days=ten_days,
    )

    fallback = summary["fallback_strategy"]
    assert "LOW_strategy" in fallback
    for day in ten_days:
        assert day in fallback["LOW_strategy"]
    assert "D+6" not in fallback["LOW_strategy"]
    assert "D+20" not in fallback["LOW_strategy"]


def test_prepare_probability_map_builds_x_pred_without_name_error(tmp_path: Path):
    from generate_prediction_evidence import _prepare_probability_map
    from sklearn.preprocessing import LabelEncoder
    from model_fallbacks import ConstantDayModel
    import joblib

    pred_rows = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_001",
                "MONTH": "OCT-2026",
                "SOURCE_RISK": "LOW",
                "D-5": "SMS-11AM-ENGLISH",
                "D+1": "SMS-11AM-ENGLISH",
            }
        ]
    )

    feat_file = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {
                "ENTITY_KEY": "ACC_001",
                "MONTH": "SEP-2026",
                "RISK": "LOW",
                "SMS_TOTAL_INTENSITY": 1.0,
            }
        ]
    ).to_csv(feat_file, index=False)

    sched_file = tmp_path / "schedules.csv"
    pd.DataFrame(
        [
            {
                "ENTITY_KEY": "ACC_001",
                "MONTH": "OCT-2026",
                "TARGET_MONTH": "OCT-2026",
                "TARGET_MONTH_PERIOD": "2026-10",
                "RISK": "LOW",
                "D-5": "SMS-11AM-ENGLISH",
                "D+1": "SMS-11AM-ENGLISH",
            }
        ]
    ).to_csv(sched_file, index=False)

    base_cases = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "apac_card_number": "ACC_001",
                "risk": "LOW",
                "vertical": "LAP",
                "collectable_amount": "1000",
                "emi_date": "10/09/2026",
            }
        ]
    ).to_csv(base_cases, index=False)

    le = LabelEncoder()
    le.fit(["SMS-11AM-ENGLISH"])
    models = {
        "D-5": ConstantDayModel(0, 1),
        "D+1": ConstantDayModel(0, 1),
    }
    label_encoders = {
        "D-5": le,
        "D+1": le,
    }
    bundle = {
        "models": models,
        "label_encoders": label_encoders,
        "feature_columns": ["SMS_TOTAL_INTENSITY", "RISK_LOW"],
        "history_window_months": 1,
    }
    model_file = tmp_path / "model.joblib"
    joblib.dump(bundle, model_file)

    prob_map = _prepare_probability_map(
        prediction_rows=pred_rows,
        feature_file=feat_file,
        schedule_file=sched_file,
        model_file=model_file,
        base_population_file=base_cases,
        prediction_source_month="SEP-2026",
    )
    assert ("ACC_001", "D-5") in prob_map
    assert ("ACC_001", "D+1") in prob_map


def test_parse_day_offset():
    from pipeline_common import parse_day_offset

    # Integers
    assert parse_day_offset(-5) == -5
    assert parse_day_offset(0) == 0
    assert parse_day_offset(5) == 5

    # Floats
    assert parse_day_offset(-5.0) == -5
    assert parse_day_offset(5.0) == 5

    # Strings
    assert parse_day_offset("-5") == -5
    assert parse_day_offset("5") == 5
    assert parse_day_offset("+5") == 5
    assert parse_day_offset("  -3  ") == -3
    assert parse_day_offset("D-5") == -5
    assert parse_day_offset("d-5") == -5
    assert parse_day_offset("D+5") == 5
    assert parse_day_offset("d+5") == 5
    assert parse_day_offset("D") == 0
    assert parse_day_offset("d") == 0

    # Invalid / empty
    assert parse_day_offset("") is None
    assert parse_day_offset("   ") is None
    assert parse_day_offset(None) is None
    assert parse_day_offset("invalid") is None
    assert parse_day_offset("D+abc") is None
    assert parse_day_offset("D-xyz") is None


def test_offsets_from_window_range():
    from pipeline_common import offsets_from_window_range

    # Symmetric window: -5 to 5 (excluding 0)
    assert offsets_from_window_range("-5", "5") == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    assert offsets_from_window_range("D-5", "D+5") == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    # Pre-due only: -5 to -1
    assert offsets_from_window_range(-5, -1) == [-5, -4, -3, -2, -1]
    assert offsets_from_window_range("D-5", "D-1") == [-5, -4, -3, -2, -1]

    # Post-due only: 1 to 5
    assert offsets_from_window_range(1, 5) == [1, 2, 3, 4, 5]
    assert offsets_from_window_range("D+1", "D+5") == [1, 2, 3, 4, 5]

    # Inverted bounds: handles 5 to -5 gracefully by swapping
    assert offsets_from_window_range(5, -5) == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    # Invalid bounds
    assert offsets_from_window_range(None, 5) is None
    assert offsets_from_window_range("-5", None) is None
    assert offsets_from_window_range("-5", "invalid") is None


class DummyCursorResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class DummyDbConn:
    def __init__(self, rows_by_table=None, default_rows=None, exception_on_table=None, execute_handler=None):
        self.rows_by_table = rows_by_table or {}
        self.default_rows = default_rows or []
        self.exception_on_table = exception_on_table or {}
        self.execute_handler = execute_handler
        self.rolled_back = False
        self.executed_queries = []

    def execute(self, query, params=None):
        q_str = str(query)
        self.executed_queries.append((q_str, params))
        if self.execute_handler:
            return self.execute_handler(q_str, params)
        for tbl, exc in self.exception_on_table.items():
            if tbl in q_str:
                raise exc
        for tbl, rows in self.rows_by_table.items():
            if tbl in q_str:
                return DummyCursorResult(rows)
        return DummyCursorResult(self.default_rows)

    def rollback(self):
        self.rolled_back = True


def test_resolve_prediction_window_from_db_success():
    from run_monthly_inference_pipeline import resolve_prediction_window_from_db

    conn = DummyDbConn(
        default_rows=[
            ("aiml.start.day", "-5"),
            ("aiml.end.day", "5"),
        ]
    )
    offsets = resolve_prediction_window_from_db(conn, source_schema="test_src", target_schema="test_tgt")
    assert offsets == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def test_resolve_prediction_window_from_db_predue_only():
    from run_monthly_inference_pipeline import resolve_prediction_window_from_db

    conn = DummyDbConn(
        default_rows=[
            ("aiml.start.day", "D-5"),
            ("aiml.end.day", "D-1"),
        ]
    )
    offsets = resolve_prediction_window_from_db(conn)
    assert offsets == [-5, -4, -3, -2, -1]


def test_resolve_prediction_window_from_db_partial_or_missing():
    from run_monthly_inference_pipeline import resolve_prediction_window_from_db

    # Only start day present
    conn_partial = DummyDbConn(default_rows=[("aiml.start.day", "-5")])
    assert resolve_prediction_window_from_db(conn_partial) is None

    # Empty rows
    conn_empty = DummyDbConn(default_rows=[])
    assert resolve_prediction_window_from_db(conn_empty) is None


def test_resolve_prediction_window_from_db_unparseable():
    from run_monthly_inference_pipeline import resolve_prediction_window_from_db

    conn = DummyDbConn(
        default_rows=[
            ("aiml.start.day", "NOT_A_DAY"),
            ("aiml.end.day", "5"),
        ]
    )
    assert resolve_prediction_window_from_db(conn) is None


def test_resolve_prediction_window_from_db_table_fallback():
    from psycopg import errors
    from run_monthly_inference_pipeline import resolve_prediction_window_from_db

    def handler(q_str, params):
        if "test_src" not in q_str:
            raise errors.UndefinedTable("Table not found")
        return DummyCursorResult([
            ("aiml.start.day", "-3"),
            ("aiml.end.day", "3"),
        ])

    conn = DummyDbConn(execute_handler=handler)
    offsets = resolve_prediction_window_from_db(conn, source_schema="test_src")
    assert offsets == [-3, -2, -1, 1, 2, 3]
    assert conn.rolled_back is True


def test_resolve_prediction_window_priorities(tmp_path: Path):
    from run_monthly_inference_pipeline import resolve_prediction_window

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text('{"allowed_day_offsets": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]}')

    db_conn = DummyDbConn(
        default_rows=[
            ("aiml.start.day", "-2"),
            ("aiml.end.day", "2"),
        ]
    )

    # Priority 1: CLI overrides both DB and config_file
    offsets, src = resolve_prediction_window(
        conn=db_conn,
        config_file=cfg_file,
        cli_allowed_offsets=[-1, 1],
    )
    assert offsets == [-1, 1]
    assert src == "cli"

    # Priority 2: DB overrides config_file when CLI is absent
    offsets, src = resolve_prediction_window(
        conn=db_conn,
        config_file=cfg_file,
        cli_allowed_offsets=None,
    )
    assert offsets == [-2, -1, 1, 2]
    assert src == "data_config"

    # Priority 3: config_file is used when DB returns None (missing keys)
    empty_db_conn = DummyDbConn(default_rows=[])
    offsets, src = resolve_prediction_window(
        conn=empty_db_conn,
        config_file=cfg_file,
        cli_allowed_offsets=None,
    )
    assert offsets == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    assert src == "config_file"

    # Priority 4: Default full range when neither DB nor config_file has allowed_day_offsets
    empty_cfg = tmp_path / "empty_cfg.json"
    empty_cfg.write_text('{}')
    offsets, src = resolve_prediction_window(
        conn=empty_db_conn,
        config_file=empty_cfg,
        cli_allowed_offsets=None,
    )
    assert offsets is None
    assert src == "default_full_range"


def test_build_strategy_schedule_dataset_cli_allowed_offsets(tmp_path: Path, monkeypatch):
    import sys
    from build_strategy_schedule_dataset import main as build_schedule_main

    input_csv = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "ENTITY_KEY": "ACC_01",
                "MONTH": "MAY-2026",
                "DAY": "D-5",
                "RISK": "LOW",
                "PREDICTED_STRATEGY": "SMS-11AM-ENGLISH",
            },
            {
                "ENTITY_KEY": "ACC_01",
                "MONTH": "MAY-2026",
                "DAY": "D+1",
                "RISK": "LOW",
                "PREDICTED_STRATEGY": "WH-2PM-ENGLISH",
            },
        ]
    ).to_csv(input_csv, index=False)

    out_csv = tmp_path / "schedule_out.csv"
    # CLI allowed offsets: only -5 and 1
    test_args = [
        "build_strategy_schedule_dataset.py",
        "--input-file", str(input_csv),
        "--output-file", str(out_csv),
        "--allowed-day-offsets", "-5", "1",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    build_schedule_main()

    df = pd.read_csv(out_csv)
    assert "D-5" in df.columns
    assert "D+1" in df.columns
    assert "D-1" not in df.columns
    assert "D+5" not in df.columns
    assert "D+20" not in df.columns


def test_identify_new_customer_fallbacks_cli_allowed_offsets(tmp_path: Path, monkeypatch):
    import sys
    from identify_new_customer_fallbacks import main as fallbacks_main

    cases_file = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "apacCardNumber": "NEW_ACC_01",
                "risk": "LOW",
                "vertical": "LAP",
                "emi_date": "05/05/2026",
            }
        ]
    ).to_csv(cases_file, index=False)

    sched_file = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "ENTITY_KEY": "HIST_01",
                "MONTH": "APR-2026",
                "RISK": "LOW",
                "D-5": "SMS-11AM-ENGLISH",
                "D-1": "SMS-11AM-ENGLISH",
                "D+1": "WH-12PM-ENGLISH",
                "D+5": "VOICE-11AM-ENGLISH",
            }
        ]
    ).to_csv(sched_file, index=False)

    pred_file = tmp_path / "pred.csv"
    pd.DataFrame(
        [
            {
                "Loan_number": "NEW_ACC_01",
                "MONTH": "MAY-2026",
                "SOURCE_RISK": "LOW",
                "VERTICAL": "LAP",
                "emi_date": "05/05/2026",
                "D-5": "-",
                "D-1": "-",
                "D+1": "-",
                "D+5": "-",
            }
        ]
    ).to_csv(pred_file, index=False)

    out_fallbacks = tmp_path / "fallbacks.csv"
    # CLI allowed offsets: only -5 and -1 (pre-due only)
    test_args = [
        "identify_new_customer_fallbacks.py",
        "--cases-file", str(cases_file),
        "--schedule-file", str(sched_file),
        "--output-file", str(out_fallbacks),
        "--prediction-file", str(pred_file),
        "--allowed-day-offsets", "-5", "-1",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    fallbacks_main()

    updated_pred = pd.read_csv(pred_file)
    assert updated_pred.iloc[0]["D-5"] != "-"
    assert updated_pred.iloc[0]["D-1"] != "-"
    assert updated_pred.iloc[0]["D+1"] == "-"
    assert updated_pred.iloc[0]["D+5"] == "-"


def test_monthly_inference_pipeline_dry_run_flag(monkeypatch):
    from run_monthly_inference_pipeline import parse_args

    # Without --dry-run
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_inference_pipeline.py",
            "--host", "localhost",
            "--dbname", "testdb",
            "--user", "testuser",
            "--password", "testpw",
            "--source-month", "2026-04",
            "--predict-month", "2026-05",
        ],
    )
    args = parse_args()
    assert hasattr(args, "dry_run")
    assert args.dry_run is False

    # With --dry-run
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_inference_pipeline.py",
            "--host", "localhost",
            "--dbname", "testdb",
            "--user", "testuser",
            "--password", "testpw",
            "--source-month", "2026-04",
            "--predict-month", "2026-05",
            "--dry-run",
        ],
    )
    args_dry = parse_args()
    assert args_dry.dry_run is True


def test_generate_summary_workbooks_inherits_dataframe_columns(tmp_path: Path):
    from generate_prediction_summary import generate_summary_workbooks

    days_d5_to_d10 = [f"D-{i}" for i in range(5, 0, -1)] + [f"D+{i}" for i in range(1, 11)]
    pred_data = {
        "MONTH": ["OCT-2026"],
        "APAC_NO": ["ACC_01"],
        "SOURCE_RISK": ["LOW"],
    }
    for day in days_d5_to_d10:
        pred_data[day] = ["SMS-11AM-ENGLISH"]
    pred_df = pd.DataFrame(pred_data)

    counts_file, pct_file = generate_summary_workbooks(
        pred_df,
        tmp_path,
        "test_inherit",
    )

    counts_xl = pd.read_excel(counts_file, sheet_name="Low Risk")
    for day in days_d5_to_d10:
        assert day in counts_xl.columns, f"{day} should be present in summary counts"
    assert "D+11" not in counts_xl.columns


def test_apply_non_working_day_suppressions_preserves_extended_window(tmp_path: Path, monkeypatch):
    import apply_non_working_day_suppressions

    days_d5_to_d10 = [f"D-{i}" for i in range(5, 0, -1)] + [f"D+{i}" for i in range(1, 11)]
    pred_data = {
        "MONTH": ["OCT-2026"],
        "APAC_NO": ["ACC_01"],
        "SOURCE_RISK": ["LOW"],
        "emi_date": ["10-10-2026"],
    }
    for day in days_d5_to_d10:
        pred_data[day] = ["SMS-11AM-ENGLISH"]
    pred_df = pd.DataFrame(pred_data)

    pred_csv = tmp_path / "test_extended_pred.csv"
    pred_df.to_csv(pred_csv, index=False)

    test_args = [
        "apply_non_working_day_suppressions.py",
        "--prediction-file", str(pred_csv),
        "--saturday-mode", "bank_saturdays",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    apply_non_working_day_suppressions.main()

    counts_file = tmp_path / "test_extended_pred_summary_counts.xlsx"
    assert counts_file.exists()
    counts_xl = pd.read_excel(counts_file, sheet_name="Low Risk")
    for day in days_d5_to_d10:
        assert day in counts_xl.columns, f"{day} should be preserved in summary counts post-suppression"
    assert "D+11" not in counts_xl.columns


def test_format_hour_label_and_parse_hour_to_int():
    assert format_hour_label(9) == "9AM"
    assert format_hour_label(12) == "12PM"
    assert format_hour_label(13) == "1PM"
    assert format_hour_label(15) == "3PM"
    assert format_hour_label(18) == "6PM"

    assert parse_hour_to_int("9AM") == 9
    assert parse_hour_to_int("10AM") == 10
    assert parse_hour_to_int("12PM") == 12
    assert parse_hour_to_int("1PM") == 13
    assert parse_hour_to_int("3PM") == 15
    assert parse_hour_to_int("6PM") == 18
    assert parse_hour_to_int("8-11") == 9
    assert parse_hour_to_int("12-3") == 12
    assert parse_hour_to_int("4-7") == 16
    assert parse_hour_to_int("14") == 14


def test_snap_strategy_to_window_step_3():
    # Window 9 to 18 with 3-hour steps -> candidate hours [9, 12, 15, 18]
    window = {"start_hour": 9, "end_hour": 18, "step_hours": 3}

    # 9AM stays 9AM
    assert snap_strategy_to_window("SMS-9AM-ENGLISH", window) == "SMS-9AM-ENGLISH"
    # 10AM snaps to 9AM
    assert snap_strategy_to_window("SMS-10AM-ENGLISH", window) == "SMS-9AM-ENGLISH"
    # 11AM snaps to 12PM
    assert snap_strategy_to_window("SMS-11AM-ENGLISH", window) == "SMS-12PM-ENGLISH"
    # 12PM stays 12PM
    assert snap_strategy_to_window("WH-12PM-HINDI", window) == "WH-12PM-HINDI"
    # 1PM (13) snaps to 12PM (12)
    assert snap_strategy_to_window("WH-1PM-HINDI", window) == "WH-12PM-HINDI"
    # 2PM (14) snaps to 3PM (15)
    assert snap_strategy_to_window("VOICE-2PM-MARATHI", window) == "VOICE-3PM-MARATHI"
    # 3PM (15) stays 3PM
    assert snap_strategy_to_window("VOICE-3PM-MARATHI", window) == "VOICE-3PM-MARATHI"
    # 4PM (16) snaps to 3PM (15)
    assert snap_strategy_to_window("VOICE_BOT-4PM-ENGLISH", window) == "VOICE_BOT-3PM-ENGLISH"
    # 5PM (17) snaps to 6PM (18)
    assert snap_strategy_to_window("VOICE_BOT-5PM-ENGLISH", window) == "VOICE_BOT-6PM-ENGLISH"
    # 6PM (18) stays 6PM
    assert snap_strategy_to_window("VOICE_BOT-6PM-ENGLISH", window) == "VOICE_BOT-6PM-ENGLISH"

    # Multi-mode pipe with - rank preservation
    raw_multi = "SMS-10AM-ENGLISH|WH-11AM-HINDI|-"
    assert snap_strategy_to_window(raw_multi, window) == "SMS-9AM-ENGLISH|WH-12PM-HINDI|-"

    # Deduplication if two tokens snap to the exact same strategy
    raw_dup = "SMS-10AM-ENGLISH|SMS-9AM-ENGLISH"
    assert snap_strategy_to_window(raw_dup, window) == "SMS-9AM-ENGLISH"

    # Blank / dash
    assert snap_strategy_to_window("-", window) == "-"
    assert snap_strategy_to_window("", window) == "-"
    assert snap_strategy_to_window(None, window) == "-"


def test_snap_prediction_dataframe_to_window():
    window = {"start_hour": 9, "end_hour": 18, "step_hours": 3}
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC_1",
                "D-5": "SMS-10AM-ENGLISH",
                "D-2": "WH-1PM-HINDI",
                "D+1": "VOICE-2PM-ENGLISH|SMS-10AM-ENGLISH",
            },
            {
                "Loan_number": "ACC_2",
                "D-5": "-",
                "D-2": "SMS-11AM-ENGLISH",
                "D+1": "-",
            },
        ]
    )

    snapped_df = snap_prediction_dataframe_to_window(df, window, day_columns=["D-5", "D-2", "D+1"])
    assert snapped_df.at[0, "D-5"] == "SMS-9AM-ENGLISH"
    assert snapped_df.at[0, "D-2"] == "WH-12PM-HINDI"
    assert snapped_df.at[0, "D+1"] == "VOICE-3PM-ENGLISH|SMS-9AM-ENGLISH"
    assert snapped_df.at[1, "D-5"] == "-"
    assert snapped_df.at[1, "D-2"] == "SMS-12PM-ENGLISH"
    assert snapped_df.at[1, "D+1"] == "-"


def test_predict_catboost_cli_step_hours_arg():
    from predict_next_month_strategy_catboost import parse_args

    test_args = [
        "predict_next_month_strategy_catboost.py",
        "--feature-file", "features.csv",
        "--schedule-file", "schedules.csv",
        "--model-file", "model.joblib",
        "--base-population-file", "cases.csv",
        "--prediction-file", "preds.csv",
        "--step-hours", "3",
    ]
    monkeypatch_argv = sys.argv
    try:
        sys.argv = test_args
        args = parse_args()
        assert args.step_hours == 3
    finally:
        sys.argv = monkeypatch_argv


def test_predict_catboost_snaps_final_prediction_and_leaves_raw_untouched(tmp_path: Path, monkeypatch):
    import json
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from model_fallbacks import ConstantDayModel
    from predict_next_month_strategy_catboost import main as predict_main

    le = LabelEncoder()
    le.fit(["-", "SMS-10AM-ENGLISH"])
    day_cols = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
    models = {d: ConstantDayModel(1, len(le.classes_)) for d in day_cols}
    encoders = {d: le for d in day_cols}

    bundle = {
        "models": models,
        "label_encoders": encoders,
        "feature_columns": ["SMS_TOTAL_INTENSITY_M1", "RISK_LOW"],
        "history_window_months": 1,
        "target_offset_months": 1,
    }
    model_file = tmp_path / "model.joblib"
    joblib.dump(bundle, model_file)

    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps({
        "allowed_day_offsets": [-5, -1, 1],
        "send_hour_window": {
            "start_hour": 9,
            "end_hour": 18,
            "step_hours": 3
        }
    }))

    feat_file = tmp_path / "features.csv"
    pd.DataFrame([
        {
            "ENTITY_KEY": "ACC_001",
            "MONTH": "SEP-2026",
            "RISK": "LOW",
            "VERTICAL": "LAP",
            "SMS_TOTAL_INTENSITY": 1,
            "SMS_TOTAL_INTENSITY_M1": 1,
            "RISK_LOW": 1,
        }
    ]).to_csv(feat_file, index=False)

    sched_file = tmp_path / "schedules.csv"
    pd.DataFrame([
        {
            "ENTITY_KEY": "ACC_001",
            "MONTH": "SEP-2026",
            "TARGET_MONTH": "OCT-2026",
            "TARGET_MONTH_PERIOD": "2026-10",
            "TARGET_RISK": "LOW",
            "DAY": "D-5",
            "PREDICTED_STRATEGY": "SMS-10AM-ENGLISH",
        }
    ]).to_csv(sched_file, index=False)

    base_cases = tmp_path / "cases.csv"
    pd.DataFrame([
        {
            "SOURCE_MONTH": "SEP-2026",
            "apac_card_number": "ACC_001",
            "risk": "LOW",
            "vertical": "LAP",
            "collectable_amount": "1000",
            "emi_date": "10/09/2026",
        }
    ]).to_csv(base_cases, index=False)

    pred_file = tmp_path / "predictions_oct.csv"

    argv = [
        "predict_next_month_strategy_catboost.py",
        "--feature-file", str(feat_file),
        "--schedule-file", str(sched_file),
        "--model-file", str(model_file),
        "--base-population-file", str(base_cases),
        "--prediction-file", str(pred_file),
        "--prediction-source-months", "SEP-2026",
        "--config-file", str(config_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    predict_main()

    raw_file = tmp_path / "predictions_oct_raw.csv"
    assert raw_file.exists()
    assert pred_file.exists()

    raw_df = pd.read_csv(raw_file, dtype=str)
    final_df = pd.read_csv(pred_file, dtype=str)

    # Raw file keeps 10AM (1-hr step from model)
    assert raw_df.iloc[0]["D-5"] == "SMS-10AM-ENGLISH"

    # Final file has snapped 9AM (3-hr step candidate)
    assert final_df.iloc[0]["D-5"] == "SMS-9AM-ENGLISH"

    # Reason in final file matches 9AM
    reason_json = json.loads(final_df.iloc[0]["PREDICTION_REASON"])
    assert "9AM" in reason_json["D-5"]


def test_identify_new_customer_fallbacks_snaps_hours_to_send_hour_window(tmp_path: Path, monkeypatch):
    import json
    from identify_new_customer_fallbacks import main as fallback_main

    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps({
        "allowed_day_offsets": [-5, -1, 1],
        "send_hour_window": {
            "start_hour": 9,
            "end_hour": 18,
            "step_hours": 3
        }
    }))

    pred_file = tmp_path / "preds.csv"
    pd.DataFrame([
        {
            "Loan_number": "ACC_NEW",
            "SOURCE_RISK": "LOW",
            "VERTICAL": "LAP",
            "MONTH": "OCT-2026",
            "SOURCE_MONTH_USED": "SEP-2026",
            "D-5": "",
            "D-1": "",
            "D+1": "",
            "PREDICTION_REASON": "{}",
            "IS_NEW_CUSTOMER": "True",
        }
    ]).to_csv(pred_file, index=False)

    cases_file = tmp_path / "cases.csv"
    pd.DataFrame([
        {
            "apacCardNumber": "ACC_NEW",
            "originalRisk": "LOW",
            "vertical": "LAP",
        }
    ]).to_csv(cases_file, index=False)

    sched_file = tmp_path / "sched.csv"
    pd.DataFrame([
        {
            "RISK": "LOW",
            "D-5": "SMS-10AM-ENGLISH",
            "D-1": "WH-1PM-HINDI",
            "D+1": "VOICE-4PM-ENGLISH",
        }
    ]).to_csv(sched_file, index=False)

    output_file = tmp_path / "cold_start_output.csv"

    argv = [
        "identify_new_customer_fallbacks.py",
        "--cases-file", str(cases_file),
        "--schedule-file", str(sched_file),
        "--output-file", str(output_file),
        "--prediction-file", str(pred_file),
        "--config-file", str(config_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    fallback_main()

    res_df = pd.read_csv(pred_file, dtype=str)
    # With step_hours=3, 10AM -> 9AM, 1PM -> 12PM, 4PM -> 3PM
    assert res_df.iloc[0]["D-5"] == "SMS-9AM-ENGLISH"
    assert res_df.iloc[0]["D-1"] == "WH-12PM-HINDI"
    assert res_df.iloc[0]["D+1"] == "SMS-9AM-ENGLISH|WH-12PM-ENGLISH|VOICE-3PM-ENGLISH"










