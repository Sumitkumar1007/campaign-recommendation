from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetch_month_from_postgres import (
    _extract_scheduler_emi_cycle,
    _extract_scheduler_emi_dates,
    configured_prediction_month_from_emi_dates,
    configured_source_month_from_emi_dates,
    emi_cycle_dates,
    month_bounds,
)
from generate_strategy_dataset import bucket_send_hour, candidate_hours as dataset_candidate_hours, process_chunk

from quartz_job_data import build_mcollect_job_data, serialize_quartz_job_data_map
from export_mcollect_scheduler import (
    DirectDatabaseMcollectPublisher,
    build_cron_trigger_specs,
    dataset_query_for,
    relative_campaign_day_to_date,
)
from run_monthly_inference_pipeline import (
    build_campaign_mappings,
    build_campaign_recommendations,
    month_label,
    resolve_active_campaign_vendors_by_mode,
    resolve_campaign_vendors,
    resolve_campaign_vendors_by_mode,
    selected_history_files,
    validate_month_pair,
)
from train_next_month_strategy_model_catboost import (
    analyze_day_target_variation,
    build_feature_matrix,
    build_rolling_feature_windows,
    fit_day_model,
    predict_top_k_by_risk,
    resolve_source_month_splits,
    validate_day_target_variation,
)
from pipeline_common import DAY_COLUMNS
from predict_next_month_strategy_catboost import build_prediction_population, build_prediction_reason
from campaign_recommendation.recommend import RecommendationPolicy, candidate_hours, generate_candidates


def test_validate_month_pair_accepts_adjacent_months() -> None:
    validate_month_pair("2026-04", "2026-05")
    assert month_label("2026-04") == "APR-2026"


def test_validate_month_pair_rejects_non_adjacent_months() -> None:
    with pytest.raises(ValueError, match="exactly one month after"):
        validate_month_pair("2026-04", "2026-06")


def test_validate_day_target_variation_returns_fallback_value_for_single_class_target() -> None:
    y_train = pd.DataFrame({"D+1": ["-", "-", "-"]})

    assert validate_day_target_variation("D+1", y_train) == "-"


def test_fit_day_model_uses_dash_fallback_for_missing_or_single_class_day(tmp_path: Path) -> None:
    X_train = pd.DataFrame({"feature_a": [1, 2, 3]})
    y_train = pd.DataFrame({"D+1": ["-", "-", "-"]})

    day, model, encoder = fit_day_model(
        "D+1",
        X_train,
        y_train,
        iterations=500,
        learning_rate=0.05,
        depth=8,
        checkpoint_dir=tmp_path,
        checkpoint_metadata={"train_rows": 3},
    )

    assert day == "D+1"
    assert encoder.classes_.tolist() == ["-"]
    output = predict_top_k_by_risk(model, encoder, X_train, pd.Series(["LOW", "MEDIUM", "HIGH"]))
    assert output.tolist() == ["-", "-", "-"]


def test_fit_day_model_uses_constant_non_dash_fallback_for_single_class_day(tmp_path: Path) -> None:
    X_train = pd.DataFrame({"feature_a": [1, 2, 3]})
    y_train = pd.DataFrame({"D-1": ["SMS-9AM-ENGLISH", "SMS-9AM-ENGLISH", "SMS-9AM-ENGLISH"]})

    day, model, encoder = fit_day_model(
        "D-1",
        X_train,
        y_train,
        iterations=500,
        learning_rate=0.05,
        depth=8,
        checkpoint_dir=tmp_path,
        checkpoint_metadata={"train_rows": 3},
    )

    assert day == "D-1"
    assert encoder.classes_.tolist() == ["SMS-9AM-ENGLISH"]
    output = predict_top_k_by_risk(model, encoder, X_train, pd.Series(["LOW", "MEDIUM", "HIGH"]))
    assert output.tolist() == ["SMS-9AM-ENGLISH", "SMS-9AM-ENGLISH", "SMS-9AM-ENGLISH"]


def test_analyze_day_target_variation_marks_missing_day_data() -> None:
    y_train = pd.DataFrame(index=[0, 1, 2])

    fallback = analyze_day_target_variation("D-2", y_train, original_day_present=False)

    assert fallback is not None
    assert fallback["fallback_reason"] == "missing_day_data"
    assert fallback["fallback_value"] == "-"


def test_resolve_source_month_splits_uses_latest_dataset_months_when_defaults_empty() -> None:
    dataset = pd.DataFrame(
        {
            "SOURCE_MONTH": ["JAN-2026", "FEB-2026", "MAR-2026", "APR-2026", "MAY-2026", "JUN-2026"],
            "TARGET_MONTH": ["FEB-2026", "MAR-2026", "APR-2026", "MAY-2026", "JUN-2026", pd.NA],
        }
    )

    train, validation, test, prediction = resolve_source_month_splits(
        dataset,
        train_source_months=[],
        validation_source_months=[],
        test_source_months=[],
        prediction_source_months=[],
    )

    assert train == ["JAN-2026", "FEB-2026", "MAR-2026", "APR-2026"]
    assert validation == ["MAY-2026"]
    assert test == []
    assert prediction == ["JUN-2026"]


def test_month_bounds_use_half_open_calendar_window() -> None:
    start, end = month_bounds("2026-02")
    assert str(start.date()) == "2026-02-01"
    assert str(end.date()) == "2026-03-01"


def test_extract_scheduler_emi_cycle_reads_multiple_days_from_config_dates() -> None:
    assert _extract_scheduler_emi_cycle(["05-06-2026", "10-06-2026", "05-07-2026"]) == [5, 10]


def test_extract_scheduler_emi_dates_reads_full_config_dates() -> None:
    dates = _extract_scheduler_emi_dates(["15/06/2026", "05/06/2026", "15/06/2026"])

    assert [date.strftime("%d/%m/%Y") for date in dates] == ["05/06/2026", "15/06/2026"]


def test_emi_cycle_dates_use_current_month_and_year() -> None:
    dates = emi_cycle_dates([5, 31], today=pd.Timestamp("2026-01-21"))

    assert [date.strftime("%d/%m/%Y") for date in dates] == ["05/01/2026", "31/01/2026"]


def test_configured_months_follow_full_emi_date_month() -> None:
    emi_dates = _extract_scheduler_emi_dates(["15/07/2026"])

    assert configured_prediction_month_from_emi_dates(emi_dates) == "2026-07"
    assert configured_source_month_from_emi_dates(emi_dates) == "2026-06"


def test_selected_history_files_uses_previous_two_months_plus_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    communication_dir = tmp_path / "data" / "communication"
    communication_dir.mkdir(parents=True)
    feb = communication_dir / "comm_data_FEB2026.csv"
    mar = communication_dir / "comm_data_MAR2026.csv"
    latest = communication_dir / "comm_data_APR2026.csv"
    for path in [feb, mar, latest]:
        path.write_text("id\n1\n", encoding="utf-8")

    monkeypatch.setattr("run_monthly_inference_pipeline.COMMUNICATION_DATA_DIR", communication_dir)

    assert selected_history_files("2026-04", latest) == [feb, mar, latest]


def test_selected_history_files_ignores_noncanonical_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    communication_dir = tmp_path / "data" / "communication"
    communication_dir.mkdir(parents=True)
    feb = communication_dir / "comm_data_FEB2026.csv"
    mar = communication_dir / "comm_data_MAR2026.csv"
    latest = communication_dir / "comm_data_APR2026.csv"
    noisy = communication_dir / "backup_MAR2026_comm_data.csv"
    for path in [feb, mar, latest, noisy]:
        path.write_text("id\n1\n", encoding="utf-8")

    monkeypatch.setattr("run_monthly_inference_pipeline.COMMUNICATION_DATA_DIR", communication_dir)

    assert selected_history_files("2026-04", latest) == [feb, mar, latest]


def test_rolling_feature_window_sums_latest_account_history() -> None:
    features = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1", "A1", "A1"],
            "MONTH": ["JAN-2026", "FEB-2026", "MAR-2026"],
            "SMS_TOTAL_INTENSITY": [1, 2, 4],
            "WH_TOTAL_INTENSITY": [0, 1, 1],
            "VOICE_TOTAL_INTENSITY": [0, 0, 0],
            "SMS_SUCCESS_9AM_ENGLISH": [1, 0, 2],
            "RISK": ["LOW", "LOW", "LOW"],
        }
    )

    rolled = build_rolling_feature_windows(features, history_window_months=2)
    march = rolled[rolled["MONTH"] == "MAR-2026"].iloc[0]

    assert march["SMS_TOTAL_INTENSITY"] == 6
    assert march["WH_TOTAL_INTENSITY"] == 2
    assert march["SMS_SUCCESS_9AM_ENGLISH"] == 2
    assert march["RISK"] == "LOW"


def test_process_chunk_uses_risk_from_communications_and_emi_month() -> None:
    chunk = pd.DataFrame(
        {
            "apac_card_number": ["A1"],
            "comm_status": ["DELIVERED"],
            "communication_type": ["SMS"],
            "verbiage_language": ["english"],
            "risk": ["high"],
            "emi_date": ["2026-05-05"],
            "date": ["2026-05-01"],
            "created_date": ["2026-04-30 09:15:00"],
        }
    )

    feature_counts, strategy_counts, risk_counts = process_chunk(chunk, month_source="emi_date")

    assert risk_counts.loc[0, "RISK"] == "HIGH"
    assert risk_counts.loc[0, "MONTH"] == "MAY-2026"
    assert feature_counts.loc[0, "MONTH"] == "MAY-2026"
    assert strategy_counts.loc[0, "feature"] == "SMS-9AM-ENGLISH"


def test_send_hour_rule_clamps_to_9_through_18() -> None:
    assert bucket_send_hour(8) == 9
    assert bucket_send_hour(9) == 9
    assert bucket_send_hour(17) == 17
    assert bucket_send_hour(18) == 18
    assert bucket_send_hour(23) == 18
    assert candidate_hours({"start_hour": 9, "end_hour": 18, "step_hours": 1}) == list(range(9, 19))
    assert candidate_hours({"start_hour": 9, "end_hour": 18, "step_hours": 2}) == [9, 11, 13, 15, 17, 18]
    assert dataset_candidate_hours({"start_hour": 9, "end_hour": 18, "step_hours": 2}) == [9, 11, 13, 15, 17, 18]
    assert bucket_send_hour(10, {"start_hour": 9, "end_hour": 18, "step_hours": 2}) == 9
    assert bucket_send_hour(23, {"start_hour": 9, "end_hour": 18, "step_hours": 2}) == 18


def test_generate_candidates_uses_hourly_business_window() -> None:
    base_population = pd.DataFrame(
        {
            "apac_card_number": ["A1"],
            "risk": ["LOW"],
            "collectable_amount": [100.0],
            "campaign_identifier": ["C1"],
            "emi_date": ["2026-05-05"],
        }
    )
    policy = RecommendationPolicy(
        allowed_day_offsets=[-1],
        risk_quota={"low": 1},
        send_hour_window={"start_hour": 9, "end_hour": 18, "step_hours": 1},
        channel_priority=["SMS", "WHATSAPP"],
    )

    candidates = generate_candidates(base_population, policy)

    assert candidates["send_hour"].tolist() == list(range(9, 19)) * 2
    assert candidates["communication_type"].tolist() == ["SMS"] * 10 + ["WHATSAPP"] * 10


def test_build_campaign_recommendations_groups_unique_scheduler_rows(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["05/04/2026"],
            "D-5": ["SMS-9AM-HINDI|WH-5PM-HINDI|IVR-2PM-HINDI"],
            "D-4": ["SMS-10AM-HINDI"],
            "D-3": ["-"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["SMS-9AM-HINDI"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
        }
    ).to_csv(prediction_file, index=False)

    output = build_campaign_recommendations(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech", "kaleyra"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
    )

    sms_pre_one = output[
        (output["name"] == "PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426_1")
    ].iloc[0]
    assert sms_pre_one["mode"] == "SMS"
    assert sms_pre_one["date"] == "D-5"
    assert sms_pre_one["time"] == "09:00:00"
    assert sms_pre_one["template_name"] == "PREDUE_AIML_SMS_HINDI"
    assert sms_pre_one["dataset_name"] == "PREDUE AIML SMS LAP HINDI HR EMI 5TH DM5 09"
    assert sms_pre_one["vendor"] == "prutech"
    assert sms_pre_one["active"] == "T"

    sms_pre_two = output[
        (output["name"] == "PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426_2")
    ].iloc[0]
    assert sms_pre_two["date"] == "D-4"
    assert sms_pre_two["time"] == "10:00:00"

    assert "POSTDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426" in set(output["name"])
    assert "PREDUE_AIML_WA_LAP_HINDI_HR_5TH_PRUTECH_080426" in set(output["name"])
    assert "PREDUE_AIML_VOICE_LAP_HINDI_HR_5TH_PRUTECH_080426" in set(output["name"])
    assert "PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_KALEYRA_080426_1" in set(output["name"])


def test_build_campaign_recommendations_groups_same_time_across_days(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions_same_time.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["05/04/2026"],
            "D-5": ["SMS-9AM-HINDI"],
            "D-4": ["-"],
            "D-3": ["SMS-9AM-HINDI"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["-"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
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
    )

    assert len(output) == 1
    row = output.iloc[0]
    assert row["name"] == "PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426"
    assert row["date"] == "D-5,D-3"
    assert row["time"] == "09:00:00"


def test_build_campaign_recommendations_groups_same_day_set_across_times(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions_same_day_set.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["05/04/2026"],
            "D-5": ["SMS-9AM-HINDI|SMS-10AM-HINDI"],
            "D-4": ["SMS-9AM-HINDI|SMS-10AM-HINDI"],
            "D-3": ["-"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["-"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
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
    )

    assert len(output) == 1
    row = output.iloc[0]
    assert row["name"] == "PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426"
    assert row["date"] == "D-5,D-4"
    assert row["time"] == "09:00:00,10:00:00"


def test_build_campaign_mappings_links_accounts_to_grouped_campaign_rows(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions_mapping.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH", "HIGH"],
            "Loan_number": ["L1", "L2"],
            "SOURCE_MONTH_USED": ["APR-2026", "APR-2026"],
            "MONTH": ["MAY-2026", "MAY-2026"],
            "EMI_DATE": ["05/04/2026", "05/04/2026"],
            "D-5": ["SMS-9AM-HINDI|SMS-10AM-HINDI", "SMS-9AM-HINDI|SMS-10AM-HINDI"],
            "D-4": ["SMS-9AM-HINDI|SMS-10AM-HINDI", "SMS-9AM-HINDI|SMS-10AM-HINDI"],
            "D-3": ["-", "-"],
            "D-2": ["-", "-"],
            "D-1": ["-", "-"],
            "D": ["-", "-"],
            "D+1": ["-", "-"],
            "D+2": ["-", "-"],
            "D+3": ["-", "-"],
            "D+4": ["-", "-"],
            "D+5": ["-", "-"],
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
    )

    assert len(mappings) == 2
    assert set(mappings["campaign_name"]) == {"PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426"}
    assert set(mappings["loan_number"]) == {"L1", "L2"}
    assert set(mappings["date"]) == {"D-5,D-4"}
    assert set(mappings["time"]) == {"09:00:00,10:00:00"}
    assert set(mappings["language"]) == {"HINDI"}



def test_build_campaign_mappings_includes_business_readable_reason(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions_mapping_reason.csv"
    reason = {"D-5": "SMS at 9AM in Hindi is recommended because SMS has shown positive response patterns and can be used as an early reminder."}
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["05/04/2026"],
            "D-5": ["SMS-9AM-HINDI"],
            "D-4": ["-"],
            "D-3": ["-"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["-"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
            "PREDICTION_REASON": [json.dumps(reason)],
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
    )

    assert "prediction_reason" in mappings.columns
    assert mappings.loc[0, "language"] == "HINDI"
    assert mappings.loc[0, "prediction_reason"].startswith("SMS at 9AM in Hindi is recommended")
    assert "early reminder" in mappings.loc[0, "prediction_reason"]


def test_build_campaign_recommendations_skips_regional_language(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["05/04/2026"],
            "D-5": ["SMS-9AM-REGIONAL|SMS-10AM-HINDI"],
            "D-4": ["-"],
            "D-3": ["-"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["-"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
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
    )

    assert "PREDUE_AIML_SMS_LAP_HINDI_HR_5TH_PRUTECH_080426" in set(output["name"])
    assert not any("REGIONAL" in name for name in output["name"])
    assert not any(output["template_name"].str.contains("REGIONAL", na=False))
    assert not any(output["dataset_name"].str.contains("REGIONAL", na=False))


def test_build_feature_matrix_removes_identifiers_and_one_hot_encodes() -> None:
    dataset = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1"],
            "SOURCE_MONTH": ["APR-2026"],
            "SOURCE_MONTH_PERIOD": [pd.Period("2026-04", freq="M")],
            "RISK": ["HIGH"],
            "SMS_TOTAL_INTENSITY": [3],
            "D-5": ["SMS-9AM-ENGLISH"],
            "TARGET_MONTH": [pd.NA],
        }
    )

    matrix = build_feature_matrix(dataset)

    assert "APAC_CARD_NUMBER" not in matrix.columns
    assert "SOURCE_MONTH_PERIOD" not in matrix.columns
    assert "SOURCE_MONTH_APR-2026" in matrix.columns
    assert "RISK_HIGH" in matrix.columns
    assert matrix.loc[0, "SMS_TOTAL_INTENSITY"] == 3


def test_resolve_campaign_vendors_reads_all_vendors_from_json_list() -> None:
    class DummyResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class DummyConn:
        def execute(self, query, params):
            return DummyResult(('["KALEYRA", "PRUTECH-CPASS", "prutech"]',))

    vendors = resolve_campaign_vendors(
        DummyConn(),
        source_schema="digital_collections",
        target_schema="digital_collections",
        fallback_vendor="prutech-cpass",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors == ["kaleyra", "prutech-cpass", "prutech"]


def test_resolve_campaign_vendors_normalizes_csv_vendor_variants() -> None:
    class DummyResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class DummyConn:
        def execute(self, query, params):
            return DummyResult(("prutech-cpass,prutech,kaleyra",))

    vendors = resolve_campaign_vendors(
        DummyConn(),
        source_schema="digital_collections",
        target_schema="digital_collections",
        fallback_vendor="prutech-cpass",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors == ["prutech-cpass", "prutech", "kaleyra"]


def test_resolve_campaign_vendors_by_mode_reads_service_specific_keys() -> None:
    class DummyResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class DummyConn:
        def execute(self, query, params):
            key = params[0]
            values = {
                "sms.service.vendor-list": ("kaleyra,prutech",),
                "voice.service.vendor-list": ("value-first,prutech-cpass",),
                "whatsapp.service.vendor-list": ("prutech-v2,kaleyra",),
            }
            return DummyResult(values.get(key))

    vendors = resolve_campaign_vendors_by_mode(
        DummyConn(),
        source_schema="digital_collections",
        target_schema="digital_collections",
        fallback_vendor="prutech-cpass",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors == {
        "SMS": ["kaleyra", "prutech"],
        "VOICE": ["value-first", "prutech-cpass"],
        "WHATSAPP": ["prutech-v2", "kaleyra"],
    }


def test_resolve_active_campaign_vendors_by_mode_reads_current_month_active_service() -> None:
    class DummyResult:
        def __init__(self, row=None, rows=None):
            self._row = row
            self._rows = rows or []

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class DummyConn:
        def execute(self, query, params):
            if len(params) == 2:
                return DummyResult(rows=[
                    ("SMS", "KALEYRA"),
                    ("VOICE", "PRUTECH-CPASS"),
                    ("WHATSAPP", "KALEYRA"),
                ])
            return DummyResult()

    vendors = resolve_active_campaign_vendors_by_mode(
        DummyConn(),
        source_schema="digital_collections",
        source_table="communications",
        source_month="2026-06",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors == {
        "SMS": ["kaleyra"],
        "VOICE": ["prutech-cpass"],
        "WHATSAPP": ["kaleyra"],
    }


def test_resolve_campaign_vendors_supports_active_service_json_values() -> None:
    class DummyResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class DummyConn:
        def execute(self, query, params):
            key = params[0]
            values = {
                "sms.service.vendor-list": ('[{"active_Service": "KALEYRA"}, {"active_Service": "PRUTECH-CPASS"}]',),
                "voice.service.vendor-list": (None,),
                "whatsapp.service.vendor-list": (None,),
            }
            return DummyResult(values.get(key))

    vendors = resolve_campaign_vendors_by_mode(
        DummyConn(),
        source_schema="digital_collections",
        target_schema="digital_collections",
        fallback_vendor="prutech-cpass",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors["SMS"] == ["kaleyra", "prutech-cpass"]


def test_resolve_campaign_vendors_by_mode_uses_active_service_then_data_config() -> None:
    class DummyResult:
        def __init__(self, row=None, rows=None):
            self._row = row
            self._rows = rows or []

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class DummyConn:
        def execute(self, query, params):
            if len(params) == 2:
                return DummyResult(rows=[
                    ("VOICE", "PRUTECH-CPASS"),
                    ("VOICE", "VALUE-FIRST"),
                ])
            key = params[0]
            values = {
                "sms.service.vendor-list": ("kaleyra,prutech",),
                "voice.service.vendor-list": ("value-first,prutech-cpass",),
                "whatsapp.service.vendor-list": ("prutech-v2,kaleyra",),
            }
            return DummyResult(row=values.get(key))

    vendors = resolve_campaign_vendors_by_mode(
        DummyConn(),
        source_schema="digital_collections",
        target_schema="digital_collections",
        source_table="communications",
        source_month="2026-06",
        fallback_vendor="prutech-cpass",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors["VOICE"] == ["prutech-cpass", "value-first"]


def test_resolve_campaign_vendors_by_mode_filters_to_active_vendors() -> None:
    class DummyResult:
        def __init__(self, row=None, rows=None):
            self._row = row
            self._rows = rows or []

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class DummyConn:
        def execute(self, query, params):
            if len(params) == 2:
                return DummyResult(rows=[
                    ("SMS", "KALEYRA"),
                    ("VOICE", "PRUTECH-CPASS"),
                ])
            key = params[0]
            values = {
                "sms.service.vendor-list": ("kaleyra,prutech",),
                "voice.service.vendor-list": ("value-first,prutech-cpass",),
                "whatsapp.service.vendor-list": ("prutech-v2,kaleyra",),
            }
            return DummyResult(row=values.get(key))

    vendors = resolve_campaign_vendors_by_mode(
        DummyConn(),
        source_schema="digital_collections",
        target_schema="digital_collections",
        source_table="communications",
        source_month="2026-06",
        fallback_vendor="prutech-cpass",
        logger=__import__("logging").getLogger("test_vendor"),
    )

    assert vendors == {
        "SMS": ["kaleyra"],
        "VOICE": ["prutech-cpass"],
        "WHATSAPP": ["prutech-v2", "kaleyra"],
    }


def test_build_campaign_recommendations_honors_empty_vendor_map(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions_empty_vendor_map.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["05/04/2026"],
            "D-5": ["SMS-9AM-HINDI"],
            "D-4": ["-"],
            "D-3": ["-"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["-"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
        }
    ).to_csv(prediction_file, index=False)

    campaigns, mappings = __import__("run_monthly_inference_pipeline")._prepare_campaign_outputs(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech"],
        vendor_map={"SMS": [], "VOICE": [], "WHATSAPP": []},
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
    )

    assert campaigns.empty
    assert mappings.empty


def test_predict_top_k_by_risk_uses_bucket_quota() -> None:
    class DummyModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.10, 0.65, 0.25],
                    [0.10, 0.25, 0.65],
                ]
            )

    class DummyEncoder:
        classes_ = np.array(["-", "SMS-9AM-ENGLISH", "WH-10AM-HINDI"])

    output = predict_top_k_by_risk(
        DummyModel(),
        DummyEncoder(),
        pd.DataFrame({"x": [1, 2, 3]}),
        pd.Series(["LOW", "MEDIUM", "HIGH"]),
    )

    assert output.tolist() == [
        "-",
        "SMS-9AM-ENGLISH|WH-10AM-HINDI",
        "WH-10AM-HINDI|SMS-9AM-ENGLISH|-",
    ]


def test_build_campaign_recommendations_derives_emi_cycle_from_emi_date(tmp_path: Path) -> None:
    prediction_file = tmp_path / "predictions_multi_cycle.csv"
    pd.DataFrame(
        {
            "SOURCE_RISK": ["HIGH"],
            "Loan_number": ["L1"],
            "SOURCE_MONTH_USED": ["APR-2026"],
            "MONTH": ["MAY-2026"],
            "EMI_DATE": ["10/04/2026"],
            "D-5": ["SMS-9AM-HINDI"],
            "D-4": ["-"],
            "D-3": ["-"],
            "D-2": ["-"],
            "D-1": ["-"],
            "D": ["-"],
            "D+1": ["-"],
            "D+2": ["-"],
            "D+3": ["-"],
            "D+4": ["-"],
            "D+5": ["-"],
        }
    ).to_csv(prediction_file, index=False)

    output = build_campaign_recommendations(
        prediction_file,
        source_month_label="APR-2026",
        prediction_month_label="MAY-2026",
        model_name="catboost_3m",
        emi_cycles=[5, 10, 12],
        vertical="LAP",
        vendors=["prutech"],
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
    )

    assert output.loc[0, "emi_cycle"] == 10
    assert "10TH" in output.loc[0, "name"]
    assert "EMI 10TH" in output.loc[0, "dataset_name"]


def test_build_prediction_population_keeps_same_account_multiple_emi_dates() -> None:
    dataset = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1"],
            "SOURCE_MONTH": ["APR-2026"],
            "SOURCE_MONTH_PERIOD": [pd.Period("2026-04", freq="M")],
            "RISK": ["HIGH"],
            "VERTICAL": ["LAP"],
            "SMS_TOTAL_INTENSITY": [3],
            "TARGET_MONTH": [pd.NA],
            "TARGET_MONTH_PERIOD": [pd.Period("2026-05", freq="M")],
            "TARGET_RISK": [pd.NA],
            "D-5": [pd.NA],
        }
    )
    base_population = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1", "A1"],
            "SOURCE_MONTH": ["APR-2026", "APR-2026"],
            "RISK": ["HIGH", "HIGH"],
            "VERTICAL": ["LAP", "LAP"],
            "collectable_amount": [100.0, 100.0],
            "emi_date": ["05/04/2026", "10/04/2026"],
        }
    )

    with_history, without_history = build_prediction_population(
        dataset=dataset,
        base_population=base_population,
        prediction_source_month="APR-2026",
    )

    assert without_history.empty
    assert sorted(with_history["emi_date"].tolist()) == ["05/04/2026", "10/04/2026"]


def test_build_prediction_population_keeps_base_accounts_without_history() -> None:
    dataset = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1"],
            "SOURCE_MONTH": ["APR-2026"],
            "SOURCE_MONTH_PERIOD": [pd.Period("2026-04", freq="M")],
            "RISK": ["HIGH"],
            "VERTICAL": ["LAP"],
            "SMS_TOTAL_INTENSITY": [3],
            "TARGET_MONTH": [pd.NA],
            "TARGET_MONTH_PERIOD": [pd.Period("2026-05", freq="M")],
            "TARGET_RISK": [pd.NA],
            "D-5": [pd.NA],
        }
    )
    base_population = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1", "A2"],
            "SOURCE_MONTH": ["APR-2026", "APR-2026"],
            "RISK": ["HIGH", "LOW"],
            "VERTICAL": ["LAP", "BUSINESS LOAN"],
            "collectable_amount": [100.0, 50.0],
            "emi_date": ["05/04/2026", "05/04/2026"],
        }
    )

    with_history, without_history = build_prediction_population(
        dataset=dataset,
        base_population=base_population,
        prediction_source_month="APR-2026",
    )

    assert with_history["APAC_CARD_NUMBER"].tolist() == ["A1"]
    assert without_history["APAC_CARD_NUMBER"].tolist() == ["A2"]


def test_build_prediction_population_uses_latest_prior_history_for_source_month() -> None:
    dataset = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1", "A1"],
            "SOURCE_MONTH": ["FEB-2026", "MAR-2026"],
            "SOURCE_MONTH_PERIOD": [pd.Period("2026-02", freq="M"), pd.Period("2026-03", freq="M")],
            "RISK": ["HIGH", "HIGH"],
            "VERTICAL": ["LAP", "LAP"],
            "SMS_TOTAL_INTENSITY": [2, 7],
            "TARGET_MONTH": [pd.NA, pd.NA],
            "TARGET_MONTH_PERIOD": [pd.Period("2026-03", freq="M"), pd.Period("2026-04", freq="M")],
            "TARGET_RISK": [pd.NA, pd.NA],
            "D-5": [pd.NA, pd.NA],
        }
    )
    base_population = pd.DataFrame(
        {
            "APAC_CARD_NUMBER": ["A1"],
            "SOURCE_MONTH": ["APR-2026"],
            "RISK": ["HIGH"],
            "VERTICAL": ["LAP"],
            "collectable_amount": [100.0],
            "emi_date": ["05/04/2026"],
        }
    )

    with_history, without_history = build_prediction_population(
        dataset=dataset,
        base_population=base_population,
        prediction_source_month="APR-2026",
    )

    assert without_history.empty
    assert with_history["APAC_CARD_NUMBER"].tolist() == ["A1"]
    assert with_history.loc[0, "SMS_TOTAL_INTENSITY"] == 7
    assert with_history.loc[0, "SOURCE_MONTH"] == "APR-2026"
    assert with_history.loc[0, "SOURCE_MONTH_PERIOD"] == pd.Period("2026-04", freq="M")


def test_build_prediction_reason_explains_matching_success_signal() -> None:
    prediction_row = pd.Series(
        {
            "SOURCE_RISK": "HIGH",
            "D-5": "SMS-9AM-HINDI|WH-5PM-HINDI|-",
            "D-4": "-",
        }
    )
    source_row = pd.Series(
        {
            "RISK": "HIGH",
            "SMS_TOTAL_INTENSITY": 7,
            "WH_TOTAL_INTENSITY": 3,
            "VOICE_TOTAL_INTENSITY": 1,
            "SMS_SUCCESS_9AM_HINDI": 4,
            "WH_SUCCESS_5PM_HINDI": 2,
            "VOICE_FAILED_HINDI": 1,
        }
    )

    reason = json.loads(
        build_prediction_reason(
            prediction_row=prediction_row,
            source_row=source_row,
        )
    )

    assert reason["D-5"] == (
        "SMS at 9AM in Hindi is recommended as an early EMI reminder. "
        "This is a light-touch communication before the due date and is suitable for starting the follow-up journey."
    )



def test_build_prediction_reason_respects_no_campaign_primary_rank() -> None:
    prediction_row = pd.Series(
        {
            "SOURCE_RISK": "MEDIUM",
            "D+2": "-|SMS-8AM-REGIONAL",
        }
    )
    source_row = pd.Series(
        {
            "RISK": "MEDIUM",
            "SMS_TOTAL_INTENSITY": 3,
            "WH_TOTAL_INTENSITY": 1,
            "VOICE_TOTAL_INTENSITY": 0,
        }
    )

    reason = json.loads(build_prediction_reason(prediction_row=prediction_row, source_row=source_row))

    assert reason["D+2"] == (
        "At this stage, no campaign is recommended to prevent excessive communication with the customer. "
        "If additional follow-up is required, a Regional language SMS may be sent at 8:00 AM as the next course of action."
    )


def test_build_prediction_reason_uses_business_friendly_d_plus_5_language() -> None:
    prediction_row = pd.Series(
        {
            "SOURCE_RISK": "MEDIUM",
            "D+5": "-|SMS-3PM-ENGLISH",
        }
    )
    source_row = pd.Series(
        {
            "RISK": "MEDIUM",
            "SMS_TOTAL_INTENSITY": 2,
            "WH_TOTAL_INTENSITY": 0,
            "VOICE_TOTAL_INTENSITY": 0,
        }
    )

    reason = json.loads(build_prediction_reason(prediction_row=prediction_row, source_row=source_row))

    assert reason["D+5"] == (
        "At this stage, no campaign is recommended to prevent excessive communication with the customer. "
        "If the account still requires follow-up five days after the Cycle date, an English SMS may be sent at 3:00 PM as the next course of action."
    )


def test_build_prediction_reason_explains_missing_day_fallback() -> None:
    prediction_row = pd.Series({"SOURCE_RISK": "LOW", "D-1": "-"})

    reason = json.loads(
        build_prediction_reason(
            prediction_row=prediction_row,
            source_row=None,
            day_fallback_config={"D-1": {"fallback_reason": "missing_day_data", "fallback_value": "-"}},
        )
    )

    assert reason["D-1"] == (
        "No campaign is recommended because historical training data was not available for this day, so the system has defaulted to no communication."
    )


def test_build_prediction_reason_explains_single_unique_value_fallback() -> None:
    prediction_row = pd.Series({"SOURCE_RISK": "LOW", "D+2": "SMS-9AM-ENGLISH"})

    reason = json.loads(
        build_prediction_reason(
            prediction_row=prediction_row,
            source_row=None,
            day_fallback_config={"D+2": {"fallback_reason": "single_unique_value", "fallback_value": "SMS-9AM-ENGLISH"}},
        )
    )

    assert reason["D+2"] == (
        "SMS at 9AM in English is recommended because historical training data for this day contained only one unique outcome, so the system has applied that same recommendation consistently."
    )


def test_build_prediction_reason_explains_no_history_blank_predictions() -> None:
    prediction_row = pd.Series({"SOURCE_RISK": "LOW", "D-5": "-"})

    reason = json.loads(build_prediction_reason(prediction_row=prediction_row, source_row=None))

    assert reason["D-5"] == (
        "At this stage, no campaign is recommended to prevent excessive communication with the customer."
    )
    assert set(reason) == set(DAY_COLUMNS)


def test_dataset_query_for_uses_mapping_table_filters() -> None:
    row = pd.Series({
        "mode": "WHATSAPP",
        "vertical": "LAP",
        "template_name": "POSTDUE_AIML_WHATSAPP_TELUGU",
        "risk": "MR",
        "emi_cycle": 15,
        "date": "D-3",
        "time": "09:00:00",
        "source_month": "JUN-2026",
        "prediction_month": "JUL-2026",
    })

    query = dataset_query_for(
        row,
        schema="digital_collections",
        campaign_table="ai_ml_campaign_recommendations",
        mapping_table="ai_ml_campaign_mapping",
    )

    assert query == (
        "select distinct dc.* "
        "from digital_collections.digital_cases dc "
        "join digital_collections.ai_ml_campaign_mapping amcm on dc.apac_card_number = amcm.loan_number "
        "where amcm.mode = 'WHATSAPP' "
        "and amcm.vertical = 'LAP' "
        "and amcm.language = 'TELUGU' "
        "and amcm.risk = 'MR' "
        "and amcm.emi_cycle = 15 "
        "and amcm.date = 'D-3' "
        "and amcm.time = '09:00:00' "
        "and amcm.source_month = 'JUN-2026' "
        "and amcm.prediction_month = 'JUL-2026' "
        "AND dc.apac_card_number NOT IN ( "
        "SELECT dnd.apac_card_number FROM digital_collections.dnd_digital_cases dnd"
        " ) "
        "AND dc.apac_card_number NOT IN ( "
        "SELECT p.apac_card_number "
        "FROM payment p "
        "WHERE p.apac_card_number = dc.apac_card_number "
        "AND p.status = 'SUCCESS' "
        "AND p.payment_datetime::date BETWEEN ( "
        "to_date(dc.emi_date, 'DD/MM/YYYY') - ( "
        "SELECT value::integer FROM data_config WHERE key_name = 'payment_start_date_range' "
        ") * interval '1 day' "
        ") AND ( "
        "to_date(dc.emi_date, 'DD/MM/YYYY') + ( "
        "SELECT value::integer FROM data_config WHERE key_name = 'payment_end_date_range' "
        ") * interval '1 day' "
        ") "
        ")"
    )


class _RecordingCursor:
    def __init__(self, log: list[tuple[str, str]]) -> None:
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        self.log.append(("execute", str(query)))

    def executemany(self, query, params_seq) -> None:
        self.log.append(("executemany", str(query)))


class _RecordingConn:
    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def execute(self, query, params=None) -> None:
        self.log.append(("execute", str(query)))

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.log)


def test_mcollect_publish_skips_digital_rules_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    campaigns = pd.DataFrame(
        [
            {
                "name": "camp-1",
                "mode": "SMS",
                "date": "D+1",
                "time": "16:00:00",
                "template_name": "POSTDUE_AIML_SMS_ENGLISH",
                "dataset_name": "dataset-1",
                "vendor": "kaleyra",
                "vertical": "LAP",
                "risk": "MR",
                "emi_cycle": 5,
                "source_month": "APR-2026",
                "prediction_month": "MAY-2026",
                "model_name": "catboost_3m",
                "active": "T",
                "campaign_type": "POSTDUE",
                "due_type": "POSTDUE",
            }
        ]
    )
    conn = _RecordingConn()
    publisher = DirectDatabaseMcollectPublisher(
        conn,
        schema="digital_collections",
        campaign_table="ai_ml_campaign_recommendations",
        mapping_table="ai_ml_campaign_mapping",
        trigger_state="PAUSED",
    )
    monkeypatch.setattr(publisher, "_replace_quartz_jobs", lambda campaigns: 2)

    summary = publisher.publish(campaigns)

    assert summary.templates == 1
    assert all("digital_rules" not in query.lower() for _, query in conn.log)



def test_serialize_quartz_job_data_map_matches_existing_sms_job_data() -> None:
    expected_hex = (
        "aced0005737200156f72672e71756172747a2e4a6f62446174614d61709fb083e8bfa9b0cb020000787200"
        "266f72672e71756172747a2e7574696c732e537472696e674b65794469727479466c61674d61708208e8"
        "c3fbc55d280200015a0013616c6c6f77735472616e7369656e74446174617872001d6f72672e717561"
        "72747a2e7574696c732e4469727479466c61674d617013e62ead28760ace0200025a000564697274794c"
        "00036d617074000f4c6a6176612f7574696c2f4d61703b787001737200116a6176612e7574696c2e"
        "486173684d61700507dac1c31660d103000246000a6c6f6164466163746f724900097468726573686f"
        "6c6478703f4000000000000c770800000010000000027400084461746173657473740012504f535444"
        "5545204d52485020515545525974000854656d706c61746574000c504f535444554520534d53357800"
    )

    actual = serialize_quartz_job_data_map(
        [("Datasets", "POSTDUE MRHP QUERY"), ("Template", "POSTDUE SMS5")]
    )

    assert actual.hex() == expected_hex


def test_build_mcollect_job_data_matches_existing_voice_bot_vendor_job_data() -> None:
    expected_hex = (
        "aced0005737200156f72672e71756172747a2e4a6f62446174614d61709fb083e8bfa9b0cb020000787200"
        "266f72672e71756172747a2e7574696c732e537472696e674b65794469727479466c61674d61708208e8"
        "c3fbc55d280200015a0013616c6c6f77735472616e7369656e74446174617872001d6f72672e717561"
        "72747a2e7574696c732e4469727479466c61674d617013e62ead28760ace0200025a000564697274794c"
        "00036d617074000f4c6a6176612f7574696c2f4d61703b787001737200116a6176612e7574696c2e"
        "486173684d61700507dac1c31660d103000246000a6c6f6164466163746f724900097468726573686f"
        "6c6478703f4000000000000c770800000010000000037400084461746173657473740020504f535444"
        "554520414c4c5249534b20454e474c49534820564220515545525974000656656e646f727400097665"
        "7262616c797a6574000854656d706c61746574001456455242414c595a4520454e474c495348205642"
        "7800"
    )

    actual = build_mcollect_job_data(
        dataset_name="POSTDUE ALLRISK ENGLISH VB QUERY",
        vendor="verbalyze",
        template_name="VERBALYZE ENGLISH VB",
    )

    assert actual.hex() == expected_hex


def test_mcollect_relative_campaign_day_to_date_handles_previous_month() -> None:
    assert relative_campaign_day_to_date("D-5", "MAY-2026", 5).strftime("%Y-%m-%d") == "2026-04-30"
    assert relative_campaign_day_to_date("D-1", "MAY-2026", 5).strftime("%Y-%m-%d") == "2026-05-04"
    assert relative_campaign_day_to_date("D+5", "MAY-2026", 5).strftime("%Y-%m-%d") == "2026-05-10"


def test_build_cron_trigger_specs_splits_month_boundaries() -> None:
    row = pd.Series(
        {
            "name": "TEST",
            "date": "D-5,D-4",
            "time": "09:00:00,10:00:00",
            "prediction_month": "MAY-2026",
            "emi_cycle": 5,
        }
    )

    specs = build_cron_trigger_specs(row, trigger_state="PAUSED")

    assert [spec.cron_expression for spec in specs] == ["0 0 9,10 30 4 ?", "0 0 9,10 1 5 ?"]
    assert {spec.trigger_state for spec in specs} == {"PAUSED"}
