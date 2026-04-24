from __future__ import annotations

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

from fetch_month_from_postgres import emi_cycle_dates, month_bounds, resolve_emi_cycle
from generate_strategy_dataset import bucket_send_hour, candidate_hours as dataset_candidate_hours, process_chunk
from run_monthly_inference_pipeline import (
    build_campaign_recommendations,
    month_label,
    selected_history_files,
    validate_month_pair,
)
from train_next_month_strategy_model_catboost import (
    build_feature_matrix,
    build_rolling_feature_windows,
    predict_top_k_by_risk,
)
from campaign_recommendation.recommend import RecommendationPolicy, candidate_hours, generate_candidates


def test_validate_month_pair_accepts_adjacent_months() -> None:
    validate_month_pair("2026-04", "2026-05")
    assert month_label("2026-04") == "APR-2026"


def test_validate_month_pair_rejects_non_adjacent_months() -> None:
    with pytest.raises(ValueError, match="exactly one month after"):
        validate_month_pair("2026-04", "2026-06")


def test_month_bounds_use_half_open_calendar_window() -> None:
    start, end = month_bounds("2026-02")
    assert str(start.date()) == "2026-02-01"
    assert str(end.date()) == "2026-03-01"


def test_resolve_emi_cycle_reads_multiple_days_from_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"emi_cycle": ["5", "10", 5]}', encoding="utf-8")

    assert resolve_emi_cycle(str(config_file), "") == [5, 10]


def test_emi_cycle_dates_use_current_month_and_year() -> None:
    dates = emi_cycle_dates([5, 31], today=pd.Timestamp("2026-01-21"))

    assert [date.strftime("%d/%m/%Y") for date in dates] == ["05/01/2026", "31/01/2026"]


def test_selected_history_files_uses_previous_two_months_plus_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    communication_dir = tmp_path / "data" / "communication" / "MFL_COMMUNICATION_DATA"
    communication_dir.mkdir(parents=True)
    feb = communication_dir / "mfl_recomm_model_FEB2026_comm_data.csv"
    mar = communication_dir / "mfl_recomm_model_MAR2026_comm_data.csv"
    latest = communication_dir / "latest_APR2026_comm_data.csv"
    for path in [feb, mar, latest]:
        path.write_text("id\n1\n", encoding="utf-8")

    monkeypatch.setattr("run_monthly_inference_pipeline.COMMUNICATION_DATA_DIR", communication_dir)

    assert selected_history_files("2026-04", latest) == [feb, mar, latest]


def test_selected_history_files_ignores_noncanonical_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    communication_dir = tmp_path / "data" / "communication" / "MFL_COMMUNICATION_DATA"
    communication_dir.mkdir(parents=True)
    feb = communication_dir / "mfl_recomm_model_FEB2026_comm_data.csv"
    mar = communication_dir / "mfl_recomm_model_MAR2026_comm_data.csv"
    latest = communication_dir / "latest_APR2026_comm_data.csv"
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
        emi_cycle=5,
        vertical="LAP",
        vendor="prutech-cpass",
        run_date=pd.Timestamp("2026-04-08").to_pydatetime(),
    )

    sms_pre = output[
        (output["name"] == "PRE_AIML_NORMAL_SMS_LAP_HINDI_5TH_PRUTECH_CPASS_HR_080426")
    ].iloc[0]
    assert sms_pre["mode"] == "SMS"
    assert sms_pre["date"] == "D-5,D-4,D-3,D-2,D-1"
    assert sms_pre["time"] == "09:00:00,10:00:00"
    assert sms_pre["template_name"] == "PREDUE AIML SMS LAP HINDI"
    assert sms_pre["dataset_name"] == "PREDUE AIML HINDI HR SMS LAP NORMAL FOR EMI 5TH"
    assert sms_pre["vendor"] == "prutech-cpass"
    assert sms_pre["active"] == "T"

    assert "POST_AIML_NORMAL_SMS_LAP_HINDI_5TH_PRUTECH_CPASS_HR_080426" in set(output["name"])
    assert "PRE_AIML_NORMAL_WA_LAP_HINDI_5TH_PRUTECH_CPASS_HR_080426" in set(output["name"])
    assert "PRE_AIML_NORMAL_IVR_LAP_HINDI_5TH_PRUTECH_CPASS_HR_080426" in set(output["name"])


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
