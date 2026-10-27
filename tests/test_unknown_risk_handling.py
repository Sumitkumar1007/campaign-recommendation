import sys
from pathlib import Path
import pytest
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_strategy_dataset import normalize_risk as gen_normalize_risk, process_chunk, summarize_chunk_audit
from build_strategy_schedule_dataset import normalize_risk as sched_normalize_risk
from identify_new_customer_fallbacks import build_fallback_rows
from pipeline_common import apply_predue_risk_rules, build_feature_matrix
from run_monthly_inference_pipeline import _prepare_campaign_outputs, _mapping_prediction_reason


def test_normalize_risk_handles_all_null_representations():
    s = pd.Series([None, "", "   ", "nan", "NAN", "None", "NULL", "null", "<NA>", "Low", "medium", "HIGH"])
    norm = gen_normalize_risk(s)
    expected = [
        "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN",
        "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN",
        "LOW", "MEDIUM", "HIGH"
    ]
    assert norm.tolist() == expected

    norm_sched = sched_normalize_risk(s)
    assert norm_sched.tolist() == expected


def test_process_chunk_and_audit_retain_unknown_risk():
    raw_df = pd.DataFrame([
        {
            "apac_card_number": "CARD_001",
            "risk": None,
            "communication_type": "SMS",
            "comm_status": "DELIVERED",
            "emi_date": "2026-09-05",
            "date": "2026-09-02",  # D-3
            "created_date": "2026-09-02 11:00:00",
            "verbiage_language": "ENGLISH",
        },
        {
            "apac_card_number": "CARD_002",
            "risk": "LOW",
            "communication_type": "WHATSAPP",
            "comm_status": "READ",
            "emi_date": "2026-09-05",
            "date": "2026-09-03",  # D-2
            "created_date": "2026-09-03 13:00:00",
            "verbiage_language": "ENGLISH",
        },
    ])
    audit = summarize_chunk_audit(raw_df)
    assert audit["total_rows"] == 2
    assert audit["unknown_risk_rows"] == 1
    assert audit["kept_model_rows"] == 2

    feature_counts, strategy_counts, risk_counts = process_chunk(
        raw_df,
        month_source="emi_date",
        allowed_day_offsets=[-5, -4, -3, -2, -1, 1, 2, 3],
    )
    assert len(risk_counts) == 2
    cards = risk_counts["ENTITY_KEY"].tolist()
    assert "CARD_001" in cards
    assert "CARD_002" in cards
    assert risk_counts.loc[risk_counts["ENTITY_KEY"] == "CARD_001", "RISK"].iloc[0] == "UNKNOWN"
    assert risk_counts.loc[risk_counts["ENTITY_KEY"] == "CARD_002", "RISK"].iloc[0] == "LOW"


def test_schedule_and_monthly_features_prioritize_known_risk(tmp_path):
    from build_strategy_schedule_dataset import main as build_sched_main
    from build_monthly_feature_dataset import main as build_monthly_main
    import sys

    # CARD_001 has 1 UNKNOWN and 1 LOW
    # CARD_002 has only UNKNOWN
    day_df = pd.DataFrame([
        {
            "ENTITY_KEY": "CARD_001",
            "APAC_CARD_NUMBER": "CARD_001",
            "MONTH": "SEP-2026",
            "DAY": "D-3",
            "RISK": "UNKNOWN",
            "VERTICAL": "LAP",
            "PREDICTED_STRATEGY": "SMS-11AM-ENGLISH",
            "SMS_TOTAL_INTENSITY": 1,
            "WH_TOTAL_INTENSITY": 0,
        },
        {
            "ENTITY_KEY": "CARD_001",
            "APAC_CARD_NUMBER": "CARD_001",
            "MONTH": "SEP-2026",
            "DAY": "D-2",
            "RISK": "LOW",
            "VERTICAL": "LAP",
            "PREDICTED_STRATEGY": "WH-1PM-ENGLISH",
            "SMS_TOTAL_INTENSITY": 0,
            "WH_TOTAL_INTENSITY": 1,
        },
        {
            "ENTITY_KEY": "CARD_002",
            "APAC_CARD_NUMBER": "CARD_002",
            "MONTH": "SEP-2026",
            "DAY": "D-1",
            "RISK": "UNKNOWN",
            "VERTICAL": "LAP",
            "PREDICTED_STRATEGY": "SMS-10AM-ENGLISH",
            "SMS_TOTAL_INTENSITY": 1,
            "WH_TOTAL_INTENSITY": 0,
        },
    ])
    input_file = tmp_path / "day_data.csv"
    day_df.to_csv(input_file, index=False)

    sched_out = tmp_path / "schedules.csv"
    sys.argv = ["build_strategy_schedule_dataset.py", "--input-file", str(input_file), "--output-file", str(sched_out)]
    build_sched_main()
    sched_res = pd.read_csv(sched_out)
    assert len(sched_res) == 2
    # CARD_001 should resolve to LOW (known prioritized)
    c1_risk = sched_res.loc[sched_res["ENTITY_KEY"] == "CARD_001", "RISK"].iloc[0]
    assert c1_risk == "LOW"
    # CARD_002 should resolve to UNKNOWN (not dropped!)
    c2_risk = sched_res.loc[sched_res["ENTITY_KEY"] == "CARD_002", "RISK"].iloc[0]
    assert c2_risk == "UNKNOWN"

    feat_out = tmp_path / "monthly_features.csv"
    sys.argv = ["build_monthly_feature_dataset.py", "--input-file", str(input_file), "--output-file", str(feat_out)]
    build_monthly_main()
    feat_res = pd.read_csv(feat_out)
    assert len(feat_res) == 2
    f1_risk = feat_res.loc[feat_res["ENTITY_KEY"] == "CARD_001", "RISK"].iloc[0]
    assert f1_risk == "LOW"
    f2_risk = feat_res.loc[feat_res["ENTITY_KEY"] == "CARD_002", "RISK"].iloc[0]
    assert f2_risk == "UNKNOWN"


def test_identify_new_customer_fallbacks_uses_low_risk_for_unknown():
    new_cases = pd.DataFrame([
        {
            "apacCardNumber": "NEW_001",
            "originalRisk": "UNKNOWN",
            "vertical": "LAP",
        }
    ])
    fallback_by_risk = {
        "LOW": {"D-5": "SMS-11AM-ENGLISH", "D-1": "WH-1PM-ENGLISH"},
        "HIGH": {"D-5": "SMS-10AM-ENGLISH", "D-4": "VOICE-11AM-ENGLISH"},
    }
    records = build_fallback_rows(new_cases, fallback_by_risk, allowed_day_labels={"D-5", "D-1"})
    rec_d5 = records.loc[(records["apacCardNumber"] == "NEW_001") & (records["day"] == "D-5")].iloc[0]
    assert rec_d5["recommendedStrategy"] == "SMS-11AM-ENGLISH"
    assert rec_d5["fallbackRiskUsed"] == "LOW"
    assert rec_d5["originalRisk"] == "UNKNOWN"
    assert "LOW-risk historical average" in rec_d5["reason"]


def test_apply_predue_risk_rules_caps_unknown_as_low():
    df = pd.DataFrame([
        {
            "SOURCE_RISK": "UNKNOWN",
            "D-5": "SMS-11AM-ENGLISH",
            "D-4": "WH-1PM-ENGLISH",
            "D-3": "VOICE-4PM-ENGLISH",
            "D-2": "SMS-10AM-ENGLISH",
            "D-1": "WH-2PM-ENGLISH",
        }
    ])
    predue_days = ["D-5", "D-4", "D-3", "D-2", "D-1"]
    capped = apply_predue_risk_rules(df, predue_days=predue_days)
    # LOW risk allows exactly 2 pre-due communications
    active_count = sum(1 for d in predue_days if capped.at[0, d] not in ("", "-"))
    assert active_count == 2
    assert capped.at[0, "D-5"] == "SMS-11AM-ENGLISH"


def test_prepare_campaign_outputs_maps_unknown_to_lr(tmp_path):
    pred_df = pd.DataFrame([
        {
            "Loan_number": "ACC_UNK_01",
            "SOURCE_RISK": "UNKNOWN",
            "SOURCE_VERTICAL": "LAP",
            "EMI_DATE": "05/10/2026",
            "SOURCE_MONTH_USED": "SEP-2026",
            "MONTH": "OCT-2026",
            "D-5": "SMS-11AM-ENGLISH",
            "D-1": "WH-1PM-ENGLISH",
            "PREDICTION_REASON": '{"D-5": "Model prediction"}',
            "IS_NEW_CUSTOMER": "False",
        }
    ])
    pred_file = tmp_path / "predictions.csv"
    pred_df.to_csv(pred_file, index=False)

    campaigns, mappings = _prepare_campaign_outputs(
        pred_file,
        source_month_label="SEP-2026",
        prediction_month_label="OCT-2026",
        model_name="catboost_3m",
        emi_cycles=[5],
        vertical="LAP",
        vendors=["prutech-cpass"],
        campaign_days=["D-5", "D-1"],
    )
    assert not campaigns.empty
    assert not mappings.empty
    assert (campaigns["risk"] == "LR").all()
    assert (mappings["risk"] == "LR").all()
    assert "_LR_" in campaigns["name"].iloc[0]
    assert "LR" in campaigns["dataset_name"].iloc[0]
    # D-1 has no raw reason in PREDICTION_REASON, so it uses the baseline strategy guidelines
    d1_mapping = mappings[mappings["date"].str.contains("04-10-2026") | mappings["campaign_name"].str.contains("WHATSAPP")]
    assert any("LOW-risk strategy guidelines" in r for r in mappings["prediction_reason"])
