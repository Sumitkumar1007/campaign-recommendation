from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app_logging import log_step, setup_logging
from project_paths import FEATURE_DATA_DIR, MODEL_DIR, PREDICTIONS_DIR, SCHEDULE_DATA_DIR, ensure_parent_dir
from train_next_month_strategy_model_catboost import (
    DAY_COLUMNS,
    build_feature_matrix,
    build_rolling_feature_windows,
    month_to_period,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain CatBoost strategy predictions for one loan account or all prediction rows."
    )
    parser.add_argument(
        "--loan-number",
        default="",
        help="Optional single APAC/loan account number to explain. If omitted, all prediction rows are explained.",
    )
    parser.add_argument(
        "--prediction-file",
        default=str(PREDICTIONS_DIR / "2026_05_strategy_predictions_catboost_3m.csv"),
        help="Prediction CSV produced by predict_next_month_strategy_catboost.py.",
    )
    parser.add_argument(
        "--feature-file",
        default=str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
        help="Monthly feature CSV.",
    )
    parser.add_argument(
        "--schedule-file",
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
        help="Schedule history CSV.",
    )
    parser.add_argument(
        "--model-file",
        default=str(MODEL_DIR / "next_month_strategy_catboost_3m.joblib"),
        help="Saved CatBoost model bundle.",
    )
    parser.add_argument(
        "--source-month",
        default="",
        help="Source month label like APR-2026. Defaults to SOURCE_MONTH_USED from prediction file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/explanations",
        help="Directory where explanation reports will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of model probability candidates to include per day.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Application log file. Defaults to artifacts/logs/strategy_prediction_explainer.log.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_probability_list(items: list[dict[str, Any]]) -> str:
    return "; ".join(f"{item['label']}={item['probability']:.4f}" for item in items)


def selected_prediction_rows(prediction_file: Path, loan_number: str) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_file)
    predictions["Loan_number"] = predictions["Loan_number"].astype(str)
    if loan_number:
        predictions = predictions[predictions["Loan_number"] == loan_number].copy()
        if predictions.empty:
            raise ValueError(f"Loan number not found in prediction file: {loan_number}")
    return predictions.reset_index(drop=True)


def prepare_selected_source_rows(
    feature_file: Path,
    schedule_file: Path,
    prediction_rows: pd.DataFrame,
    source_month: str,
    target_offset_months: int,
    history_window_months: int,
) -> pd.DataFrame:
    accounts = set(prediction_rows["Loan_number"].astype(str))
    features = pd.read_csv(feature_file)
    features["APAC_CARD_NUMBER"] = features["APAC_CARD_NUMBER"].astype(str)
    features = features[features["APAC_CARD_NUMBER"].isin(accounts)].copy()
    rolled = build_rolling_feature_windows(features, history_window_months)
    rolled["SOURCE_MONTH"] = rolled["MONTH"]
    rolled["SOURCE_MONTH_PERIOD"] = month_to_period(rolled["SOURCE_MONTH"])
    rolled["TARGET_MONTH_PERIOD"] = rolled["SOURCE_MONTH_PERIOD"] + target_offset_months
    rolled = rolled.drop(columns=["MONTH"])

    schedule = pd.read_csv(schedule_file)
    schedule["Loan_number"] = schedule["Loan_number"].astype(str)
    schedule = schedule[schedule["Loan_number"].isin(accounts)].copy()
    schedule["TARGET_MONTH"] = schedule["MONTH"]
    schedule["TARGET_MONTH_PERIOD"] = month_to_period(schedule["TARGET_MONTH"])
    schedule = schedule.rename(columns={"Loan_number": "APAC_CARD_NUMBER", "RISK": "TARGET_RISK"})
    schedule = schedule.drop(columns=["MONTH"])

    dataset = rolled.merge(
        schedule[["APAC_CARD_NUMBER", "TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK", *DAY_COLUMNS]],
        on=["APAC_CARD_NUMBER", "TARGET_MONTH_PERIOD"],
        how="left",
    )
    dataset = dataset[dataset["SOURCE_MONTH"] == source_month].copy()
    if dataset.empty:
        raise ValueError(f"No prepared source rows found for source month {source_month}")
    return dataset.reset_index(drop=True)


def load_history(feature_file: Path, schedule_file: Path, accounts: set[str], source_month: str, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_period = pd.to_datetime(source_month, format="%b-%Y").to_period("M")
    min_period = source_period - (window - 1)

    features = pd.read_csv(feature_file)
    features["APAC_CARD_NUMBER"] = features["APAC_CARD_NUMBER"].astype(str)
    features["MONTH_PERIOD"] = month_to_period(features["MONTH"])
    features = features[
        features["APAC_CARD_NUMBER"].isin(accounts)
        & (features["MONTH_PERIOD"] >= min_period)
        & (features["MONTH_PERIOD"] <= source_period)
    ].copy()

    schedule = pd.read_csv(schedule_file)
    schedule["Loan_number"] = schedule["Loan_number"].astype(str)
    schedule["MONTH_PERIOD"] = month_to_period(schedule["MONTH"])
    schedule = schedule[
        schedule["Loan_number"].isin(accounts)
        & (schedule["MONTH_PERIOD"] >= min_period)
        & (schedule["MONTH_PERIOD"] <= source_period)
    ].copy()
    return features, schedule


def feature_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "feature_months": "",
            "sms_total": 0,
            "wh_total": 0,
            "voice_total": 0,
            "success_total": 0,
            "failed_total": 0,
            "dominant_success_signal": "",
            "dominant_success_count": 0,
            "dominant_failure_signal": "",
            "dominant_failure_count": 0,
        }

    numeric = rows.drop(columns=["APAC_CARD_NUMBER", "MONTH", "RISK", "MONTH_PERIOD"], errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0)
    success_cols = [col for col in numeric.columns if "SUCCESS" in col]
    failed_cols = [col for col in numeric.columns if "FAILED" in col]
    success_sums = numeric[success_cols].sum() if success_cols else pd.Series(dtype=float)
    failed_sums = numeric[failed_cols].sum() if failed_cols else pd.Series(dtype=float)
    dominant_success = success_sums.idxmax() if not success_sums.empty and success_sums.max() > 0 else ""
    dominant_failure = failed_sums.idxmax() if not failed_sums.empty and failed_sums.max() > 0 else ""

    return {
        "feature_months": ",".join(rows.sort_values("MONTH_PERIOD")["MONTH"].astype(str).tolist()),
        "sms_total": int(numeric.get("SMS_TOTAL_INTENSITY", pd.Series(dtype=float)).sum()),
        "wh_total": int(numeric.get("WH_TOTAL_INTENSITY", pd.Series(dtype=float)).sum()),
        "voice_total": int(numeric.get("VOICE_TOTAL_INTENSITY", pd.Series(dtype=float)).sum()),
        "success_total": int(success_sums.sum()) if not success_sums.empty else 0,
        "failed_total": int(failed_sums.sum()) if not failed_sums.empty else 0,
        "dominant_success_signal": dominant_success,
        "dominant_success_count": int(success_sums.get(dominant_success, 0)) if dominant_success else 0,
        "dominant_failure_signal": dominant_failure,
        "dominant_failure_count": int(failed_sums.get(dominant_failure, 0)) if dominant_failure else 0,
    }


def schedule_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "schedule_months": "",
            "nonblank_schedule_slots": 0,
            "all_blank_schedule_months": "",
            "schedule_history": "NO_SCHEDULE_ROWS",
        }

    rows = rows.sort_values("MONTH_PERIOD")
    nonblank_slots = int((rows[DAY_COLUMNS] != "-").sum().sum())
    all_blank_months = rows[rows[DAY_COLUMNS].eq("-").all(axis=1)]["MONTH"].astype(str).tolist()
    history_parts = []
    for _, row in rows.iterrows():
        values = [str(row[day]) for day in DAY_COLUMNS if str(row[day]) != "-"]
        if values:
            suffix = "..." if len(values) > 6 else ""
            history_parts.append(f"{row['MONTH']}=" + ";".join(values[:6]) + suffix)
        else:
            history_parts.append(f"{row['MONTH']}=ALL_BLANK")

    return {
        "schedule_months": ",".join(rows["MONTH"].astype(str).tolist()),
        "nonblank_schedule_slots": nonblank_slots,
        "all_blank_schedule_months": ",".join(all_blank_months),
        "schedule_history": " | ".join(history_parts),
    }


def probability_rows(bundle: dict[str, Any], source_rows: pd.DataFrame, top_n: int) -> dict[str, dict[str, Any]]:
    X = build_feature_matrix(source_rows).reindex(columns=bundle["feature_columns"], fill_value=0)
    by_account: dict[str, dict[str, Any]] = {}
    accounts = source_rows["APAC_CARD_NUMBER"].astype(str).tolist()

    for day in DAY_COLUMNS:
        model = bundle["models"][day]
        encoder = bundle["label_encoders"][day]
        probabilities = model.predict_proba(X)
        classes = list(encoder.classes_)
        blank_index = classes.index("-") if "-" in classes else None
        for row_idx, apac in enumerate(accounts):
            row_probs = probabilities[row_idx]
            order = row_probs.argsort()[::-1]
            top_items = [
                {"rank": rank + 1, "label": str(classes[index]), "probability": float(row_probs[index])}
                for rank, index in enumerate(order[:top_n])
            ]
            best_nonblank = next((item for item in top_items if item["label"] != "-"), None)
            blank_probability = float(row_probs[blank_index]) if blank_index is not None else 0.0
            by_account.setdefault(apac, {})[day] = {
                "top_items": top_items,
                "top_label": top_items[0]["label"],
                "top_probability": top_items[0]["probability"],
                "blank_probability": blank_probability,
                "best_nonblank_label": best_nonblank["label"] if best_nonblank else "",
                "best_nonblank_probability": best_nonblank["probability"] if best_nonblank else 0.0,
            }
    return by_account


def explain_reason(predicted_label: str, prob: dict[str, Any], summary: dict[str, Any], risk: str) -> str:
    if predicted_label == "-":
        if summary["nonblank_schedule_slots"] == 0:
            return "Predicted no-contact because the recent schedule history is fully blank/no-contact."
        if summary["failed_total"] > summary["success_total"]:
            return "Predicted no-contact because failure signal is stronger than success signal in recent history."
        if risk.upper() == "LOW":
            return "Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1."
        return "Predicted no-contact because it has the highest model probability; alternatives are shown after `|` for MEDIUM/HIGH."
    if summary["dominant_success_signal"]:
        return (
            f"Predicted {predicted_label} because it ranked highest for this day; recent strongest success signal is "
            f"{summary['dominant_success_signal']}."
        )
    return f"Predicted {predicted_label} because it has the highest model probability for this account/day."


def write_account_markdown(
    output_file: Path,
    account_row: dict[str, Any],
    day_rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Strategy prediction explanation: {account_row['APAC_CARD_NUMBER']}\n\n",
        "## Account Summary\n\n",
        markdown_table([account_row], [
            "APAC_CARD_NUMBER",
            "SOURCE_RISK",
            "SOURCE_MONTH_USED",
            "PREDICT_MONTH",
            "feature_months",
            "schedule_months",
            "sms_total",
            "wh_total",
            "voice_total",
            "success_total",
            "failed_total",
            "nonblank_schedule_slots",
            "all_blank_schedule_months",
        ]),
        "\n\n## Recent Schedule History\n\n",
        account_row["schedule_history"],
        "\n\n## Day-Level Explanation\n\n",
        markdown_table(day_rows, [
            "day",
            "prediction",
            "top_probability",
            "blank_probability",
            "best_nonblank_label",
            "best_nonblank_probability",
            "top_candidates",
            "reason",
        ]),
        "\n",
    ]
    output_file.write_text("".join(lines))


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file, "strategy_prediction_explainer")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with log_step(logger, "load_inputs"):
        prediction_file = Path(args.prediction_file)
        model_file = Path(args.model_file)
        feature_file = Path(args.feature_file)
        schedule_file = Path(args.schedule_file)
        predictions = selected_prediction_rows(prediction_file, args.loan_number)
        bundle = joblib.load(model_file)
        target_offset_months = int(bundle.get("target_offset_months", 1))
        history_window_months = int(bundle.get("history_window_months", 3))

    source_month = args.source_month or str(predictions["SOURCE_MONTH_USED"].iloc[0])
    accounts = set(predictions["Loan_number"].astype(str))

    with log_step(logger, "prepare_source_rows", accounts=len(accounts), source_month=source_month):
        source_rows = prepare_selected_source_rows(
            feature_file,
            schedule_file,
            predictions,
            source_month,
            target_offset_months,
            history_window_months,
        )
        features_history, schedule_history = load_history(
            feature_file,
            schedule_file,
            accounts,
            source_month,
            history_window_months,
        )
        probabilities = probability_rows(bundle, source_rows, args.top_n)

    account_records: list[dict[str, Any]] = []
    day_records: list[dict[str, Any]] = []
    for _, prediction in predictions.iterrows():
        apac = str(prediction["Loan_number"])
        feature_info = feature_summary(features_history[features_history["APAC_CARD_NUMBER"] == apac])
        schedule_info = schedule_summary(schedule_history[schedule_history["Loan_number"] == apac])
        account_record = {
            "APAC_CARD_NUMBER": apac,
            "SOURCE_RISK": prediction["SOURCE_RISK"],
            "SOURCE_MONTH_USED": prediction["SOURCE_MONTH_USED"],
            "PREDICT_MONTH": prediction["MONTH"],
            **feature_info,
            **schedule_info,
        }
        account_records.append(account_record)

        account_day_rows = []
        for day in DAY_COLUMNS:
            predicted_label = str(prediction[day])
            day_prob = probabilities.get(apac, {}).get(day, {})
            top_items = day_prob.get("top_items", [])
            day_record = {
                "APAC_CARD_NUMBER": apac,
                "SOURCE_RISK": prediction["SOURCE_RISK"],
                "SOURCE_MONTH_USED": prediction["SOURCE_MONTH_USED"],
                "PREDICT_MONTH": prediction["MONTH"],
                "day": day,
                "prediction": predicted_label,
                "top_probability": round(safe_float(day_prob.get("top_probability")), 4),
                "blank_probability": round(safe_float(day_prob.get("blank_probability")), 4),
                "best_nonblank_label": day_prob.get("best_nonblank_label", ""),
                "best_nonblank_probability": round(safe_float(day_prob.get("best_nonblank_probability")), 4),
                "top_candidates": format_probability_list(top_items),
                "reason": explain_reason(predicted_label.split("|")[0], day_prob, {**feature_info, **schedule_info}, str(prediction["SOURCE_RISK"])),
            }
            day_records.append(day_record)
            account_day_rows.append(day_record)

        if args.loan_number:
            write_account_markdown(
                output_dir / f"{apac}_prediction_explanation.md",
                account_record,
                account_day_rows,
            )

    account_output = ensure_parent_dir(output_dir / "prediction_explanation_account_summary.csv")
    day_output = ensure_parent_dir(output_dir / "prediction_explanation_day_detail.csv")
    pd.DataFrame(account_records).to_csv(account_output, index=False)
    pd.DataFrame(day_records).to_csv(day_output, index=False)

    if not args.loan_number:
        summary_file = output_dir / "prediction_explanation_summary.md"
        summary_rows = pd.DataFrame(account_records)
        summary_file.write_text(
            "\n".join(
                [
                    "# Strategy prediction explanation summary",
                    "",
                    f"Accounts explained: {len(account_records):,}",
                    f"Day explanation rows: {len(day_records):,}",
                    f"Source month: {source_month}",
                    f"Prediction file: `{prediction_file}`",
                    "",
                    "## Risk Counts",
                    "",
                    summary_rows["SOURCE_RISK"].value_counts(dropna=False).to_frame("count").to_csv(),
                    "",
                    "Detailed CSV outputs are in this folder.",
                    "",
                ]
            )
        )

    print(f"Accounts explained: {len(account_records):,}")
    print(f"Day explanation rows: {len(day_records):,}")
    print(f"Account summary CSV: {account_output}")
    print(f"Day detail CSV: {day_output}")
    if args.loan_number:
        print(f"Markdown explanation: {output_dir / f'{args.loan_number}_prediction_explanation.md'}")
    else:
        print(f"Markdown summary: {output_dir / 'prediction_explanation_summary.md'}")


if __name__ == "__main__":
    main()
