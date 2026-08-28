from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from app_logging import log_step, setup_logging
from entity_keys import annotate_entity_key, entity_key_column_name
from model_fallbacks import register_legacy_joblib_aliases
from pipeline_common import (
    DAY_COLUMNS,
    SCHEDULE_DAY_COLUMNS,
    build_feature_matrix,
    month_to_period,
    predict_top_k_by_risk,
    prepare_next_month_dataset,
)
from project_paths import (
    CASE_DATA_DIR,
    FEATURE_DATA_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    ensure_parent_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference only for the next-month CatBoost strategy model using "
            "an already trained model bundle."
        )
    )
    parser.add_argument(
        "--feature-file",
        default=str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
        help="Monthly aggregated feature CSV.",
    )
    parser.add_argument(
        "--schedule-file",
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
        help="Schedule-style strategy target CSV.",
    )
    parser.add_argument(
        "--model-file",
        default=str(MODEL_DIR / "next_month_strategy_catboost.joblib"),
        help="Saved CatBoost model bundle.",
    )
    parser.add_argument(
        "--prediction-file",
        default=str(PREDICTIONS_DIR / "apr_2026_strategy_predictions_catboost.csv"),
        help="Output predictions CSV file.",
    )
    parser.add_argument(
        "--prediction-source-months",
        nargs="*",
        default=[],
        help="Source months to score for next-month prediction.",
    )
    parser.add_argument(
        "--base-population-file",
        default=str(CASE_DATA_DIR / "digital_cases_APR2026.csv"),
        help="Current-month digital cases CSV used as full prediction base population.",
    )
    parser.add_argument(
        "--prepared-dataset-file",
        default=str(FEATURE_DATA_DIR / "strategy_model_input_inference.csv"),
        help="Audit CSV containing the prepared inference model input after month-lag feature expansion.",
    )
    parser.add_argument(
        "--daywise-prepared-dataset-file",
        default=str(FEATURE_DATA_DIR / "strategy_model_input_inference_daywise.csv"),
        help="Audit CSV showing one prepared inference input row per entity/source month/day.",
    )
    parser.add_argument(
        "--model-input-audit-row-limit",
        type=int,
        default=100,
        help="Number of base rows to expand in day-wise model-input audit files. Use 0 for all rows.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Application log file. Defaults to artifacts/logs/catboost_inference.log.",
    )
    args = parser.parse_args()
    missing = [
        name
        for name, value in {
            "--feature-file": args.feature_file,
            "--schedule-file": args.schedule_file,
            "--model-file": args.model_file,
            "--base-population-file": args.base_population_file,
        }.items()
        if not value
    ]
    if missing:
        parser.error("Missing required CLI args: " + ", ".join(missing))
    return args


def write_inference_daywise_model_input_audit(
    *,
    output_file: Path,
    prediction_rows: pd.DataFrame,
    X_pred: pd.DataFrame,
    prediction_output: pd.DataFrame,
    output_key: str,
    row_limit: int,
    logger,
) -> None:
    audit_source = prediction_rows.copy()
    feature_source = X_pred.copy()
    prediction_source = prediction_output.copy()
    if row_limit > 0:
        audit_source = audit_source.head(row_limit).copy()
        feature_source = feature_source.head(row_limit).copy()
        prediction_source = prediction_source.head(row_limit).copy()

    feature_columns = feature_source.columns.tolist()
    records: list[dict[str, object]] = []
    for row_position, (_, row) in enumerate(audit_source.iterrows()):
        feature_values = feature_source.iloc[row_position].to_dict()
        prediction_row = prediction_source.iloc[row_position]
        for day in DAY_COLUMNS:
            records.append(
                {
                    output_key: row.get(output_key),
                    "SOURCE_MONTH": row.get("SOURCE_MONTH"),
                    "PREDICTION_MONTH": prediction_row.get("MONTH"),
                    "RISK": row.get("RISK"),
                    "VERTICAL": row.get("VERTICAL"),
                    "emi_date": row.get("emi_date"),
                    "day": day,
                    "predictedStrategy": prediction_row.get(day),
                    **feature_values,
                }
            )

    output_file = ensure_parent_dir(output_file)
    output_columns = [
        output_key,
        "SOURCE_MONTH",
        "PREDICTION_MONTH",
        "RISK",
        "VERTICAL",
        "emi_date",
        "day",
        "predictedStrategy",
        *feature_columns,
    ]
    pd.DataFrame(records).reindex(columns=output_columns).to_csv(output_file, index=False)
    logger.info(
        "Saved day-wise inference model input audit | path=%s rows=%s source_rows=%s expanded_days=%s feature_columns=%s",
        output_file,
        len(records),
        len(audit_source),
        len(DAY_COLUMNS),
        len(feature_columns),
    )


def _normalize_text(series: pd.Series, default: str = "UNKNOWN") -> pd.Series:
    return (
        series.fillna(default)
        .astype(str)
        .str.strip()
        .replace({"": default})
    )


def load_base_population(base_population_file: Path, source_month_label: str) -> pd.DataFrame:
    base = pd.read_csv(base_population_file).copy()
    required = {"apac_card_number", "risk", "vertical", "collectable_amount", "emi_date"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"Base population file is missing required columns: {sorted(missing)}")

    base = annotate_entity_key(base, loan_column="apac_card_number", party_column="party_id", output_column=entity_key_column_name())
    base["APAC_CARD_NUMBER"] = _normalize_text(base["apac_card_number"], default="")
    base["ENTITY_KEY"] = _normalize_text(base[entity_key_column_name()], default="")
    base["RISK"] = _normalize_text(base["risk"]).str.upper()
    base["VERTICAL"] = _normalize_text(base["vertical"]).str.upper()
    base["SOURCE_MONTH"] = source_month_label
    base["collectable_amount"] = pd.to_numeric(base["collectable_amount"], errors="coerce").fillna(0.0)
    base["emi_date"] = _normalize_text(base["emi_date"], default="")
    base = base[base["APAC_CARD_NUMBER"].ne("") & base["ENTITY_KEY"].ne("")].copy()
    base = base.drop_duplicates(subset=["APAC_CARD_NUMBER", "emi_date"], keep="first").reset_index(drop=True)
    return base[["APAC_CARD_NUMBER", "ENTITY_KEY", "SOURCE_MONTH", "RISK", "VERTICAL", "collectable_amount", "emi_date"]]


def build_prediction_population(
    *,
    dataset: pd.DataFrame,
    base_population: pd.DataFrame,
    prediction_source_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_month_period = month_to_period(pd.Series([prediction_source_month])).iloc[0]
    key_column = "ENTITY_KEY" if "ENTITY_KEY" in dataset.columns else "APAC_CARD_NUMBER"
    feature_rows = dataset[dataset["TARGET_MONTH"].isna()].copy()
    feature_rows = feature_rows[feature_rows["SOURCE_MONTH_PERIOD"].notna()].copy()
    feature_rows = feature_rows[feature_rows["SOURCE_MONTH_PERIOD"] <= source_month_period].copy()
    feature_rows = (
        feature_rows.sort_values([key_column, "SOURCE_MONTH_PERIOD"])
        .drop_duplicates(subset=[key_column], keep="last")
        .reset_index(drop=True)
    )
    feature_rows = feature_rows.drop(
        columns=["RISK", "VERTICAL", "TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK"],
        errors="ignore",
    )
    feature_rows["SOURCE_MONTH"] = prediction_source_month
    feature_rows["SOURCE_MONTH_PERIOD"] = source_month_period

    merged = base_population.merge(
        feature_rows,
        on=[key_column, "SOURCE_MONTH"],
        how="left",
        indicator=True,
    )
    merged["SOURCE_MONTH_PERIOD"] = month_to_period(merged["SOURCE_MONTH"])
    with_history = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
    without_history = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    return with_history, without_history


def _safe_numeric(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _strategy_parts(label: str) -> tuple[str, str, str] | None:
    if not label or label == "-" or pd.isna(label):
        return None
    parts = str(label).split("-", 2)
    if len(parts) != 3:
        return None
    channel, hour, language = parts
    channel_map = {"SMS": "SMS", "WH": "WH", "WHATSAPP": "WH", "IVR": "VOICE", "VOICE": "VOICE", "VOICE_BOT": "VOICE_BOT"}
    normalized_channel = channel_map.get(channel.upper())
    if not normalized_channel:
        return None
    return normalized_channel, hour.upper(), language.upper()


def _matching_success_signal(row: pd.Series, label: str) -> tuple[str, float]:
    parts = _strategy_parts(label)
    if not parts:
        return "", 0.0
    channel, hour, language = parts
    column = f"{channel}_SUCCESS_{hour}_{language}"
    value = _safe_numeric(row.get(column))
    return (column, value) if value > 0 else ("", 0.0)


def _business_friendly_time(hour: str) -> str:
    normalized = str(hour).upper().strip()
    if normalized.endswith("AM") or normalized.endswith("PM"):
        suffix = normalized[-2:]
        value = normalized[:-2]
        if value.isdigit():
            return f"{int(value)}:00 {suffix}"
    return hour


def _readable_strategy(label: str) -> str:
    parts = _strategy_parts(label)
    if not parts:
        return "no campaign"
    channel, hour, language = parts
    channel_name = {"SMS": "SMS", "WH": "WhatsApp", "VOICE": "voice call", "VOICE_BOT": "voice bot"}.get(channel, channel)
    language_name = "Regional language" if language.upper() == "REGIONAL" else language.title()
    return f"{channel_name} at {hour} in {language_name}"


def _ranked_business_labels(predicted: object) -> list[str]:
    return [label.strip() for label in str(predicted or "-").split("|") if label.strip()]


def _channel_totals(source_row: pd.Series | None) -> dict[str, int]:
    return {
        "SMS": int(_safe_numeric(source_row.get("SMS_TOTAL_INTENSITY"))) if source_row is not None else 0,
        "WH": int(_safe_numeric(source_row.get("WH_TOTAL_INTENSITY"))) if source_row is not None else 0,
        "VOICE": int(_safe_numeric(source_row.get("VOICE_TOTAL_INTENSITY"))) if source_row is not None else 0,
        "VOICE_BOT": int(_safe_numeric(source_row.get("VOICE_BOT_TOTAL_INTENSITY"))) if source_row is not None else 0,
    }


def _channel_total_for_label(source_row: pd.Series | None, label: str) -> tuple[str, float]:
    parts = _strategy_parts(label)
    if not parts:
        return "", 0.0
    channel, _hour, _language = parts
    column = f"{channel}_TOTAL_INTENSITY"
    return column, _safe_numeric(source_row.get(column)) if source_row is not None else 0.0


def _support_percentage(source_row: pd.Series | None, label: str) -> tuple[str, float, str, float]:
    success_column, success_count = _matching_success_signal(source_row, label)
    total_column, total_count = _channel_total_for_label(source_row, label)
    if total_count <= 0 or success_count <= 0:
        return success_column, success_count, total_column, 0.0
    return success_column, success_count, total_column, round((success_count / total_count) * 100.0, 2)


def _feature_value(source_row: pd.Series | None, column: str) -> float:
    return _safe_numeric(source_row.get(column)) if source_row is not None else 0.0


def _camel_case_feature(column: str) -> str:
    parts = str(column).lower().split("_")
    if not parts:
        return column
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _has_channel_success(source_row: pd.Series | None, channel: str) -> bool:
    if source_row is None:
        return False
    prefix = f"{channel}_SUCCESS"
    return any(str(column).startswith(prefix) and _safe_numeric(value) > 0 for column, value in source_row.items())


def _no_campaign_reason(day: str) -> str:
    if day == "D":
        return "At this stage, no campaign is recommended on the EMI due date to avoid unnecessary communication with the customer."
    return "At this stage, no campaign is recommended to prevent excessive communication with the customer."


def _business_reason_for_label(label: str, day: str, source_row: pd.Series | None) -> str:
    parts = _strategy_parts(label)
    if not parts:
        return "Campaign is recommended because it is aligned with the customer's past communication pattern."

    channel, _hour, language = parts
    readable = _readable_strategy(label)
    totals = _channel_totals(source_row)
    has_exact_success = bool(_matching_success_signal(source_row, label)[0]) if source_row is not None else False
    has_channel_success = _has_channel_success(source_row, channel)
    is_predue = day.startswith("D-")
    is_postdue = day.startswith("D+")

    if channel == "SMS":
        if day == "D-5":
            return f"{readable} is recommended as an early EMI reminder. This is a light-touch communication before the due date and is suitable for starting the follow-up journey."
        if day == "D-4":
            return f"{readable} is recommended because SMS has been an effective communication channel for this customer in earlier interactions."
        if day == "D+1":
            return f"As an immediate follow-up after the Cycle date, {readable} is recommended because SMS has shown a positive response pattern in earlier customer communication."
        if day == "D+2":
            if language == "REGIONAL":
                return f"For continued follow-up after the Cycle date, {readable} is recommended because regional language communication may help improve customer understanding and response."
            return f"For continued follow-up after the Cycle date, {readable} is recommended because SMS has remained a suitable communication option for this customer."
        if day == "D+3":
            return f"If payment is still pending at this stage, {readable} is recommended because SMS remains a suitable follow-up option for this customer."
        if day == "D+5":
            return f"If the account still requires attention at this stage, {readable} is recommended because SMS remains a suitable communication option for follow-up."
        if has_exact_success or has_channel_success:
            return f"{readable} is recommended because SMS has worked well for this customer in previous communication."
        if language == "REGIONAL" and is_postdue:
            return f"{readable} is recommended because regional language communication may help improve customer understanding and response."
        if is_predue and totals["SMS"] >= max(totals["WH"], totals["VOICE"]):
            return f"{readable} is recommended because SMS has been a suitable communication channel for this customer before the due date."
        return f"{readable} is recommended because it is aligned with the customer's past communication pattern."

    if channel == "WH":
        if has_exact_success or has_channel_success or totals["WH"] >= max(totals["SMS"], totals["VOICE"]):
            return f"{readable} is recommended because WhatsApp has been an effective communication channel for this customer in earlier interactions."
        return f"{readable} is recommended because it is a suitable digital reminder option for this customer."

    if channel == "VOICE":
        if has_exact_success or has_channel_success:
            return f"{readable} is recommended because direct customer interaction has shown a positive response in earlier communication."
        return f"{readable} is recommended because direct customer interaction may be helpful for this account."

    if channel == "VOICE_BOT":
        if has_exact_success or has_channel_success:
            return f"{readable} is recommended because automated voice bot communication has shown a positive response in earlier interactions."
        return f"{readable} is recommended because an automated voice bot reminder may be helpful for this account."

    return f"{readable} is recommended because it is aligned with the customer's past communication pattern."

def _business_alternate_reason_for_label(label: str, day: str, source_row: pd.Series | None) -> str:
    parts = _strategy_parts(label)
    if not parts:
        return "An alternate communication option may be used if additional follow-up is required."

    channel, hour, language = parts
    time_text = _business_friendly_time(hour)
    language_text = "Regional language" if language == "REGIONAL" else language.title()

    if channel == "SMS":
        if day == "D-1":
            article = "an" if language_text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            return f"If a final reminder is required before the EMI due date, {article} {language_text} SMS may be sent at {time_text} as the next course of action."
        article = "an" if language_text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        if day == "D+3":
            return f"If payment is still pending at this stage, {article} {language_text} SMS may be sent at {time_text} as the next course of action."
        if day == "D+5":
            return f"If the account still requires follow-up five days after the Cycle date, {article} {language_text} SMS may be sent at {time_text} as the next course of action."
        return f"If additional follow-up is required, {article} {language_text} SMS may be sent at {time_text} as the next course of action."

    if channel == "WH":
        article = "an" if language_text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"If additional follow-up is required, {article} {language_text} WhatsApp message may be sent at {time_text} as the next course of action."

    if channel == "VOICE":
        article = "an" if language_text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"If direct customer interaction is required, {article} {language_text} voice call may be initiated at {time_text} as the next course of action."

    if channel == "VOICE_BOT":
        article = "an" if language_text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"If an automated follow-up is required, {article} {language_text} voice bot reminder may be initiated at {time_text} as the next course of action."

    return f"If additional follow-up is required, the next communication may be initiated at {time_text} in {language_text}."


def _fallback_prediction_reason(day: str, fallback_config: dict[str, str], predicted_label: str) -> str:
    fallback_reason = fallback_config.get("fallback_reason")
    if fallback_reason == "missing_day_data":
        return "No campaign is recommended because historical training data was not available for this day, so the system has defaulted to no communication."
    if fallback_reason == "single_unique_value":
        if predicted_label == "-":
            return "No campaign is recommended because historical training data for this day contained only one unique outcome, so the system has defaulted to no communication."
        readable = _readable_strategy(predicted_label)
        return f"{readable} is recommended because historical training data for this day contained only one unique outcome, so the system has applied that same recommendation consistently."
    return "The recommendation for this day used fallback logic because standard training data was not available."


def build_prediction_reason(
    *,
    prediction_row: pd.Series,
    source_row: pd.Series | None,
    day_fallback_config: dict[str, dict[str, str]] | None = None,
) -> str:
    payload: dict[str, str] = {}
    for day in SCHEDULE_DAY_COLUMNS:
        if day == "D":
            payload[day] = _no_campaign_reason(day)
            continue

        ranked_labels = _ranked_business_labels(prediction_row.get(day, "-"))
        first_label = ranked_labels[0] if ranked_labels else "-"
        fallback_info = (day_fallback_config or {}).get(day)
        if fallback_info is not None:
            payload[day] = _fallback_prediction_reason(day, fallback_info, first_label)
            continue
        alternate_label = next((label for label in ranked_labels[1:] if label != "-"), None)
        if first_label == "-" and alternate_label is not None:
            payload[day] = _no_campaign_reason(day) + " " + _business_alternate_reason_for_label(
                alternate_label,
                day,
                source_row,
            )
            continue

        best_label = first_label if first_label != "-" else None
        if best_label is None:
            payload[day] = _no_campaign_reason(day)
            continue

        payload[day] = _business_reason_for_label(best_label, day, source_row)

    return json.dumps(payload, sort_keys=True)


def main() -> None:
    register_legacy_joblib_aliases()
    args = parse_args()
    logger = setup_logging(args.log_file, "catboost_inference")
    logger.info("CatBoost inference args: %s", vars(args))

    model_file = Path(args.model_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Saved CatBoost model bundle not found: {model_file}. "
            "Train the model once before running inference."
        )

    prediction_file = ensure_parent_dir(args.prediction_file)
    try:
        with log_step(logger, "load_model", model_file=model_file):
            bundle = joblib.load(model_file)
            target_offset_months = int(bundle.get("target_offset_months", 1))
            history_window_months = int(bundle.get("history_window_months", 1))
            day_fallback_config = bundle.get("day_fallback_config", {})
            logger.info(
                "Loaded model bundle | target_offset_months=%s history_window_months=%s feature_columns=%s",
                target_offset_months,
                history_window_months,
                len(bundle["feature_columns"]),
            )

        with log_step(logger, "prepare_dataset", feature_file=args.feature_file, schedule_file=args.schedule_file):
            dataset = prepare_next_month_dataset(
                feature_file=Path(args.feature_file),
                schedule_file=Path(args.schedule_file),
                target_offset_months=target_offset_months,
                history_window_months=history_window_months,
            )
            logger.info("Prepared inference dataset | rows=%s columns=%s", len(dataset), len(dataset.columns))
            if not any(str(column).startswith("VOICE_BOT_") for column in dataset.columns):
                logger.warning("VOICE_BOT channel was not present in the inference input data. Inference will continue without VOICE_BOT history.")

        if len(args.prediction_source_months) != 1:
            raise ValueError("Inference with digital_cases base population requires exactly one prediction source month.")
        prediction_source_month = args.prediction_source_months[0]
        available_dataset_months = sorted(dataset["SOURCE_MONTH"].dropna().astype(str).unique().tolist()) if "SOURCE_MONTH" in dataset.columns else []
        logger.info(
            "Inference month summary | prediction_source_month=%s target_offset_months=%s history_window_months=%s available_dataset_source_months=%s",
            prediction_source_month,
            target_offset_months,
            history_window_months,
            available_dataset_months,
        )

        with log_step(logger, "load_base_population", base_population_file=args.base_population_file):
            base_population = load_base_population(Path(args.base_population_file), prediction_source_month)
            logger.info("Loaded base population | rows=%s", len(base_population))

        with log_step(logger, "select_prediction_rows", source_month=prediction_source_month):
            prediction_rows, blank_rows = build_prediction_population(
                dataset=dataset,
                base_population=base_population,
                prediction_source_month=prediction_source_month,
            )
            logger.info(
                "Selected prediction rows | with_history=%s without_history=%s",
                len(prediction_rows),
                len(blank_rows),
            )

        prediction_outputs: list[pd.DataFrame] = []

        if not prediction_rows.empty:
            with log_step(logger, "build_prediction_matrix"):
                output_key = "APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in prediction_rows.columns else ("ENTITY_KEY" if "ENTITY_KEY" in prediction_rows.columns else None)
                X_pred = build_feature_matrix(prediction_rows).reindex(
                    columns=bundle["feature_columns"],
                    fill_value=0,
                )
                logger.info("Prediction matrix | shape=%s", X_pred.shape)
                prepared_dataset_file = ensure_parent_dir(args.prepared_dataset_file)
                inference_model_input = prediction_rows[[output_key, "SOURCE_MONTH", "RISK", "VERTICAL", "emi_date"]].copy()
                inference_model_input = pd.concat([inference_model_input.reset_index(drop=True), X_pred.reset_index(drop=True)], axis=1)
                inference_model_input.to_csv(prepared_dataset_file, index=False)
                logger.info(
                    "Saved actual inference model input dataset | path=%s rows=%s columns=%s",
                    prepared_dataset_file,
                    len(inference_model_input),
                    len(inference_model_input.columns),
                )

            with log_step(logger, "predict_day_columns", rows=len(X_pred)):
                prediction_output = prediction_rows[[output_key, "SOURCE_MONTH", "RISK", "VERTICAL", "emi_date"]].copy()
                prediction_output = prediction_output.rename(
                    columns={
                        output_key: "Loan_number",
                        "SOURCE_MONTH": "SOURCE_MONTH_USED",
                        "RISK": "SOURCE_RISK",
                        "VERTICAL": "SOURCE_VERTICAL",
                        "emi_date": "EMI_DATE",
                    }
                )

                source_period = month_to_period(prediction_output["SOURCE_MONTH_USED"])
                prediction_output["MONTH"] = (
                    source_period + target_offset_months
                ).dt.to_timestamp().dt.strftime("%b-%Y").str.upper()

                prediction_output["IS_NEW_CUSTOMER"] = "False"
                for day in DAY_COLUMNS:
                    logger.info("Predicting day column | day=%s", day)
                    encoder = bundle["label_encoders"][day]
                    model = bundle["models"][day]
                    prediction_output[day] = predict_top_k_by_risk(
                        model,
                        encoder,
                        X_pred,
                        prediction_rows["RISK"],
                        bounce_flags=prediction_rows["bounce_flag"] if "bounce_flag" in prediction_rows.columns else None,
                        day=day,
                        logger=logger,
                    )
                prediction_output["PREDICTION_REASON"] = [
                    build_prediction_reason(
                        prediction_row=prediction_output.iloc[row_idx],
                        source_row=prediction_rows.iloc[row_idx],
                        day_fallback_config=day_fallback_config,
                    )
                    for row_idx in range(len(prediction_output))
                ]
                write_inference_daywise_model_input_audit(
                    output_file=Path(args.daywise_prepared_dataset_file),
                    prediction_rows=prediction_rows,
                    X_pred=X_pred,
                    prediction_output=prediction_output,
                    output_key=output_key,
                    row_limit=args.model_input_audit_row_limit,
                    logger=logger,
                )
                prediction_outputs.append(prediction_output)

        if not blank_rows.empty:
            blank_output = blank_rows[["APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in blank_rows.columns else "ENTITY_KEY", "SOURCE_MONTH", "RISK", "VERTICAL", "emi_date"]].copy()
            blank_output = blank_output.rename(
                columns={("APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in blank_rows.columns else "ENTITY_KEY"): "Loan_number",
                    "SOURCE_MONTH": "SOURCE_MONTH_USED",
                    "RISK": "SOURCE_RISK",
                    "VERTICAL": "SOURCE_VERTICAL",
                    "emi_date": "EMI_DATE",
                }
            )
            source_period = month_to_period(blank_output["SOURCE_MONTH_USED"])
            blank_output["MONTH"] = (
                source_period + target_offset_months
            ).dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
            for day in DAY_COLUMNS:
                blank_output[day] = "-"
            blank_output["PREDICTION_REASON"] = [
                build_prediction_reason(
                    prediction_row=blank_output.iloc[row_idx],
                    source_row=None,
                )
                for row_idx in range(len(blank_output))
            ]
            blank_output["IS_NEW_CUSTOMER"] = "True"
            new_loans = blank_output["Loan_number"].tolist()
            logger.info("New customers identified (no history): count=%d, loan_numbers=%s", len(new_loans), new_loans)
            prediction_outputs.append(blank_output)

        if prediction_outputs:
            prediction_output = pd.concat(prediction_outputs, ignore_index=True)
        else:
            prediction_output = pd.DataFrame(
                columns=[
                    "SOURCE_RISK",
                    "SOURCE_VERTICAL",
                    "EMI_DATE",
                    "Loan_number",
                    "SOURCE_MONTH_USED",
                    "MONTH",
                    *SCHEDULE_DAY_COLUMNS,
                    "PREDICTION_REASON",
                    "IS_NEW_CUSTOMER",
                ]
            )

        with log_step(logger, "write_predictions", prediction_file=prediction_file):
            prediction_output["D"] = "-"
            prediction_output = prediction_output[
                [
                    "SOURCE_RISK",
                    "SOURCE_VERTICAL",
                    "EMI_DATE",
                    "Loan_number",
                    "SOURCE_MONTH_USED",
                    "MONTH",
                    *SCHEDULE_DAY_COLUMNS,
                    "PREDICTION_REASON",
                    "IS_NEW_CUSTOMER",
                ]
            ]
            prediction_output.to_csv(prediction_file, index=False)
            logger.info("Saved predictions | rows=%s bytes=%s", len(prediction_output), prediction_file.stat().st_size)

        print(f"Prediction rows: {len(prediction_output):,}")
        print(f"Saved inference-only predictions to {prediction_file}")
        logger.info("CatBoost inference completed successfully.")
    except Exception:
        logger.exception("CatBoost inference failed.")
        raise


if __name__ == "__main__":
    main()
