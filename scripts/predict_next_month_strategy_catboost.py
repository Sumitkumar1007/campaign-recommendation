from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from app_logging import log_step, setup_logging
from pipeline_common import (
    DAY_COLUMNS,
    build_feature_matrix,
    month_to_period,
    predict_top_k_by_risk,
    prepare_next_month_dataset,
    split_by_source_month,
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
        default=["MAR-2026"],
        help="Source months to score for next-month prediction.",
    )
    parser.add_argument(
        "--base-population-file",
        default=str(CASE_DATA_DIR / "digital_cases_APR2026.csv"),
        help="Current-month digital cases CSV used as full prediction base population.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Application log file. Defaults to artifacts/logs/catboost_inference.log.",
    )
    return parser.parse_args()


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

    base["APAC_CARD_NUMBER"] = _normalize_text(base["apac_card_number"], default="")
    base["RISK"] = _normalize_text(base["risk"]).str.upper()
    base["VERTICAL"] = _normalize_text(base["vertical"]).str.upper()
    base["SOURCE_MONTH"] = source_month_label
    base["collectable_amount"] = pd.to_numeric(base["collectable_amount"], errors="coerce").fillna(0.0)
    base["emi_date"] = _normalize_text(base["emi_date"], default="")
    base = base[base["APAC_CARD_NUMBER"].ne("")].copy()
    base = base.drop_duplicates(subset=["APAC_CARD_NUMBER"], keep="first").reset_index(drop=True)
    return base[["APAC_CARD_NUMBER", "SOURCE_MONTH", "RISK", "VERTICAL", "collectable_amount", "emi_date"]]


def build_prediction_population(
    *,
    dataset: pd.DataFrame,
    base_population: pd.DataFrame,
    prediction_source_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_df = split_by_source_month(dataset, [prediction_source_month], require_target=False)
    feature_rows = prediction_df[prediction_df["TARGET_MONTH"].isna()].copy()
    feature_rows = feature_rows.drop(
        columns=["RISK", "VERTICAL", "TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK"],
        errors="ignore",
    )

    merged = base_population.merge(
        feature_rows,
        on=["APAC_CARD_NUMBER", "SOURCE_MONTH"],
        how="left",
        indicator=True,
    )
    merged["SOURCE_MONTH_PERIOD"] = month_to_period(merged["SOURCE_MONTH"])
    with_history = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
    without_history = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    return with_history, without_history


def main() -> None:
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

        if len(args.prediction_source_months) != 1:
            raise ValueError("Inference with digital_cases base population requires exactly one prediction source month.")
        prediction_source_month = args.prediction_source_months[0]

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
                X_pred = build_feature_matrix(prediction_rows).reindex(
                    columns=bundle["feature_columns"],
                    fill_value=0,
                )
                logger.info("Prediction matrix | shape=%s", X_pred.shape)

            with log_step(logger, "predict_day_columns", rows=len(X_pred)):
                prediction_output = prediction_rows[["APAC_CARD_NUMBER", "SOURCE_MONTH", "RISK", "VERTICAL"]].copy()
                prediction_output = prediction_output.rename(
                    columns={
                        "APAC_CARD_NUMBER": "Loan_number",
                        "SOURCE_MONTH": "SOURCE_MONTH_USED",
                        "RISK": "SOURCE_RISK",
                        "VERTICAL": "SOURCE_VERTICAL",
                    }
                )

                source_period = month_to_period(prediction_output["SOURCE_MONTH_USED"])
                prediction_output["MONTH"] = (
                    source_period + target_offset_months
                ).dt.to_timestamp().dt.strftime("%b-%Y").str.upper()

                for day in DAY_COLUMNS:
                    logger.info("Predicting day column | day=%s", day)
                    encoder = bundle["label_encoders"][day]
                    model = bundle["models"][day]
                    prediction_output[day] = predict_top_k_by_risk(
                        model,
                        encoder,
                        X_pred,
                        prediction_rows["RISK"],
                    )
                prediction_outputs.append(prediction_output)

        if not blank_rows.empty:
            blank_output = blank_rows[["APAC_CARD_NUMBER", "SOURCE_MONTH", "RISK", "VERTICAL"]].copy()
            blank_output = blank_output.rename(
                columns={
                    "APAC_CARD_NUMBER": "Loan_number",
                    "SOURCE_MONTH": "SOURCE_MONTH_USED",
                    "RISK": "SOURCE_RISK",
                    "VERTICAL": "SOURCE_VERTICAL",
                }
            )
            source_period = month_to_period(blank_output["SOURCE_MONTH_USED"])
            blank_output["MONTH"] = (
                source_period + target_offset_months
            ).dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
            for day in DAY_COLUMNS:
                blank_output[day] = "-"
            prediction_outputs.append(blank_output)

        if prediction_outputs:
            prediction_output = pd.concat(prediction_outputs, ignore_index=True)
        else:
            prediction_output = pd.DataFrame(
                columns=[
                    "SOURCE_RISK",
                    "SOURCE_VERTICAL",
                    "Loan_number",
                    "SOURCE_MONTH_USED",
                    "MONTH",
                    *DAY_COLUMNS,
                ]
            )

        with log_step(logger, "write_predictions", prediction_file=prediction_file):
            prediction_output["D"] = "-"
            prediction_output = prediction_output[
                [
                    "SOURCE_RISK",
                    "SOURCE_VERTICAL",
                    "Loan_number",
                    "SOURCE_MONTH_USED",
                    "MONTH",
                    "D-5",
                    "D-4",
                    "D-3",
                    "D-2",
                    "D-1",
                    "D",
                    "D+1",
                    "D+2",
                    "D+3",
                    "D+4",
                    "D+5",
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
