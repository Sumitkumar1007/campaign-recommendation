from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from app_logging import log_step, setup_logging
from project_paths import (
    FEATURE_DATA_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    ensure_parent_dir,
)
from train_next_month_strategy_model_catboost import (
    DAY_COLUMNS,
    build_feature_matrix,
    month_to_period,
    prepare_dataset,
    predict_top_k_by_risk,
    split_by_source_month,
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
        "--log-file",
        default=None,
        help="Application log file. Defaults to artifacts/logs/catboost_inference.log.",
    )
    return parser.parse_args()


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
            dataset = prepare_dataset(
                Path(args.feature_file),
                Path(args.schedule_file),
                target_offset_months,
                history_window_months,
            )
            logger.info("Prepared inference dataset | rows=%s columns=%s", len(dataset), len(dataset.columns))

        with log_step(logger, "select_prediction_rows", source_months=args.prediction_source_months):
            prediction_df = split_by_source_month(
                dataset,
                args.prediction_source_months,
                require_target=False,
            )
            prediction_rows = prediction_df[prediction_df["TARGET_MONTH"].isna()].copy()
            logger.info(
                "Selected prediction rows | candidates=%s future_rows=%s",
                len(prediction_df),
                len(prediction_rows),
            )

        if prediction_rows.empty:
            pd.DataFrame(
                columns=[
                    "SOURCE_RISK",
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
            ).to_csv(prediction_file, index=False)
            logger.warning("No rows found for inference. Saved empty prediction file to %s", prediction_file)
            print("No rows found for inference.")
            print(f"Saved empty prediction file to {prediction_file}")
            return

        with log_step(logger, "build_prediction_matrix"):
            X_pred = build_feature_matrix(prediction_rows).reindex(
                columns=bundle["feature_columns"],
                fill_value=0,
            )
            logger.info("Prediction matrix | shape=%s", X_pred.shape)

        with log_step(logger, "predict_day_columns", rows=len(X_pred)):
            prediction_output = prediction_rows[["APAC_CARD_NUMBER", "SOURCE_MONTH", "RISK"]].copy()
            prediction_output = prediction_output.rename(
                columns={
                    "APAC_CARD_NUMBER": "Loan_number",
                    "SOURCE_MONTH": "SOURCE_MONTH_USED",
                    "RISK": "SOURCE_RISK",
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

        with log_step(logger, "write_predictions", prediction_file=prediction_file):
            prediction_output["D"] = "-"
            prediction_output = prediction_output[
                [
                    "SOURCE_RISK",
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
