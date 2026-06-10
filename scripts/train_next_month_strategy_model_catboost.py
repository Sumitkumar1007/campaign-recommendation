from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from app_logging import log_step, setup_logging
from pipeline_common import (
    DAY_COLUMNS,
    build_feature_matrix,
    build_rolling_feature_windows,
    month_to_period,
    predict_top_k_by_risk,
    prepare_next_month_dataset,
    split_by_source_month,
)
from project_paths import (
    CATBOOST_INFO_DIR,
    CHECKPOINT_DIR,
    FEATURE_DATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    ensure_parent_dir,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a next-month CatBoost strategy model using source-month monthly "
            "features and target-month schedule labels."
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
        help="Output model file.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_DIR / "next_month_strategy_catboost_metrics.json"),
        help="Output metrics JSON file.",
    )
    parser.add_argument(
        "--prediction-file",
        default=str(PREDICTIONS_DIR / "apr_2026_strategy_predictions_catboost.csv"),
        help="Output predictions CSV file.",
    )
    parser.add_argument(
        "--target-offset-months",
        type=int,
        default=1,
        help="How many months ahead the target schedule should come from.",
    )
    parser.add_argument(
        "--history-window-months",
        type=int,
        default=3,
        help="Number of latest source months to roll into each account feature row.",
    )
    parser.add_argument(
        "--train-source-months",
        nargs="*",
        default=["NOV-2025", "DEC-2025", "JAN-2026"],
        help="Source months used for training.",
    )
    parser.add_argument(
        "--validation-source-months",
        nargs="*",
        default=["FEB-2026"],
        help="Source months used for validation.",
    )
    parser.add_argument(
        "--test-source-months",
        nargs="*",
        default=[],
        help="Source months used for testing.",
    )
    parser.add_argument(
        "--prediction-source-months",
        nargs="*",
        default=["MAR-2026"],
        help="Source months used for future inference.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Number of parallel workers for fitting the per-day models.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Boosting rounds for each CatBoost model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for each CatBoost model.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Tree depth for each CatBoost model.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Application log file. Defaults to artifacts/logs/catboost_training.log.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(CHECKPOINT_DIR / "catboost_3m"),
        help="Directory for per-day model checkpoints.",
    )
    return parser.parse_args()




def _sorted_month_labels(labels: list[str]) -> list[str]:
    if not labels:
        return []
    periods = pd.to_datetime(pd.Series(labels), format="%b-%Y", errors="coerce")
    pairs = [(label, period) for label, period in zip(labels, periods, strict=False) if not pd.isna(period)]
    pairs.sort(key=lambda item: item[1])
    return [label for label, _ in pairs]


def resolve_source_month_splits(
    dataset: pd.DataFrame,
    train_source_months: list[str],
    validation_source_months: list[str],
    test_source_months: list[str],
    prediction_source_months: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    targetable_months = _sorted_month_labels(
        dataset.loc[dataset["TARGET_MONTH"].notna(), "SOURCE_MONTH"].dropna().astype(str).unique().tolist()
    )
    all_months = _sorted_month_labels(dataset["SOURCE_MONTH"].dropna().astype(str).unique().tolist())

    if split_by_source_month(dataset, train_source_months, require_target=True).empty and targetable_months:
        if len(targetable_months) >= 2:
            effective_train = targetable_months[:-1]
            effective_validation = targetable_months[-1:]
        else:
            effective_train = targetable_months
            effective_validation = []
        effective_test = []
        effective_prediction = [all_months[-1]] if all_months else []
        return effective_train, effective_validation, effective_test, effective_prediction

    return train_source_months, validation_source_months, test_source_months, prediction_source_months

def prepare_dataset(
    feature_file: Path,
    schedule_file: Path,
    target_offset_months: int,
    history_window_months: int,
) -> pd.DataFrame:
    return prepare_next_month_dataset(
        feature_file=feature_file,
        schedule_file=schedule_file,
        target_offset_months=target_offset_months,
        history_window_months=history_window_months,
    )


def fit_day_model(
    day: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    iterations: int,
    learning_rate: float,
    depth: int,
    checkpoint_dir: Path,
    checkpoint_metadata: dict,
) -> tuple[str, CatBoostClassifier, LabelEncoder]:
    logger = logging.getLogger("catboost_training")
    checkpoint_file = checkpoint_dir / f"{day.replace('+', 'plus').replace('-', 'minus')}.joblib"
    if checkpoint_file.exists():
        checkpoint = joblib.load(checkpoint_file)
        expected_metadata = {
            **checkpoint_metadata,
            "day": day,
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "feature_columns": X_train.columns.tolist(),
            "target_value_counts": y_train[day].astype(str).value_counts().sort_index().to_dict(),
        }
        if checkpoint.get("metadata") == expected_metadata:
            logger.info("LOAD fit_day_model checkpoint | day=%s path=%s", day, checkpoint_file)
            return day, checkpoint["model"], checkpoint["label_encoder"]
        logger.info("SKIP stale fit_day_model checkpoint | day=%s path=%s", day, checkpoint_file)

    start = time.perf_counter()
    logger.info(
        "START fit_day_model | day=%s rows=%s columns=%s iterations=%s learning_rate=%s depth=%s",
        day,
        len(X_train),
        len(X_train.columns),
        iterations,
        learning_rate,
        depth,
    )
    train_dir = CATBOOST_INFO_DIR / day
    train_dir.mkdir(parents=True, exist_ok=True)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train[day].astype(str))
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        thread_count=1,
        train_dir=str(train_dir),
    )
    model.fit(X_train, y_encoded)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "day": day,
            "model": model,
            "label_encoder": encoder,
            "metadata": {
                **checkpoint_metadata,
                "day": day,
                "iterations": iterations,
                "learning_rate": learning_rate,
                "depth": depth,
                "feature_columns": X_train.columns.tolist(),
                "target_value_counts": y_train[day].astype(str).value_counts().sort_index().to_dict(),
            },
            "classes": encoder.classes_.tolist(),
        },
        checkpoint_file,
    )
    logger.info(
        "END fit_day_model | day=%s classes=%s checkpoint=%s elapsed_seconds=%.2f",
        day,
        len(encoder.classes_),
        checkpoint_file,
        time.perf_counter() - start,
    )
    return day, model, encoder


def evaluate_models(models: dict, label_encoders: dict, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    if X.empty:
        return {"rows": 0}

    per_day_accuracy: dict[str, float] = {}
    pred_columns: dict[str, pd.Series] = {}
    for day in DAY_COLUMNS:
        encoder = label_encoders[day]
        model = models[day]
        y_pred_encoded = model.predict(X)
        y_pred_encoded = pd.Series(y_pred_encoded.reshape(-1), index=y.index)
        y_pred = encoder.inverse_transform(y_pred_encoded.astype(int))
        pred_columns[day] = pd.Series(y_pred, index=y.index)
        per_day_accuracy[day] = float(accuracy_score(y[day], y_pred))

    pred_df = pd.DataFrame(pred_columns)
    exact_match_accuracy = float((pred_df == y).all(axis=1).mean())
    average_day_accuracy = float(pd.Series(per_day_accuracy).mean())
    return {
        "rows": int(len(X)),
        "exact_match_accuracy": exact_match_accuracy,
        "average_day_accuracy": average_day_accuracy,
        "per_day_accuracy": per_day_accuracy,
    }


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file, "catboost_training")
    logger.info("CatBoost training args: %s", vars(args))

    model_file = ensure_parent_dir(args.model_file)
    metrics_file = ensure_parent_dir(args.metrics_file)
    prediction_file = ensure_parent_dir(args.prediction_file)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        with log_step(logger, "prepare_dataset", feature_file=args.feature_file, schedule_file=args.schedule_file):
            dataset = prepare_dataset(
                Path(args.feature_file),
                Path(args.schedule_file),
                args.target_offset_months,
                args.history_window_months,
            )
            logger.info(
                "Prepared dataset | rows=%s columns=%s source_months=%s",
                len(dataset),
                len(dataset.columns),
                sorted(dataset["SOURCE_MONTH"].dropna().unique().tolist()),
            )

        with log_step(logger, "split_dataset"):
            effective_train_source_months, effective_validation_source_months, effective_test_source_months, effective_prediction_source_months = resolve_source_month_splits(
                dataset,
                args.train_source_months,
                args.validation_source_months,
                args.test_source_months,
                args.prediction_source_months,
            )
            train_df = split_by_source_month(dataset, effective_train_source_months, require_target=True)
            validation_df = split_by_source_month(dataset, effective_validation_source_months, require_target=True)
            test_df = split_by_source_month(dataset, effective_test_source_months, require_target=True)
            prediction_df = split_by_source_month(dataset, effective_prediction_source_months, require_target=False)
            logger.info(
                "Split rows | train=%s validation=%s test=%s prediction_candidates=%s",
                len(train_df),
                len(validation_df),
                len(test_df),
                len(prediction_df),
            )
            logger.info(
                "Effective source months | train=%s validation=%s test=%s prediction=%s",
                effective_train_source_months,
                effective_validation_source_months,
                effective_test_source_months,
                effective_prediction_source_months,
            )

        if train_df.empty:
            raise ValueError("No training rows found for the selected source months.")

        with log_step(logger, "build_feature_matrices"):
            X_train = build_feature_matrix(train_df)
            X_validation = build_feature_matrix(validation_df).reindex(columns=X_train.columns, fill_value=0)
            X_test = build_feature_matrix(test_df).reindex(columns=X_train.columns, fill_value=0)
            logger.info(
                "Feature matrices | train=%s validation=%s test=%s columns=%s",
                X_train.shape,
                X_validation.shape,
                X_test.shape,
                len(X_train.columns),
            )

        if len(train_df) < 2:
            raise ValueError(f"Training data insufficient. Found {len(train_df)} training row(s) after month split.")

        y_train = train_df[DAY_COLUMNS].fillna("-")
        y_validation = validation_df[DAY_COLUMNS].fillna("-")
        y_test = test_df[DAY_COLUMNS].fillna("-")
        logger.info("Target matrices | train=%s validation=%s test=%s", y_train.shape, y_validation.shape, y_test.shape)

        with log_step(logger, "fit_all_day_models", n_jobs=args.n_jobs, days=",".join(DAY_COLUMNS)):
            completed_checkpoints = sorted(path.name for path in checkpoint_dir.glob("*.joblib"))
            checkpoint_metadata = {
                "target_offset_months": args.target_offset_months,
                "history_window_months": args.history_window_months,
                "train_source_months": effective_train_source_months,
                "train_rows": int(len(train_df)),
            }
            logger.info(
                "Checkpoint directory | path=%s existing_checkpoints=%s",
                checkpoint_dir,
                completed_checkpoints,
            )
            fitted_models = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="threads")(
                delayed(fit_day_model)(
                    day,
                    X_train,
                    y_train,
                    args.iterations,
                    args.learning_rate,
                    args.depth,
                    checkpoint_dir,
                    checkpoint_metadata,
                )
                for day in DAY_COLUMNS
            )
            models: dict[str, CatBoostClassifier] = {day: model for day, model, _ in fitted_models}
            label_encoders: dict[str, LabelEncoder] = {day: encoder for day, _, encoder in fitted_models}

        with log_step(logger, "evaluate_models"):
            train_metrics = evaluate_models(models, label_encoders, X_train, y_train)
            validation_metrics = evaluate_models(models, label_encoders, X_validation, y_validation)
            test_metrics = evaluate_models(models, label_encoders, X_test, y_test)
            logger.info("Train metrics: %s", train_metrics)
            logger.info("Validation metrics: %s", validation_metrics)
            logger.info("Test metrics: %s", test_metrics)

        metrics = {
            "assumption": (
                "Source-month monthly communication features predict next-month schedule "
                f"targets with month offset {args.target_offset_months} using the latest "
                f"{args.history_window_months} months of per-account feature history."
            ),
            "target_offset_months": args.target_offset_months,
            "history_window_months": args.history_window_months,
            "train_source_months": effective_train_source_months,
            "validation_source_months": effective_validation_source_months,
            "test_source_months": effective_test_source_months,
            "prediction_source_months": effective_prediction_source_months,
            "n_jobs": args.n_jobs,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

        model_bundle = {
            "models": models,
            "label_encoders": label_encoders,
            "feature_columns": X_train.columns.tolist(),
            "target_columns": DAY_COLUMNS,
            "target_offset_months": args.target_offset_months,
            "history_window_months": args.history_window_months,
        }

        with log_step(logger, "save_model_and_metrics"):
            joblib.dump(model_bundle, model_file)
            metrics_file.write_text(json.dumps(metrics, indent=2))
            logger.info("Saved model | path=%s bytes=%s", model_file, model_file.stat().st_size)
            logger.info("Saved metrics | path=%s bytes=%s", metrics_file, metrics_file.stat().st_size)

        with log_step(logger, "write_predictions"):
            prediction_rows = prediction_df[prediction_df["TARGET_MONTH"].isna()].copy()
            logger.info("Prediction rows needing future target | rows=%s", len(prediction_rows))
            if not prediction_rows.empty:
                X_pred = build_feature_matrix(prediction_rows).reindex(columns=X_train.columns, fill_value=0)
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
                    source_period + args.target_offset_months
                ).dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
                for day in DAY_COLUMNS:
                    encoder = label_encoders[day]
                    prediction_output[day] = predict_top_k_by_risk(
                        models[day],
                        encoder,
                        X_pred,
                        prediction_rows["RISK"],
                    )
                prediction_output["D"] = "-"
                prediction_output = prediction_output[
                    ["SOURCE_RISK", "Loan_number", "SOURCE_MONTH_USED", "MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
                ]
                prediction_output.to_csv(prediction_file, index=False)
            else:
                pd.DataFrame(
                    columns=["SOURCE_RISK", "Loan_number", "SOURCE_MONTH_USED", "MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
                ).to_csv(prediction_file, index=False)
            logger.info("Saved predictions | path=%s bytes=%s", prediction_file, prediction_file.stat().st_size)

        print(f"Training rows: {len(train_df):,}")
        print(f"Validation rows: {len(validation_df):,}")
        print(f"Test rows: {len(test_df):,}")
        print(f"Prediction rows: {len(prediction_rows):,}")
        print(f"Saved model to {model_file}")
        print(f"Saved metrics to {metrics_file}")
        print(f"Saved future predictions to {prediction_file}")
        logger.info("CatBoost training completed successfully.")
    except Exception:
        logger.exception("CatBoost training failed.")
        raise


if __name__ == "__main__":
    main()
