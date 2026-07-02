from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from artifact_versioning import copy_to_latest, next_versioned_path, write_versioned_json
import pandas as pd
from joblib import Parallel, delayed
from pipeline_common import (
    DAY_COLUMNS,
    build_feature_matrix,
    month_to_period,
    prepare_next_month_dataset,
    split_by_source_month,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from project_paths import (
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
            "Train a next-month strategy model using source-month monthly features "
            "and target-month schedule labels."
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
        default=str(MODEL_DIR / "next_month_strategy_logistic.joblib"),
        help="Output model file.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_DIR / "next_month_strategy_logistic_metrics.json"),
        help="Output metrics JSON file.",
    )
    parser.add_argument(
        "--prediction-file",
        default=str(PREDICTIONS_DIR / "apr_2026_strategy_predictions_logistic.csv"),
        help="Output predictions CSV file.",
    )
    parser.add_argument(
        "--target-offset-months",
        type=int,
        default=1,
        help="How many months ahead the target schedule should come from.",
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
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations for each logistic regression model.",
    )
    return parser.parse_args()


def prepare_dataset(
    feature_file: Path,
    schedule_file: Path,
    target_offset_months: int,
) -> pd.DataFrame:
    return prepare_next_month_dataset(
        feature_file=feature_file,
        schedule_file=schedule_file,
        target_offset_months=target_offset_months,
    )


def fit_day_model(
    day: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    max_iter: int,
) -> tuple[str, LogisticRegression, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train[day].astype(str))
    model = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_encoded)
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
        y_pred = encoder.inverse_transform(y_pred_encoded)
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
    model_file = ensure_parent_dir(args.model_file)
    metrics_file = ensure_parent_dir(args.metrics_file)
    prediction_file = ensure_parent_dir(args.prediction_file)
    versioned_model_file = next_versioned_path(model_file)
    versioned_metrics_file = next_versioned_path(metrics_file)
    versioned_prediction_file = next_versioned_path(prediction_file)
    dataset = prepare_dataset(
        Path(args.feature_file),
        Path(args.schedule_file),
        args.target_offset_months,
    )

    train_df = split_by_source_month(dataset, args.train_source_months, require_target=True)
    validation_df = split_by_source_month(
        dataset, args.validation_source_months, require_target=True
    )
    test_df = split_by_source_month(dataset, args.test_source_months, require_target=True)
    prediction_df = split_by_source_month(
        dataset, args.prediction_source_months, require_target=False
    )

    if train_df.empty:
        raise ValueError("No training rows found for the selected source months.")

    X_train = build_feature_matrix(train_df)
    X_validation = build_feature_matrix(validation_df).reindex(columns=X_train.columns, fill_value=0)
    X_test = build_feature_matrix(test_df).reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[DAY_COLUMNS].fillna("-")
    y_validation = validation_df[DAY_COLUMNS].fillna("-")
    y_test = test_df[DAY_COLUMNS].fillna("-")

    fitted_models = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="threads")(
        delayed(fit_day_model)(day, X_train, y_train, args.max_iter)
        for day in DAY_COLUMNS
    )
    models: dict[str, LogisticRegression] = {day: model for day, model, _ in fitted_models}
    label_encoders: dict[str, LabelEncoder] = {
        day: encoder for day, _, encoder in fitted_models
    }

    metrics = {
        "assumption": (
            "Source-month monthly communication features predict next-month schedule "
            f"targets with month offset {args.target_offset_months}."
        ),
        "target_offset_months": args.target_offset_months,
        "train_source_months": args.train_source_months,
        "validation_source_months": args.validation_source_months,
        "test_source_months": args.test_source_months,
        "prediction_source_months": args.prediction_source_months,
        "n_jobs": args.n_jobs,
        "max_iter": args.max_iter,
        "train_metrics": evaluate_models(models, label_encoders, X_train, y_train),
        "validation_metrics": evaluate_models(models, label_encoders, X_validation, y_validation),
        "test_metrics": evaluate_models(models, label_encoders, X_test, y_test),
    }

    model_bundle = {
        "models": models,
        "label_encoders": label_encoders,
        "feature_columns": X_train.columns.tolist(),
        "target_columns": DAY_COLUMNS,
        "target_offset_months": args.target_offset_months,
    }
    joblib.dump(model_bundle, versioned_model_file)
    copy_to_latest(source_path=versioned_model_file, latest_path=model_file)
    write_versioned_json(payload=metrics, latest_path=metrics_file, versioned_path=versioned_metrics_file)

    prediction_rows = prediction_df[prediction_df["TARGET_MONTH"].isna()].copy()
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
            prediction_output[day] = encoder.inverse_transform(models[day].predict(X_pred))
        prediction_output["D"] = "-"
        prediction_output = prediction_output[
            ["SOURCE_RISK", "Loan_number", "SOURCE_MONTH_USED", "MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
        ]
        prediction_output.to_csv(versioned_prediction_file, index=False)
    else:
        pd.DataFrame(
            columns=["SOURCE_RISK", "Loan_number", "SOURCE_MONTH_USED", "MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
        ).to_csv(versioned_prediction_file, index=False)

    copy_to_latest(source_path=versioned_prediction_file, latest_path=prediction_file)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Prediction rows: {len(prediction_rows):,}")
    print(f"Saved model to {model_file} (versioned copy: {versioned_model_file})")
    print(f"Saved metrics to {metrics_file} (versioned copy: {versioned_metrics_file})")
    print(f"Saved future predictions to {prediction_file} (versioned copy: {versioned_prediction_file})")


if __name__ == "__main__":
    main()
