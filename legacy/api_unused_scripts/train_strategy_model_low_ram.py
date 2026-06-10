from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from project_paths import (
    FEATURE_DATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    ensure_parent_dir,
)

DAY_COLUMNS = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
NON_FEATURE_COLUMNS = DAY_COLUMNS + ["D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a low-memory month-level strategy model."
    )
    parser.add_argument(
        "--feature-file",
        default=str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
        help="Monthly aggregated feature CSV.",
    )
    parser.add_argument(
        "--schedule-file",
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
        help="Schedule-style strategy targets.",
    )
    parser.add_argument(
        "--model-file",
        default=str(MODEL_DIR / "strategy_model_low_ram.joblib"),
        help="Output model file.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_DIR / "strategy_model_low_ram_metrics.json"),
        help="Output metrics JSON file.",
    )
    parser.add_argument(
        "--prediction-file",
        default=str(PREDICTIONS_DIR / "mar_2026_strategy_predictions_low_ram.csv"),
        help="Output predictions CSV file.",
    )
    parser.add_argument(
        "--train-months",
        nargs="*",
        default=["NOV-2025", "DEC-2025", "JAN-2026"],
        help="Months used for training.",
    )
    parser.add_argument(
        "--validation-months",
        nargs="*",
        default=["FEB-2026"],
        help="Months used for validation.",
    )
    parser.add_argument(
        "--test-months",
        nargs="*",
        default=["MAR-2026"],
        help="Months used for testing.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for fitting the per-day models.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum iterations for each logistic regression model.",
    )
    return parser.parse_args()


def prepare_dataset(feature_file: Path, schedule_file: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_file)
    schedule = pd.read_csv(schedule_file)

    merged = features.merge(
        schedule,
        left_on=["APAC_CARD_NUMBER", "MONTH"],
        right_on=["Loan_number", "MONTH"],
        how="inner",
        suffixes=("", "_target"),
    )
    merged = merged.drop(columns=["Loan_number"])
    if "RISK_target" in merged.columns:
        merged = merged.drop(columns=["RISK_target"])
    return merged


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    feature_df = feature_df.copy()
    feature_df["RISK"] = feature_df["RISK"].fillna("UNKNOWN")
    feature_df["MONTH"] = feature_df["MONTH"].fillna("UNKNOWN")
    feature_df = pd.get_dummies(feature_df, columns=["MONTH", "RISK"], dummy_na=False)
    feature_df = feature_df.drop(columns=["APAC_CARD_NUMBER"], errors="ignore")
    return feature_df.fillna(0)


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


def main() -> None:
    args = parse_args()
    model_file = ensure_parent_dir(args.model_file)
    metrics_file = ensure_parent_dir(args.metrics_file)
    prediction_file = ensure_parent_dir(args.prediction_file)
    dataset = prepare_dataset(Path(args.feature_file), Path(args.schedule_file))

    train_df = dataset[dataset["MONTH"].isin(args.train_months)].copy()
    validation_df = dataset[dataset["MONTH"].isin(args.validation_months)].copy()
    test_df = dataset[dataset["MONTH"].isin(args.test_months)].copy()

    if train_df.empty:
        raise ValueError("No training rows found for the selected months.")

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
    models: dict[str, LogisticRegression] = {
        day: model for day, model, _ in fitted_models
    }
    label_encoders: dict[str, LabelEncoder] = {
        day: encoder for day, _, encoder in fitted_models
    }

    metrics = {
        "train_months": args.train_months,
        "validation_months": args.validation_months,
        "test_months": args.test_months,
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
    }
    joblib.dump(model_bundle, model_file)
    metrics_file.write_text(json.dumps(metrics, indent=2))

    if not test_df.empty:
        prediction_output = test_df[["APAC_CARD_NUMBER", "MONTH", "RISK"]].copy()
        prediction_output = prediction_output.rename(columns={"APAC_CARD_NUMBER": "Loan_number"})
        for day in DAY_COLUMNS:
            encoder = label_encoders[day]
            prediction_output[day] = encoder.inverse_transform(models[day].predict(X_test))
        prediction_output["D"] = "-"
        prediction_output = prediction_output[
            ["RISK", "Loan_number", "MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
        ]
        prediction_output.to_csv(prediction_file, index=False)
    else:
        pd.DataFrame(
            columns=["RISK", "Loan_number", "MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
        ).to_csv(prediction_file, index=False)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Saved model to {model_file}")
    print(f"Saved metrics to {metrics_file}")
    print(f"Saved test-month predictions to {prediction_file}")


if __name__ == "__main__":
    main()
