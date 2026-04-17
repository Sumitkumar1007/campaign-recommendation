from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project_paths import (
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    TRAINING_DATA_DIR,
    ensure_parent_dir,
)

DAY_COLUMNS = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
NON_FEATURE_COLUMNS = DAY_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a next-month strategy model using current-month communication "
            "features and next-month day-wise strategy targets."
        )
    )
    parser.add_argument(
        "--feature-file",
        default=str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
        help="Month/day feature dataset from generate_strategy_dataset.py",
    )
    parser.add_argument(
        "--schedule-file",
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
        help="Schedule-style target dataset from build_strategy_schedule_dataset.py",
    )
    parser.add_argument(
        "--model-file",
        default=str(MODEL_DIR / "next_month_strategy_model.joblib"),
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_DIR / "next_month_strategy_metrics.json"),
        help="Where to save evaluation metrics.",
    )
    parser.add_argument(
        "--prediction-file",
        default=str(PREDICTIONS_DIR / "apr_2026_strategy_predictions.csv"),
        help="Where to save inference results for the next month after the latest source month.",
    )
    parser.add_argument(
        "--target-offset-months",
        type=int,
        default=1,
        help=(
            "How many months ahead the target schedule should come from. "
            "Use 0 for same-month modeling, 1 for next-month modeling."
        ),
    )
    parser.add_argument(
        "--train-source-months",
        nargs="*",
        default=["NOV-2025", "DEC-2025", "JAN-2026"],
        help="Source months used for training. For next-month prediction, source NOV predicts DEC, etc.",
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
        help="Source months used for future inference, typically the latest available month.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in each random forest.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Parallel workers used across the day-wise output models.",
    )
    return parser.parse_args()


def month_to_period(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce").dt.to_period("M")


def derive_risk(group: pd.DataFrame) -> str:
    voice_intensity = float(group.get("VOICE_TOTAL_INTENSITY", pd.Series(dtype=float)).sum())
    sms_intensity = float(group.get("SMS_TOTAL_INTENSITY", pd.Series(dtype=float)).sum())
    wh_intensity = float(group.get("WH_TOTAL_INTENSITY", pd.Series(dtype=float)).sum())

    voice_failures = float(
        group.filter(regex=r"^VOICE_FAILED_").sum(numeric_only=True).sum()
    )
    digital_success = float(
        group.filter(regex=r"^(SMS_SUCCESS_|WH_SUCCESS_)").sum(numeric_only=True).sum()
    )

    total_intensity = sms_intensity + wh_intensity + voice_intensity
    if total_intensity == 0:
        return "LOW"

    voice_failure_ratio = voice_failures / voice_intensity if voice_intensity else 0.0
    digital_success_ratio = digital_success / total_intensity

    if voice_failure_ratio >= 0.7 and digital_success_ratio < 0.35:
        return "HIGH"
    if digital_success_ratio >= 0.55:
        return "LOW"
    return "MEDIUM"


def build_source_feature_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy()
    feature_df["MONTH_PERIOD"] = month_to_period(feature_df["MONTH"])
    value_columns = [
        col
        for col in feature_df.columns
        if col not in {"APAC_CARD_NUMBER", "MONTH", "MONTH_PERIOD", "DAY", "PREDICTED_STRATEGY"}
    ]
    key_columns = ["APAC_CARD_NUMBER", "MONTH", "MONTH_PERIOD"]
    merged = (
        feature_df[key_columns + value_columns]
        .groupby(key_columns, as_index=False)
        .sum(numeric_only=True)
    )
    day_activity = (
        feature_df.assign(
            MONTH_TOTAL_INTENSITY=feature_df[
                ["SMS_TOTAL_INTENSITY", "WH_TOTAL_INTENSITY", "VOICE_TOTAL_INTENSITY"]
            ].sum(axis=1)
        )
        .pivot_table(
            index=key_columns,
            columns="DAY",
            values="MONTH_TOTAL_INTENSITY",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    day_activity.columns = [
        col if isinstance(col, str) and col in key_columns else f"DAY_TOTAL_{col}"
        for col in day_activity.columns
    ]
    merged = merged.merge(day_activity, on=key_columns, how="left")

    risk_df = (
        feature_df.groupby(["APAC_CARD_NUMBER", "MONTH", "MONTH_PERIOD"], as_index=False)
        .apply(lambda g: pd.Series({"RISK": derive_risk(g)}))
        .reset_index(drop=True)
    )
    merged = merged.merge(risk_df, on=key_columns, how="left")
    merged = merged.rename(columns={"MONTH": "SOURCE_MONTH", "MONTH_PERIOD": "SOURCE_MONTH_PERIOD"})
    return merged


def build_target_table(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule_df = schedule_df.copy()
    schedule_df["MONTH_PERIOD"] = month_to_period(schedule_df["MONTH"])
    keep_cols = ["Loan_number", "MONTH", "MONTH_PERIOD", "RISK", *DAY_COLUMNS]
    target_df = schedule_df[keep_cols].rename(
        columns={
            "Loan_number": "APAC_CARD_NUMBER",
            "MONTH": "TARGET_MONTH",
            "MONTH_PERIOD": "TARGET_MONTH_PERIOD",
            "RISK": "TARGET_RISK",
        }
    )
    return target_df


def build_supervised_pairs(
    feature_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    target_offset_months: int,
) -> pd.DataFrame:
    source_df = build_source_feature_table(feature_df)
    target_df = build_target_table(schedule_df)
    source_df["TARGET_MONTH_PERIOD"] = source_df["SOURCE_MONTH_PERIOD"] + target_offset_months

    merged = source_df.merge(
        target_df,
        on=["APAC_CARD_NUMBER", "TARGET_MONTH_PERIOD"],
        how="left",
    )
    return merged


def build_pipeline(X: pd.DataFrame, n_estimators: int, n_jobs: int) -> Pipeline:
    categorical_columns = [
        col for col in X.columns if is_object_dtype(X[col]) or is_string_dtype(X[col])
    ]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
                ),
                numeric_columns,
            ),
        ]
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_split(model: Pipeline, split_df: pd.DataFrame) -> dict:
    if split_df.empty:
        return {"rows": 0}

    X = split_df.drop(columns=NON_FEATURE_COLUMNS)
    y_true = split_df[DAY_COLUMNS]
    y_pred = pd.DataFrame(model.predict(X), columns=DAY_COLUMNS, index=split_df.index)

    per_day_accuracy = {
        day: float(accuracy_score(y_true[day], y_pred[day])) for day in DAY_COLUMNS
    }
    exact_match = float((y_true == y_pred).all(axis=1).mean())
    avg_day_accuracy = float(pd.Series(per_day_accuracy).mean())

    return {
        "rows": int(len(split_df)),
        "exact_match_accuracy": exact_match,
        "average_day_accuracy": avg_day_accuracy,
        "per_day_accuracy": per_day_accuracy,
    }


def split_by_source_month(
    supervised_df: pd.DataFrame,
    months: list[str],
    require_target: bool = True,
) -> pd.DataFrame:
    selected = supervised_df[supervised_df["SOURCE_MONTH"].isin(months)].copy()
    if require_target:
        selected = selected.dropna(subset=["TARGET_MONTH"])
    return selected


def main() -> None:
    args = parse_args()
    feature_file = Path(args.feature_file)
    schedule_file = Path(args.schedule_file)
    model_file = ensure_parent_dir(args.model_file)
    metrics_file = ensure_parent_dir(args.metrics_file)
    prediction_file = ensure_parent_dir(args.prediction_file)

    feature_df = pd.read_csv(feature_file)
    schedule_df = pd.read_csv(schedule_file)
    supervised_df = build_supervised_pairs(
        feature_df,
        schedule_df,
        target_offset_months=args.target_offset_months,
    )

    train_df = split_by_source_month(supervised_df, args.train_source_months, require_target=True)
    validation_df = split_by_source_month(
        supervised_df, args.validation_source_months, require_target=True
    )
    test_df = split_by_source_month(supervised_df, args.test_source_months, require_target=True)
    prediction_df = split_by_source_month(
        supervised_df, args.prediction_source_months, require_target=False
    )

    if train_df.empty:
        raise ValueError("No training rows were found for the selected source months.")

    X_train = train_df.drop(columns=NON_FEATURE_COLUMNS)
    y_train = train_df[DAY_COLUMNS]

    pipeline = build_pipeline(X_train, args.n_estimators, args.n_jobs)
    pipeline.fit(X_train, y_train)

    metrics = {
        "assumption": (
            "Month-level communication features predict schedule targets with the "
            f"configured month offset of {args.target_offset_months}."
        ),
        "target_offset_months": args.target_offset_months,
        "train_source_months": args.train_source_months,
        "validation_source_months": args.validation_source_months,
        "test_source_months": args.test_source_months,
        "prediction_source_months": args.prediction_source_months,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "train_metrics": evaluate_split(pipeline, train_df),
        "validation_metrics": evaluate_split(pipeline, validation_df),
        "test_metrics": evaluate_split(pipeline, test_df),
    }

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": X_train.columns.tolist(),
        "target_columns": DAY_COLUMNS,
    }
    joblib.dump(model_bundle, model_file)
    metrics_file.write_text(json.dumps(metrics, indent=2))

    prediction_rows = prediction_df.copy()
    prediction_rows = prediction_rows[prediction_rows["TARGET_MONTH"].isna()].copy()
    if not prediction_rows.empty:
        X_pred = prediction_rows.drop(columns=NON_FEATURE_COLUMNS)
        preds = pd.DataFrame(pipeline.predict(X_pred), columns=DAY_COLUMNS, index=prediction_rows.index)
        prediction_output = prediction_rows[
            ["APAC_CARD_NUMBER", "SOURCE_MONTH", "RISK"]
        ].copy()
        prediction_output = prediction_output.rename(
            columns={"APAC_CARD_NUMBER": "Loan_number", "SOURCE_MONTH": "SOURCE_MONTH_USED"}
        )
        source_period = pd.to_datetime(
            prediction_output["SOURCE_MONTH_USED"], format="%b-%Y", errors="coerce"
        ).dt.to_period("M")
        prediction_output["PREDICTION_MONTH"] = (
            source_period + args.target_offset_months
        ).strftime("%b-%Y").str.upper()
        prediction_output = pd.concat([prediction_output, preds.reset_index(drop=True)], axis=1)
        prediction_output["D"] = "-"
        prediction_output = prediction_output[
            ["RISK", "Loan_number", "SOURCE_MONTH_USED", "PREDICTION_MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
        ]
        prediction_output.to_csv(prediction_file, index=False)
    else:
        pd.DataFrame(
            columns=["RISK", "Loan_number", "SOURCE_MONTH_USED", "PREDICTION_MONTH", "D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
        ).to_csv(prediction_file, index=False)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Saved model to {model_file}")
    print(f"Saved metrics to {metrics_file}")
    print(f"Saved future predictions to {prediction_file}")


if __name__ == "__main__":
    main()
