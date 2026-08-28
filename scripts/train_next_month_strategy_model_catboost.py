from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

import numpy as np
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from app_logging import log_step, setup_logging
from env_utils import load_dotenv
from artifact_versioning import next_versioned_directory, next_versioned_path
from model_fallbacks import ConstantDayModel, register_legacy_joblib_aliases
from pipeline_common import (
    DAY_COLUMNS,
    DAY_WEIGHT_COLUMN_MAP,
    SCHEDULE_DAY_COLUMNS,
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
        # default=str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
        default=str(FEATURE_DATA_DIR / "strategy_monthly_features_train.csv"),
        help="Monthly aggregated feature CSV.",
    )
    parser.add_argument(
        "--schedule-file",
        # default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_train.csv"),
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
        "--prepared-dataset-file",
        default=str(FEATURE_DATA_DIR / "strategy_model_input_train.csv"),
        help="Audit CSV containing the prepared model input after month-lag feature expansion.",
    )
    parser.add_argument(
        "--daywise-prepared-dataset-file",
        default=str(FEATURE_DATA_DIR / "strategy_model_input_train_daywise.csv"),
        help="Audit CSV showing one prepared training input row per entity/source month/day.",
    )
    parser.add_argument(
        "--model-input-audit-row-limit",
        type=int,
        default=100,
        help="Number of base rows to expand in day-wise model-input audit files. Use 0 for all rows.",
    )
    parser.add_argument(
        "--train-source-months",
        nargs="*",
        default=[],
        help="Optional explicit source months used for training. Defaults to the latest available dataset months.",
    )
    parser.add_argument(
        "--validation-source-months",
        nargs="*",
        default=[],
        help="Optional explicit source months used for validation. Defaults to the latest available dataset months.",
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
        default=[],
        help="Optional explicit source months used for future inference. Defaults to the latest available dataset month.",
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
    parser.add_argument(
        "--model-version",
        default="",
        help="Optional model version token used to save version-specific artifacts, for example v002.",
    )
    return parser.parse_args()




register_legacy_joblib_aliases()


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

    has_explicit_split = any([
        train_source_months,
        validation_source_months,
        test_source_months,
        prediction_source_months,
    ])
    if not has_explicit_split and targetable_months:
        if len(targetable_months) >= 3:
            effective_train = targetable_months[:-2]
            effective_validation = targetable_months[-2:-1]
            effective_test = targetable_months[-1:]
        elif len(targetable_months) == 2:
            effective_train = targetable_months[:1]
            effective_validation = targetable_months[-1:]
            effective_test = []
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



def analyze_day_target_variation(
    day: str,
    y_train: pd.DataFrame,
    *,
    original_day_present: bool = True,
) -> dict[str, str] | None:
    if not original_day_present:
        return {
            "fallback_reason": "missing_day_data",
            "fallback_value": "-",
            "log_message": f"Training data for {day} is missing from the D-5 to D+20 window. Defaulting predictions to '-' for this day.",
        }
    raw_series = y_train[day] if day in y_train.columns else pd.Series(dtype=object)
    non_null_values = raw_series.dropna().astype(str).str.strip()
    non_null_values = non_null_values[non_null_values.ne("")]
    if non_null_values.empty:
        return {
            "fallback_reason": "missing_day_data",
            "fallback_value": "-",
            "log_message": f"Training data for {day} is missing from the D-5 to D+20 window. Defaulting predictions to '-' for this day.",
        }
    unique_values = sorted(non_null_values.unique().tolist())
    if len(unique_values) == 1:
        fallback_value = unique_values[0]
        if fallback_value == "-":
            message = f"Training data for {day} has only one unique value '-'. Defaulting predictions to '-' for this day."
        else:
            message = f"Training data for {day} has only one unique value {fallback_value!r}. Defaulting predictions to that same value for this day."
        return {
            "fallback_reason": "single_unique_value",
            "fallback_value": fallback_value,
            "log_message": message,
        }
    return None


def validate_day_target_variation(day: str, y_train: pd.DataFrame) -> str | None:
    fallback = analyze_day_target_variation(day, y_train)
    return None if fallback is None else fallback["fallback_value"]


def build_constant_day_fallback(target_value: str) -> tuple[ConstantDayModel, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit([str(target_value)])
    model = ConstantDayModel(encoded_value=0, class_count=len(encoder.classes_))
    return model, encoder


def fit_day_model(
    day: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    sample_weight: np.ndarray | None,
    iterations: int,
    learning_rate: float,
    depth: int,
    checkpoint_dir: Path,
    checkpoint_metadata: dict,
    fallback_config: dict[str, str] | None = None,
) -> tuple[str, Any, LabelEncoder]:
    logger = logging.getLogger("catboost_training")
    checkpoint_file = checkpoint_dir / f"{day.replace('+', 'plus').replace('-', 'minus')}.joblib"
    fallback_config = fallback_config or analyze_day_target_variation(day, y_train)
    fallback_value = None if fallback_config is None else fallback_config["fallback_value"]
    sample_weight_enabled = sample_weight is not None
    sample_weight_min = None if sample_weight is None or len(sample_weight) == 0 else float(np.min(sample_weight))
    sample_weight_max = None if sample_weight is None or len(sample_weight) == 0 else float(np.max(sample_weight))
    sample_weight_sum = None if sample_weight is None or len(sample_weight) == 0 else float(np.sum(sample_weight))
    expected_metadata = {
        **checkpoint_metadata,
        "day": day,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "feature_columns": X_train.columns.tolist(),
        "target_value_counts": y_train[day].astype(str).value_counts().sort_index().to_dict(),
        "sample_weight_enabled": sample_weight_enabled,
        "sample_weight_min": sample_weight_min,
        "sample_weight_max": sample_weight_max,
        "sample_weight_sum": sample_weight_sum,
    }
    if fallback_config is not None:
        expected_metadata.update({"fallback_value": str(fallback_value), "fallback_reason": fallback_config["fallback_reason"], "model_kind": "constant"})
    if checkpoint_file.exists():
        checkpoint = joblib.load(checkpoint_file)
        if checkpoint.get("metadata") == expected_metadata:
            logger.info("LOAD fit_day_model checkpoint | day=%s path=%s", day, checkpoint_file)
            return day, checkpoint["model"], checkpoint["label_encoder"]
        logger.info("SKIP stale fit_day_model checkpoint | day=%s path=%s", day, checkpoint_file)

    start = time.perf_counter()
    logger.info(
        "START fit_day_model | day=%s rows=%s columns=%s iterations=%s learning_rate=%s depth=%s sample_weight_enabled=%s sample_weight_min=%s sample_weight_max=%s",
        day,
        len(X_train),
        len(X_train.columns),
        iterations,
        learning_rate,
        depth,
        sample_weight_enabled,
        sample_weight_min,
        sample_weight_max,
    )
    if fallback_config is not None:
        logger.warning(
            "Using constant fallback for day target | day=%s value=%r reason=%s rows=%s message=%s",
            day,
            fallback_value,
            fallback_config["fallback_reason"],
            len(y_train),
            fallback_config["log_message"],
        )
        model, encoder = build_constant_day_fallback(fallback_value)
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
                    "fallback_value": str(fallback_value),
                    "fallback_reason": fallback_config["fallback_reason"],
                    "model_kind": "constant",
                },
                "classes": encoder.classes_.tolist(),
            },
            checkpoint_file,
        )
        logger.info(
            "END fit_day_model | day=%s classes=%s checkpoint=%s elapsed_seconds=%.2f fallback=true",
            day,
            len(encoder.classes_),
            checkpoint_file,
            time.perf_counter() - start,
        )
        return day, model, encoder

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
    model.fit(X_train, y_encoded, sample_weight=sample_weight)
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


def versioned_artifact_path(base_path: Path, model_version: str) -> Path:
    token = str(model_version).strip()
    if not token:
        raise ValueError("model_version is required for versioned artifact naming.")
    if not token.lower().startswith("v"):
        token = f"v{token}"
    return base_path.with_name(f"{base_path.stem}_{token}{base_path.suffix}")


def versioned_checkpoint_dir(base_dir: Path, model_version: str) -> Path:
    token = str(model_version).strip()
    if not token:
        raise ValueError("model_version is required for versioned checkpoint naming.")
    if not token.lower().startswith("v"):
        token = f"v{token}"
    return base_dir.parent / f"{base_dir.name}_{token}"


def write_training_daywise_model_input_audit(
    *,
    output_file: Path,
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    row_limit: int,
    logger: logging.Logger,
) -> None:
    audit_source = train_df.copy()
    feature_source = X_train.copy()
    if row_limit > 0:
        audit_source = audit_source.head(row_limit).copy()
        feature_source = feature_source.head(row_limit).copy()

    key_column = "APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in audit_source.columns else "ENTITY_KEY"
    feature_columns = feature_source.columns.tolist()
    records: list[dict[str, object]] = []
    for row_position, (_, row) in enumerate(audit_source.iterrows()):
        feature_values = feature_source.iloc[row_position].to_dict()
        for day in DAY_COLUMNS:
            records.append(
                {
                    key_column: row.get(key_column),
                    "SOURCE_MONTH": row.get("SOURCE_MONTH"),
                    "TARGET_MONTH": row.get("TARGET_MONTH"),
                    "RISK": row.get("RISK"),
                    "VERTICAL": row.get("VERTICAL"),
                    "day": day,
                    "targetStrategy": row.get(day),
                    "dayWeight": row.get(DAY_WEIGHT_COLUMN_MAP[day], 1.0),
                    **feature_values,
                }
            )

    output_file = ensure_parent_dir(output_file)
    output_columns = [
        key_column,
        "SOURCE_MONTH",
        "TARGET_MONTH",
        "RISK",
        "VERTICAL",
        "day",
        "targetStrategy",
        "dayWeight",
        *feature_columns,
    ]
    pd.DataFrame(records).reindex(columns=output_columns).to_csv(output_file, index=False)
    logger.info(
        "Saved day-wise training model input audit | path=%s rows=%s source_rows=%s expanded_days=%s feature_columns=%s",
        output_file,
        len(records),
        len(audit_source),
        len(DAY_COLUMNS),
        len(feature_columns),
    )


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    logger = setup_logging(args.log_file, "catboost_training")
    logger.info("CatBoost training args: %s", vars(args))

    model_file = ensure_parent_dir(args.model_file)
    metrics_file = ensure_parent_dir(args.metrics_file)
    prediction_file = ensure_parent_dir(args.prediction_file)
    if args.model_version.strip():
        versioned_model_file = versioned_artifact_path(model_file, args.model_version)
        versioned_metrics_file = versioned_artifact_path(metrics_file, args.model_version)
        checkpoint_dir = versioned_checkpoint_dir(Path(args.checkpoint_dir), args.model_version)
    else:
        versioned_model_file = next_versioned_path(model_file)
        versioned_metrics_file = next_versioned_path(metrics_file)
        checkpoint_dir = next_versioned_directory(Path(args.checkpoint_dir))
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
            logger.info(
                "Month split summary | target_offset_months=%s history_window_months=%s train_month_count=%s validation_month_count=%s test_month_count=%s prediction_month_count=%s",
                args.target_offset_months,
                args.history_window_months,
                len(effective_train_source_months),
                len(effective_validation_source_months),
                len(effective_test_source_months),
                len(effective_prediction_source_months),
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
            logger.info(
                "Training rows by source month | train=%s validation=%s test=%s prediction_candidates=%s",
                train_df["SOURCE_MONTH"].value_counts(sort=False).to_dict(),
                validation_df["SOURCE_MONTH"].value_counts(sort=False).to_dict(),
                test_df["SOURCE_MONTH"].value_counts(sort=False).to_dict(),
                prediction_df["SOURCE_MONTH"].value_counts(sort=False).to_dict(),
            )
            prepared_dataset_file = ensure_parent_dir(args.prepared_dataset_file)
            train_df.to_csv(prepared_dataset_file, index=False)
            logger.info(
                "Saved actual training model dataset | path=%s rows=%s columns=%s day_target_rows=%s weight_rows=%s",
                prepared_dataset_file,
                len(train_df),
                len(train_df.columns),
                int(train_df[DAY_COLUMNS].notna().any(axis=1).sum()),
                int(train_df[DAY_WEIGHT_COLUMN_MAP.values()].notna().any(axis=1).sum()),
            )
            write_training_daywise_model_input_audit(
                output_file=Path(args.daywise_prepared_dataset_file),
                train_df=train_df,
                X_train=X_train,
                row_limit=args.model_input_audit_row_limit,
                logger=logger,
            )

        if len(train_df) < 2:
            raise ValueError(f"Training data insufficient. Found {len(train_df)} training row(s) after month split.")

        train_target_frame = train_df.reindex(columns=DAY_COLUMNS)
        validation_target_frame = validation_df.reindex(columns=DAY_COLUMNS)
        test_target_frame = test_df.reindex(columns=DAY_COLUMNS)
        train_weight_frame = pd.DataFrame(index=train_df.index)
        for day in DAY_COLUMNS:
            weight_column = DAY_WEIGHT_COLUMN_MAP[day]
            train_weight_frame[day] = pd.to_numeric(
                train_df.get(weight_column, 1.0),
                errors="coerce",
            ).fillna(1.0)
        y_train = train_target_frame.fillna("-")
        y_validation = validation_target_frame.fillna("-")
        y_test = test_target_frame.fillna("-")
        day_fallback_config = {
            day: analyze_day_target_variation(day, train_target_frame, original_day_present=(day in train_df.columns))
            for day in DAY_COLUMNS
        }
        for day, fallback in day_fallback_config.items():
            if fallback is not None:
                logger.warning("Day fallback activated | day=%s reason=%s value=%r message=%s", day, fallback["fallback_reason"], fallback["fallback_value"], fallback["log_message"])
        logger.info("Target matrices | train=%s validation=%s test=%s", y_train.shape, y_validation.shape, y_test.shape)

        with log_step(logger, "fit_all_day_models", n_jobs=args.n_jobs, days=",".join(DAY_COLUMNS)):
            completed_checkpoints = sorted(path.name for path in checkpoint_dir.glob("*.joblib"))
            sample_weight_mode = os.getenv("STRATEGY_USE_STATUS_WEIGHTS", "false").strip().lower()
            use_sample_weights = sample_weight_mode in {"1", "true", "yes", "on"}
            checkpoint_metadata = {
                "target_offset_months": args.target_offset_months,
                "history_window_months": args.history_window_months,
                "train_source_months": effective_train_source_months,
                "train_rows": int(len(train_df)),
                "sample_weight_mode": sample_weight_mode,
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
                    train_weight_frame[day].to_numpy(dtype=float) if use_sample_weights else None,
                    args.iterations,
                    args.learning_rate,
                    args.depth,
                    checkpoint_dir,
                    checkpoint_metadata,
                    day_fallback_config.get(day),
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

        # model_bundle = {
        #     "models": models,
        #     "label_encoders": label_encoders,
        #     "feature_columns": X_train.columns.tolist(),
        #     "target_columns": DAY_COLUMNS,
        #     "target_offset_months": args.target_offset_months,
        #     "history_window_months": args.history_window_months,
        #     "day_fallback_config": {day: fallback for day, fallback in day_fallback_config.items() if fallback is not None},
        # }

        if len(X_train) > 10000:
            baseline_matrix = X_train.sample(n=10000, random_state=42)
        else:
            baseline_matrix = X_train.copy()

        model_bundle = {
            "models": models,
            "label_encoders": label_encoders,
            "feature_columns": X_train.columns.tolist(),
            "target_columns": DAY_COLUMNS,
            "target_offset_months": args.target_offset_months,
            "history_window_months": args.history_window_months,
            "day_fallback_config": {day: fallback for day, fallback in day_fallback_config.items() if fallback is not None},
            # "baseline_matrix": baseline_matrix,  # <--- Drift calculator will look for this key
        }

        with log_step(logger, "save_model_and_metrics"):
            versioned_model_file.parent.mkdir(parents=True, exist_ok=True)
            versioned_metrics_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_bundle, versioned_model_file)
            versioned_metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            logger.info("Saved versioned model | path=%s bytes=%s", versioned_model_file, versioned_model_file.stat().st_size)
            logger.info("Saved versioned metrics | path=%s bytes=%s", versioned_metrics_file, versioned_metrics_file.stat().st_size)

        with log_step(logger, "write_predictions"):
            prediction_rows = prediction_df[prediction_df["TARGET_MONTH"].isna()].copy()
            logger.info("Prediction rows needing future target | rows=%s", len(prediction_rows))
            if not prediction_rows.empty:
                X_pred = build_feature_matrix(prediction_rows).reindex(columns=X_train.columns, fill_value=0)
                output_key = "APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in prediction_rows.columns else "ENTITY_KEY"
                prediction_output = prediction_rows[[output_key, "SOURCE_MONTH", "RISK"]].copy()
                prediction_output = prediction_output.rename(
                    columns={
                        output_key: "Loan_number",
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
                        bounce_flags=prediction_rows["bounce_flag"] if "bounce_flag" in prediction_rows.columns else None,
                        day=day,
                        logger=logger,
                    )
                prediction_output["D"] = "-"
                prediction_output = prediction_output[
                    ["SOURCE_RISK", "Loan_number", "SOURCE_MONTH_USED", "MONTH", *SCHEDULE_DAY_COLUMNS]
                ]
                prediction_output.to_csv(prediction_file, index=False)
            else:
                pd.DataFrame(
                    columns=["SOURCE_RISK", "Loan_number", "SOURCE_MONTH_USED", "MONTH", *SCHEDULE_DAY_COLUMNS]
                ).to_csv(prediction_file, index=False)
            logger.info("Saved predictions | path=%s bytes=%s", prediction_file, prediction_file.stat().st_size)

        print(f"Training rows: {len(train_df):,}")
        print(f"Validation rows: {len(validation_df):,}")
        print(f"Test rows: {len(test_df):,}")
        print(f"Prediction rows: {len(prediction_rows):,}")
        print(f"Saved model to {model_file} (versioned copy: {versioned_model_file})")
        print(f"Saved metrics to {metrics_file} (versioned copy: {versioned_metrics_file})")
        print(f"Saved future predictions to {prediction_file}")
        logger.info("CatBoost training completed successfully.")
    except Exception:
        logger.exception("CatBoost training failed.")
        raise


if __name__ == "__main__":
    main()
