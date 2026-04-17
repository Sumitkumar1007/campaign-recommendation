from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas.api.types import is_object_dtype, is_string_dtype

from project_paths import METRICS_DIR, MODEL_DIR, TRAINING_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multiclass model to predict PREDICTED_STRATEGY."
    )
    parser.add_argument(
        "--input-file",
        default=str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
        help="Wide strategy dataset CSV.",
    )
    parser.add_argument(
        "--model-file",
        default=str(MODEL_DIR / "strategy_model.joblib"),
        help="Path to save the trained model bundle.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_DIR / "strategy_model_metrics.json"),
        help="Path to save evaluation metrics.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for test evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_training_data(input_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(input_file)
    if "PREDICTED_STRATEGY" not in df.columns:
        raise ValueError("Input file does not contain PREDICTED_STRATEGY.")

    df = df[df["PREDICTED_STRATEGY"].notna()].copy()
    df = df[df["PREDICTED_STRATEGY"].astype(str).str.strip().ne("")].copy()
    if df.empty:
        raise ValueError("No non-empty PREDICTED_STRATEGY rows found for training.")

    y = df["PREDICTED_STRATEGY"].astype(str)
    X = df.drop(columns=["PREDICTED_STRATEGY"])
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_columns = [
        column
        for column in X.columns
        if is_object_dtype(X[column]) or is_string_dtype(X[column])
    ]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ]
                ),
                numeric_columns,
            ),
        ],
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    model_file = ensure_parent_dir(args.model_file)
    metrics_file = ensure_parent_dir(args.metrics_file)

    X, y = load_training_data(input_file)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = pd.Series(y_encoded).value_counts()
    stratify_target = (
        y_encoded
        if len(class_counts) > 1 and int(class_counts.min()) >= 2
        else None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_target,
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)
    fitted_classes = pipeline.named_steps["classifier"].classes_
    known_test_mask = pd.Series(y_test).isin(fitted_classes).to_numpy()
    if known_test_mask.any():
        top_3_accuracy = float(
            top_k_accuracy_score(
                y_test[known_test_mask],
                test_proba[known_test_mask],
                k=min(3, len(fitted_classes)),
                labels=fitted_classes,
            )
        )
    else:
        top_3_accuracy = None

    report_labels = np.unique(np.concatenate([y_test, test_pred]))
    report_target_names = label_encoder.inverse_transform(report_labels)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "class_count": int(len(label_encoder.classes_)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "top_3_accuracy": top_3_accuracy,
        "known_test_rows_for_topk": int(known_test_mask.sum()),
        "classes": label_encoder.classes_.tolist(),
        "classification_report": classification_report(
            y_test,
            test_pred,
            labels=report_labels,
            target_names=report_target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    model_bundle = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(model_bundle, model_file)
    metrics_file.write_text(json.dumps(metrics, indent=2))

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Classes: {len(label_encoder.classes_):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Saved model to {model_file}")
    print(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()
