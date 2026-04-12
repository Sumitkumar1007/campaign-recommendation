from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DAY_AVAILABILITY_COLUMNS, HISTORY_FEATURE_COLUMNS
from .features import get_model_matrix


NUMERIC_FEATURES = [
    "day_offset",
    "send_hour",
    "collectable_amount",
    "emi_day",
    "emi_month_number",
    "is_predue",
    "is_postdue",
    "prev_month_same_day_available",
    *HISTORY_FEATURE_COLUMNS,
    *DAY_AVAILABILITY_COLUMNS,
]
CATEGORICAL_FEATURES = ["risk", "communication_type", "campaign_identifier"]


@dataclass
class CalibratedCampaignModel:
    base_model: Pipeline
    calibrator: LogisticRegression

    def predict_proba(self, X):
        raw_probabilities = self.raw_predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])

    def raw_predict_proba(self, X):
        return self.base_model.predict_proba(X)


def build_model(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=None,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _metrics_from_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def _evaluate_split(model: CalibratedCampaignModel, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X = get_model_matrix(frame)
    y = frame["target_success"].astype(int)
    raw_probabilities = model.raw_predict_proba(X)[:, 1]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "raw": _metrics_from_probabilities(y, raw_probabilities),
        "calibrated": _metrics_from_probabilities(y, probabilities),
    }

    evaluation_frame = frame[
        [
            "apac_card_number",
            "emi_date",
            "emi_month",
            "risk",
            "communication_type",
            "day_offset",
            "send_hour",
            "dataset_split",
            "target_success",
        ]
    ].copy()
    evaluation_frame["raw_success_probability"] = raw_probabilities
    evaluation_frame["predicted_success_probability"] = probabilities
    evaluation_frame["predicted_success_label"] = predictions
    return metrics, evaluation_frame


def _fit_calibrator(base_model: Pipeline, validation_frame: pd.DataFrame) -> LogisticRegression:
    X_validation = get_model_matrix(validation_frame)
    y_validation = validation_frame["target_success"].astype(int).to_numpy()
    raw_probabilities = base_model.predict_proba(X_validation)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=200, random_state=42)
    calibrator.fit(raw_probabilities, y_validation)
    return calibrator


def train_model(df: pd.DataFrame, random_state: int) -> tuple[CalibratedCampaignModel, dict, pd.DataFrame]:
    train_frame = df[df["dataset_split"] == "train"].copy()
    validation_frame = df[df["dataset_split"] == "validation"].copy()
    test_frame = df[df["dataset_split"] == "test"].copy()

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    X_train = get_model_matrix(train_frame)
    y_train = train_frame["target_success"].astype(int)

    base_model = build_model(random_state=random_state)
    base_model.fit(X_train, y_train)

    calibrator = _fit_calibrator(base_model=base_model, validation_frame=validation_frame)
    model = CalibratedCampaignModel(base_model=base_model, calibrator=calibrator)

    validation_metrics, validation_eval = _evaluate_split(model, validation_frame)
    test_metrics, test_eval = _evaluate_split(model, test_frame)

    metrics = {
        "row_count": int(len(df)),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_positive_rate": float(y_train.mean()),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    evaluation_frame = pd.concat([validation_eval, test_eval], ignore_index=True)
    return model, metrics, evaluation_frame


def save_artifacts(model: CalibratedCampaignModel, metrics: dict, output_dir: str | Path) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_dir / "model.joblib")
    with (target_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_model(model_dir: str | Path) -> CalibratedCampaignModel:
    return joblib.load(Path(model_dir) / "model.joblib")
