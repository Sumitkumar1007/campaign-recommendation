from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..data import build_modeling_dataset
from ..features import get_model_matrix
from ..modeling import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from ..settings import AppConfig


@dataclass
class ExperimentModel:
    name: str
    base_model: Pipeline
    calibrator: LogisticRegression

    def raw_predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict_proba(self, X):
        raw = self.raw_predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])


def _build_sparse_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
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


def _build_dense_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _candidate_pipelines(random_state: int) -> list[tuple[str, Pipeline]]:
    sparse_preprocessor = _build_sparse_preprocessor()
    dense_preprocessor = _build_dense_preprocessor()

    return [
        (
            "logistic_baseline",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=500,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("preprocessor", dense_preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=12,
                            min_samples_leaf=10,
                            random_state=random_state,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees",
            Pipeline(
                steps=[
                    ("preprocessor", dense_preprocessor),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_depth=12,
                            min_samples_leaf=10,
                            random_state=random_state,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                steps=[
                    ("preprocessor", dense_preprocessor),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=0.06,
                            max_depth=8,
                            max_iter=250,
                            min_samples_leaf=30,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "adaboost",
            Pipeline(
                steps=[
                    ("preprocessor", dense_preprocessor),
                    (
                        "classifier",
                        AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
                            n_estimators=200,
                            learning_rate=0.05,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return get_model_matrix(df)


def _fit_calibrator(model: Pipeline, validation_frame: pd.DataFrame) -> LogisticRegression:
    X_validation = _get_feature_frame(validation_frame)
    y_validation = validation_frame["target_success"].astype(int).to_numpy()
    raw_probabilities = model.predict_proba(X_validation)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=200, random_state=42)
    calibrator.fit(raw_probabilities, y_validation)
    return calibrator


def _metric_dict(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def _evaluate_experiment(model: ExperimentModel, frame: pd.DataFrame) -> dict:
    X = _get_feature_frame(frame)
    y = frame["target_success"].astype(int)
    raw_probs = model.raw_predict_proba(X)[:, 1]
    calibrated_probs = model.predict_proba(X)[:, 1]
    results = {
        "raw": _metric_dict(y, raw_probs),
        "calibrated": _metric_dict(y, calibrated_probs),
        "average_success_score": float(frame["target_success_score"].mean()),
    }
    return results


def run_model_benchmarks(config: AppConfig, max_rows: int | None = None) -> dict:
    training_cfg = config.raw["training"]
    effective_max_rows = max_rows or training_cfg["max_rows"]
    modeling_frame, split_meta = build_modeling_dataset(
        csv_path=config.training_data_path,
        usecols=training_cfg["usecols"],
        chunk_size=training_cfg["chunk_size"],
        max_rows=effective_max_rows,
        allowed_day_offsets=config.raw["allowed_day_offsets"],
        success_statuses=config.raw["success_statuses"],
        success_scores=config.raw["success_scores"],
        positive_boost=training_cfg["sample_weight_positive_boost"],
        emi_cycles=config.raw["emi_cycle"],
    )

    train_frame = modeling_frame[modeling_frame["dataset_split"] == "train"].copy()
    validation_frame = modeling_frame[modeling_frame["dataset_split"] == "validation"].copy()
    test_frame = modeling_frame[modeling_frame["dataset_split"] == "test"].copy()

    X_train = _get_feature_frame(train_frame)
    y_train = train_frame["target_success"].astype(int)
    train_weights = train_frame["target_sample_weight"].astype(float).to_numpy()

    rows: list[dict] = []
    experiment_details: dict[str, dict] = {}
    for model_name, pipeline in _candidate_pipelines(training_cfg["random_state"]):
        fit_kwargs = {}
        if model_name in {"logistic_baseline", "random_forest", "extra_trees"}:
            fit_kwargs["classifier__sample_weight"] = train_weights
        if model_name == "adaboost":
            fit_kwargs["classifier__sample_weight"] = train_weights

        pipeline.fit(X_train, y_train, **fit_kwargs)
        calibrator = _fit_calibrator(pipeline, validation_frame)
        experiment = ExperimentModel(name=model_name, base_model=pipeline, calibrator=calibrator)
        validation_metrics = _evaluate_experiment(experiment, validation_frame)
        test_metrics = _evaluate_experiment(experiment, test_frame)
        experiment_details[model_name] = {
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

        for split_name, split_metrics in [("validation", validation_metrics), ("test", test_metrics)]:
            for scope in ["raw", "calibrated"]:
                scoped = split_metrics[scope]
                for metric_name, metric_value in scoped.items():
                    rows.append(
                        {
                            "model_name": model_name,
                            "split": split_name,
                            "scope": scope,
                            "metric": metric_name,
                            "value": metric_value,
                        }
                    )

    benchmark_frame = pd.DataFrame(rows).sort_values(
        ["split", "scope", "metric", "value"], ascending=[True, True, True, False]
    )
    reports_dir = config.root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "model_benchmark_experiments.csv"
    json_path = reports_dir / "model_benchmark_experiments.json"
    md_path = reports_dir / "model_benchmark_experiments.md"
    benchmark_frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_meta": split_meta,
                "training_rows_used": int(len(modeling_frame)),
                "experiments": experiment_details,
            },
            handle,
            indent=2,
        )

    summary_lines = [
        "# Model Benchmark Experiments",
        "",
        "Compared models on the same month-based split and the same training sample.",
        "",
        "Split setup:",
        f"- Train: {', '.join(split_meta['train_months'])}",
        f"- Validation: {', '.join(split_meta['validation_months'])}",
        f"- Test: {', '.join(split_meta['test_months'])}",
        f"- Training rows used: {len(modeling_frame)}",
        "",
        "Models benchmarked:",
        "- logistic_baseline",
        "- random_forest",
        "- extra_trees",
        "- hist_gradient_boosting",
        "- adaboost",
        "",
        f"Detailed metrics CSV: `{csv_path.name}`",
        f"Detailed metrics JSON: `{json_path.name}`",
    ]
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "benchmark_csv": str(csv_path),
        "benchmark_json": str(json_path),
        "benchmark_markdown": str(md_path),
        "split_meta": split_meta,
        "training_rows_used": int(len(modeling_frame)),
    }
