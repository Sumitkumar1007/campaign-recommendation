from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow

from app_logging import log_step, setup_logging
from mlflow_catboost_strategy_model import CatBoostStrategyPyfuncModel
from project_paths import ARTIFACTS_DIR, METRICS_DIR, MODEL_DIR, REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log and register the saved 3-month CatBoost strategy model in MLflow."
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", str(ARTIFACTS_DIR / "mlruns")),
        help="MLflow tracking URI. Use a local path, file URI, or remote tracking server URL.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "campaign-recommendation"),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "campaign_next_month_catboost_3m"),
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--run-name",
        default=os.getenv("MLFLOW_RUN_NAME", "catboost-3m-may-2026-v1"),
        help="MLflow run name.",
    )
    parser.add_argument(
        "--model-file",
        default=str(MODEL_DIR / "next_month_strategy_catboost_3m.joblib"),
        help="Saved CatBoost joblib bundle.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(METRICS_DIR / "next_month_strategy_catboost_3m_metrics.json"),
        help="Training metrics JSON file.",
    )
    parser.add_argument(
        "--extra-artifact",
        action="append",
        default=[
            str(REPO_ROOT / "artifacts" / "analysis" / "may_2026_all_blank_27_apac_analysis.md"),
            str(REPO_ROOT / "artifacts" / "analysis" / "may_2026_all_blank_27_apac_analysis.csv"),
            str(REPO_ROOT / "docs" / "three_month_catboost_training_explainer.md"),
        ],
        help="Extra file to attach to the MLflow run. Can be used multiple times.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Application log file. Defaults to artifacts/logs/mlflow_register_catboost.log.",
    )
    return parser.parse_args()


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, metric_name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[metric_name] = float(value)
    return flattened


def log_params_from_metrics(metrics: dict[str, Any]) -> None:
    param_keys = [
        "assumption",
        "target_offset_months",
        "history_window_months",
        "train_source_months",
        "validation_source_months",
        "test_source_months",
        "prediction_source_months",
        "n_jobs",
        "iterations",
        "learning_rate",
        "depth",
    ]
    for key in param_keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            mlflow.log_param(key, value)


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file, "mlflow_register_catboost")
    logger.info("MLflow registration args: %s", vars(args))

    model_file = Path(args.model_file)
    metrics_file = Path(args.metrics_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with metrics_file.open() as file:
        metrics = json.load(file)

    with log_step(logger, "configure_mlflow", tracking_uri=args.tracking_uri):
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        logger.info("Started MLflow run | run_id=%s", run.info.run_id)
        mlflow.set_tags(
            {
                "project": "campaign-recommendation",
                "model_family": "catboost",
                "history_window_months": str(metrics.get("history_window_months", "unknown")),
                "target": "next_month_day_strategy",
            }
        )
        log_params_from_metrics(metrics)
        for name, value in flatten_metrics(metrics).items():
            mlflow.log_metric(name.replace("+", "plus").replace("-", "minus"), value)

        mlflow.log_artifact(str(metrics_file), artifact_path="metrics")
        for artifact in args.extra_artifact:
            artifact_path = Path(artifact)
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="supporting_artifacts")
            else:
                logger.warning("Skipping missing extra artifact: %s", artifact_path)

        with log_step(logger, "log_pyfunc_model", model_file=model_file):
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CatBoostStrategyPyfuncModel(),
                artifacts={"model_bundle": str(model_file)},
                code_paths=[str(REPO_ROOT / "scripts" / "mlflow_catboost_strategy_model.py")],
                registered_model_name=args.registered_model_name,
                pip_requirements=[
                    "mlflow>=2.12,<3",
                    "pandas>=2.2,<3",
                    "numpy>=1.26,<3",
                    "joblib>=1.4,<2",
                    "catboost>=1.2,<2",
                    "scikit-learn>=1.5,<2",
                ],
            )

        model_uri = f"runs:/{run.info.run_id}/model"
        logger.info("MLflow model logged | model_uri=%s", model_uri)
        print(f"MLflow run_id: {run.info.run_id}")
        print(f"MLflow model_uri: {model_uri}")
        print(f"Registered model: {args.registered_model_name}")


if __name__ == "__main__":
    main()
