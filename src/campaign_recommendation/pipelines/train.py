from __future__ import annotations

import json
from pathlib import Path

from ..data import build_base_population, build_modeling_dataset
from ..explain import build_calibration_report, build_validation_slice_report
from ..modeling import save_artifacts, train_model
from ..settings import AppConfig


def run_training(config: AppConfig, max_rows: int | None = None) -> dict:
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
    )

    model, metrics, evaluation_frame = train_model(
        df=modeling_frame,
        random_state=training_cfg["random_state"],
    )
    save_artifacts(model=model, metrics=metrics, output_dir=config.model_output_dir)

    evaluation_path = Path(config.model_output_dir) / "evaluation_sample.csv"
    evaluation_frame.head(5000).to_csv(evaluation_path, index=False)
    slice_report = build_validation_slice_report(evaluation_frame)
    slice_report.to_csv(Path(config.model_output_dir) / "validation_slice_report.csv", index=False)
    calibration_report = build_calibration_report(evaluation_frame)
    calibration_report.to_csv(Path(config.model_output_dir) / "calibration_report.csv", index=False)

    base_population = build_base_population(modeling_frame)
    base_population_path = Path(config.model_output_dir) / "training_base_population_sample.csv"
    base_population.head(5000).to_csv(base_population_path, index=False)

    run_summary = {
        "metrics": metrics,
        "split_meta": split_meta,
        "model_output_dir": str(config.model_output_dir),
        "training_rows_used": int(len(modeling_frame)),
        "base_population_rows": int(len(base_population)),
    }
    with (Path(config.model_output_dir) / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    return run_summary
