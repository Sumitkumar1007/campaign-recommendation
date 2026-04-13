from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import get_model_matrix


def explain_prediction(model, candidate_row: pd.DataFrame, top_k: int = 10) -> list[dict[str, Any]]:
    pipeline = getattr(model, "base_model", model)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    matrix = get_model_matrix(candidate_row)
    transformed = preprocessor.transform(matrix)
    coefficients = classifier.coef_[0]
    values = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    contributions = values * coefficients
    feature_names = preprocessor.get_feature_names_out()

    ranked = sorted(
        zip(feature_names, contributions),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    explanation: list[dict[str, Any]] = []
    for name, value in ranked[:top_k]:
        explanation.append(
            {
                "feature": str(name),
                "contribution": float(value),
                "direction": "up" if value >= 0 else "down",
            }
        )
    return explanation


def build_validation_slice_report(evaluation_frame: pd.DataFrame) -> pd.DataFrame:
    frame = evaluation_frame.copy()
    frame["predicted_positive"] = (frame["predicted_success_probability"] >= 0.5).astype(int)
    grouped = (
        frame.groupby(["dataset_split", "risk", "communication_type", "day_offset"], dropna=False)
        .agg(
            rows=("target_success", "size"),
            actual_success_rate=("target_success", "mean"),
            actual_success_score=("target_success_score", "mean"),
            predicted_success_rate=("predicted_success_probability", "mean"),
            predicted_positive_rate=("predicted_positive", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["dataset_split", "risk", "communication_type", "day_offset"]
    ).reset_index(drop=True)


def build_calibration_report(evaluation_frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    frame = evaluation_frame.copy()
    labels = [f"bin_{idx}" for idx in range(bins)]
    frame["score_bin"] = pd.cut(
        frame["predicted_success_probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        labels=labels,
    )
    report = (
        frame.groupby(["dataset_split", "score_bin"], dropna=False, observed=False)
        .agg(
            rows=("target_success", "size"),
            avg_predicted_probability=("predicted_success_probability", "mean"),
            actual_success_rate=("target_success", "mean"),
            avg_raw_probability=("raw_success_probability", "mean"),
        )
        .reset_index()
    )
    return report
