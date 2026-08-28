from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-6


def _safe_fraction(series: pd.Series, categories: list[object]) -> np.ndarray:
    counts = series.value_counts(dropna=False)
    total = float(len(series))
    if total == 0:
        return np.full(len(categories), 1.0 / max(len(categories), 1))
    values = np.array([counts.get(category, 0.0) / total for category in categories], dtype=float)
    return np.clip(values, EPSILON, None)


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    categories = sorted(set(expected.dropna().tolist()) | set(actual.dropna().tolist()))
    if not categories:
        return 0.0
    expected_pct = _safe_fraction(expected, categories)
    actual_pct = _safe_fraction(actual, categories)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _numeric_bins(expected: pd.Series, bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected.quantile(quantiles).to_numpy(dtype=float))
    if len(breakpoints) < 3:
        min_value = float(expected.min())
        max_value = float(expected.max())
        if math.isclose(min_value, max_value):
            return np.array([-np.inf, max_value, np.inf], dtype=float)
        breakpoints = np.linspace(min_value, max_value, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    return breakpoints


def _numeric_psi(expected: pd.Series, actual: pd.Series, bins: int) -> float:
    expected_numeric = pd.to_numeric(expected, errors="coerce").dropna()
    actual_numeric = pd.to_numeric(actual, errors="coerce").dropna()
    if expected_numeric.empty or actual_numeric.empty:
        return 0.0
    breakpoints = _numeric_bins(expected_numeric, bins)
    expected_bucket = pd.cut(expected_numeric, breakpoints, include_lowest=True)
    actual_bucket = pd.cut(actual_numeric, breakpoints, include_lowest=True)
    categories = list(expected_bucket.cat.categories)
    expected_pct = _safe_fraction(expected_bucket, categories)
    actual_pct = _safe_fraction(actual_bucket, categories)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def calculate_feature_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.fillna(0)
    actual = actual.fillna(0)
    combined = pd.concat([expected, actual], ignore_index=True)
    unique_count = int(combined.nunique(dropna=False))
    is_discrete = pd.api.types.is_bool_dtype(combined) or (
        unique_count <= 10 and pd.api.types.is_numeric_dtype(combined)
    )
    if is_discrete or pd.api.types.is_object_dtype(combined):
        return _categorical_psi(expected.astype(str), actual.astype(str))
    return _numeric_psi(expected, actual, bins=bins)


def classify_psi(psi: float) -> str:
    if psi >= 0.25:
        return "severe"
    if psi >= 0.10:
        return "moderate"
    return "stable"


def compute_drift_report(
    *,
    baseline_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    bins: int = 10,
    blank_inference_rows: int = 0,
) -> dict[str, Any]:
    columns = feature_columns or sorted(set(baseline_df.columns) | set(inference_df.columns))
    baseline = baseline_df.reindex(columns=columns, fill_value=0)
    inference = inference_df.reindex(columns=columns, fill_value=0)

    features: list[dict[str, Any]] = []
    for column in columns:
        psi = calculate_feature_psi(baseline[column], inference[column], bins=bins)
        features.append(
            {
                "feature": column,
                "psi": round(float(psi), 6),
                "severity": classify_psi(float(psi)),
                "baseline_mean": round(float(pd.to_numeric(baseline[column], errors="coerce").fillna(0).mean()), 6),
                "inference_mean": round(float(pd.to_numeric(inference[column], errors="coerce").fillna(0).mean()), 6),
                "baseline_non_zero_rate": round(float((pd.to_numeric(baseline[column], errors="coerce").fillna(0) != 0).mean()), 6),
                "inference_non_zero_rate": round(float((pd.to_numeric(inference[column], errors="coerce").fillna(0) != 0).mean()), 6),
            }
        )

    psi_values = [feature["psi"] for feature in features]
    moderate_count = sum(1 for feature in features if feature["severity"] in {"moderate", "severe"})
    severe_count = sum(1 for feature in features if feature["severity"] == "severe")
    feature_count = len(features)
    overall_psi = round(float(np.mean(psi_values)) if psi_values else 0.0, 6)
    max_feature_psi = round(float(np.max(psi_values)) if psi_values else 0.0, 6)
    drift_percentage = round((moderate_count / feature_count) * 100, 2) if feature_count else 0.0

    return {
        "baseline_rows": int(len(baseline)),
        "inference_rows": int(len(inference)),
        "blank_inference_rows": int(blank_inference_rows),
        "feature_count": int(feature_count),
        "overall_psi": overall_psi,
        "max_feature_psi": max_feature_psi,
        "moderate_feature_count": int(moderate_count),
        "severe_feature_count": int(severe_count),
        "drift_percentage": drift_percentage,
        "status": "severe" if severe_count else ("moderate" if moderate_count else "stable"),
        "feature_metrics": sorted(features, key=lambda item: (-item["psi"], item["feature"])),
    }
