from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from drift_utils import calculate_feature_psi, classify_psi, compute_drift_report  # noqa: E402


def test_feature_psi_stable_when_distributions_match() -> None:
    expected = pd.Series([0, 0, 1, 1, 0, 1])
    actual = pd.Series([1, 0, 0, 1, 0, 1])
    assert calculate_feature_psi(expected, actual) < 0.05


def test_feature_psi_increases_for_shifted_distribution() -> None:
    expected = pd.Series([0] * 90 + [1] * 10)
    actual = pd.Series([0] * 20 + [1] * 80)
    assert calculate_feature_psi(expected, actual) > 0.25


def test_compute_drift_report_summarizes_feature_levels() -> None:
    baseline = pd.DataFrame(
        {
            "a": [0, 0, 0, 1, 1],
            "b": [10, 11, 12, 13, 14],
        }
    )
    inference = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 1],
            "b": [50, 51, 52, 53, 54],
        }
    )
    report = compute_drift_report(baseline_df=baseline, inference_df=inference, blank_inference_rows=2)
    assert report["baseline_rows"] == 5
    assert report["inference_rows"] == 5
    assert report["blank_inference_rows"] == 2
    assert report["feature_count"] == 2
    assert report["drift_percentage"] > 0
    assert report["feature_metrics"][0]["severity"] in {"moderate", "severe"}
    assert classify_psi(report["max_feature_psi"]) in {"moderate", "severe"}
