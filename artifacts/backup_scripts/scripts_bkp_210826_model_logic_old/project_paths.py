from __future__ import annotations

from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

DATA_DIR = REPO_ROOT / "data"
COMMUNICATION_DATA_DIR = DATA_DIR / "communication"
CASE_DATA_DIR = DATA_DIR / "cases"
TRAINING_DATA_DIR = DATA_DIR / "training"
SCHEDULE_DATA_DIR = DATA_DIR / "schedules"
FEATURE_DATA_DIR = DATA_DIR / "features"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
CATBOOST_INFO_DIR = ARTIFACTS_DIR / "catboost_info"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
LOG_DIR = ARTIFACTS_DIR / "logs"

REPORTS_DIR = REPO_ROOT / "reports"
BENCHMARK_DIR = REPORTS_DIR / "benchmarks"


def ensure_parent_dir(path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path
