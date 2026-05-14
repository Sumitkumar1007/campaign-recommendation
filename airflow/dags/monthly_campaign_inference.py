from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from dateutil.relativedelta import relativedelta


PROJECT_DIR = "/home/ubuntu/aiml/recommendation"
CONFIG_FILE = f"{PROJECT_DIR}/config/monthly_inference_airflow.json"


def normalize_month(value: str) -> str:
    raw_value = value.strip().upper().replace("-", "")
    for fmt in ("%Y%m", "%b%Y"):
        try:
            return datetime.strptime(raw_value, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    raise ValueError(f"Invalid month value {value!r}. Use YYYY-MM or MONYYYY, for example 2026-04 or APR2026.")


@dag(
    dag_id="monthly_campaign_inference",
    description="Run campaign recommendation monthly inference pipeline.",
    schedule="0 2 6 * *",
    start_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["campaign", "recommendation", "monthly-inference"],
)
def monthly_campaign_inference():
    @task
    def resolve_months() -> dict[str, str]:
        config_path = Path(CONFIG_FILE)
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return {
                "source_month": normalize_month(config["source_month"]),
                "predict_month": normalize_month(config["predict_month"]),
            }

        today = datetime.now(timezone.utc).date().replace(day=1)
        return {
            "source_month": today.strftime("%Y-%m"),
            "predict_month": (today + relativedelta(months=1)).strftime("%Y-%m"),
        }

    months = resolve_months()

    BashOperator(
        task_id="run_monthly_inference",
        cwd=PROJECT_DIR,
        bash_command=(
            "set -euo pipefail\n"
            "echo \"Running monthly inference: "
            "source={{ ti.xcom_pull(task_ids='resolve_months')['source_month'] }} "
            "predict={{ ti.xcom_pull(task_ids='resolve_months')['predict_month'] }}\"\n"
            "./venv/bin/python scripts/run_monthly_inference_pipeline.py "
            "--source-month {{ ti.xcom_pull(task_ids='resolve_months')['source_month'] }} "
            "--predict-month {{ ti.xcom_pull(task_ids='resolve_months')['predict_month'] }}"
        ),
    )

    months


monthly_campaign_inference()
