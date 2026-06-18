from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from psycopg import errors, sql
from psycopg.types.json import Jsonb

from app_logging import log_step, setup_logging
from drift_utils import compute_drift_report
from env_utils import load_dotenv
from pipeline_common import build_feature_matrix, prepare_next_month_dataset, resolve_emi_cycle, split_by_source_month
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from predict_next_month_strategy_catboost import build_prediction_population, load_base_population
from project_paths import (
    CASE_DATA_DIR,
    COMMUNICATION_DATA_DIR,
    FEATURE_DATA_DIR,
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    SCRIPTS_DIR,
    TRAINING_DATA_DIR,
)


DAY_COLUMNS = ["D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
PREDUE_DAYS = ["D-5", "D-4", "D-3", "D-2", "D-1"]
POSTDUE_DAYS = ["D+1", "D+2", "D+3", "D+4", "D+5"]
CAMPAIGN_DAY_ORDER = {day: index for index, day in enumerate([*PREDUE_DAYS, *POSTDUE_DAYS])}
RISK_CODES = {
    "LOW": "LR",
    "MEDIUM": "MR",
    "HIGH": "HR",
}
MODE_BY_STRATEGY_CHANNEL = {
    "SMS": "SMS",
    "WH": "WHATSAPP",
    "WHATSAPP": "WHATSAPP",
    "IVR": "VOICE",
    "VOICE": "VOICE",
}
NAME_CHANNEL_BY_MODE = {
    "SMS": "SMS",
    "WHATSAPP": "WA",
    "VOICE": "VOICE",
}
VENDOR_CONFIG_KEY = "voice.service.vendor-list"
EMI_DATES_CONFIG_KEY = "upload.scheduler.emi-dates"


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=(
            "Incremental monthly inference pipeline: fetch one source month from "
            "Postgres, rebuild local processed data, run inference, and store "
            "prediction and campaign outputs back to Postgres."
        )
    )
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--source-schema",
        default=os.getenv("SOURCE_SCHEMA", "digital_collections"),
        help="Schema containing the source communications table.",
    )
    parser.add_argument(
        "--source-table",
        default=os.getenv("SOURCE_TABLE", "communications"),
        help="Source communications table.",
    )
    parser.add_argument(
        "--target-schema",
        default=os.getenv("TARGET_SCHEMA", "digital_collections"),
        help="Schema used for storing predictions and campaign outputs.",
    )
    parser.add_argument(
        "--prediction-table",
        default=os.getenv("PREDICTION_TABLE", "ai_ml_recommendations_data"),
        help="Target table for model prediction snapshots.",
    )
    parser.add_argument(
        "--campaign-table",
        default=os.getenv("CAMPAIGN_TABLE", "ai_ml_campaign_recommendations"),
        help="Target table for campaign scheduler recommendations.",
    )
    parser.add_argument(
        "--campaign-mapping-table",
        default=os.getenv("CAMPAIGN_MAPPING_TABLE", "ai_ml_campaign_mapping"),
        help="Target table for campaign-to-account mapping rows.",
    )
    parser.add_argument(
        "--campaign-vertical",
        default=os.getenv("CAMPAIGN_VERTICAL", "LAP"),
        help="Campaign vertical used in scheduler naming until a source column is available.",
    )
    parser.add_argument(
        "--campaign-vendor",
        default=os.getenv("CAMPAIGN_VENDOR", "prutech-cpass"),
        help="Fallback campaign vendor when data_config vendor lookup is unavailable.",
    )
    parser.add_argument(
        "--config-file",
        default=os.getenv("CONFIG_FILE", str(SCRIPTS_DIR.parent / "config" / "default_config.json")),
        help="JSON config file containing emi_cycle.",
    )
    parser.add_argument(
        "--source-month",
        default=os.getenv("SOURCE_MONTH") or current_month(),
        help="Source month to process in YYYY-MM format. Defaults to current month.",
    )
    parser.add_argument(
        "--predict-month",
        default=os.getenv("PREDICT_MONTH"),
        help="Target month to predict in YYYY-MM format. Defaults to one month after source month.",
    )
    parser.add_argument(
        "--model",
        choices=["catboost", "catboost_3m", "logistic"],
        default=os.getenv("MODEL_NAME", "catboost_3m"),
        help="Which next-month model pipeline to run.",
    )
    parser.add_argument(
        "--model-serving",
        choices=["local", "mlflow"],
        default=os.getenv("MODEL_SERVING", os.getenv("modelserving", "local")).lower(),
        help="Load model from local artifacts or download the bundle from MLflow registry.",
    )
    parser.add_argument(
        "--mlflow-model-uri",
        default=os.getenv(
            "MLFLOW_MODEL_URI",
            f"models:/{os.getenv('MLFLOW_REGISTERED_MODEL_NAME', 'campaign_next_month_catboost_3m')}@production",
        ),
        help="MLflow model URI used when --model-serving=mlflow.",
    )
    parser.add_argument(
        "--feature-month-source",
        choices=["emi_date", "created_date"],
        default=os.getenv("FEATURE_MONTH_SOURCE", "emi_date"),
        help="Date field used to assign monthly feature labels after fetch.",
    )
    parser.add_argument(
        "--skip-db-store",
        action="store_true",
        help="Run the local fetch/process/predict flow without storing snapshots back to Postgres.",
    )
    parser.add_argument(
        "--log-file",
        default=str(LOG_DIR / "monthly_inference_pipeline.log"),
        help="Application log file.",
    )
    args = parser.parse_args()
    missing = [
        name
        for name, value in {
            "PGHOST/--host": args.host,
            "PGDATABASE/--dbname": args.dbname,
            "PGUSER/--user": args.user,
            "PGPASSWORD/--password": args.password,
        }.items()
        if not value
    ]
    if missing:
        parser.error("Missing required environment variables or CLI args: " + ", ".join(missing))
    if not args.predict_month:
        args.predict_month = next_month(args.source_month)
    try:
        validate_month_pair(args.source_month, args.predict_month)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_month(yyyy_mm: str) -> pd.Period:
    if len(yyyy_mm) != 7 or yyyy_mm[4] != "-":
        raise ValueError(f"Invalid month '{yyyy_mm}'. Expected YYYY-MM, for example 2026-05.")
    try:
        return pd.Period(yyyy_mm, freq="M")
    except ValueError as exc:
        raise ValueError(f"Invalid month '{yyyy_mm}'. Expected YYYY-MM, for example 2026-05.") from exc


def current_month(today: pd.Timestamp | None = None) -> str:
    return (today or pd.Timestamp.today()).strftime("%Y-%m")


def next_month(yyyy_mm: str) -> str:
    return str(parse_month(yyyy_mm) + 1)


def validate_month_pair(source_month: str, predict_month: str) -> None:
    source_period = parse_month(source_month)
    predict_period = parse_month(predict_month)
    expected_period = source_period + 1
    if predict_period != expected_period:
        raise ValueError(
            "PREDICT_MONTH/--predict-month must be exactly one month after "
            f"SOURCE_MONTH/--source-month for the current next-month model. "
            f"Got source={source_month}, predict={predict_month}, expected={expected_period}."
        )


def month_label(yyyy_mm: str) -> str:
    return parse_month(yyyy_mm).to_timestamp().strftime("%b-%Y").upper()


def month_file_token(period: pd.Period) -> str:
    return period.to_timestamp().strftime("%b%Y").upper()


def latest_extract_file(source_month: str) -> Path:
    token = month_file_token(parse_month(source_month))
    return COMMUNICATION_DATA_DIR / f"latest_{token}_comm_data.csv"


def monthly_extract_file(month: str) -> Path:
    token = month_file_token(parse_month(month))
    return COMMUNICATION_DATA_DIR / f"mfl_recomm_model_{token}_comm_data.csv"


def current_cases_file(source_month: str) -> Path:
    token = month_file_token(parse_month(source_month))
    return CASE_DATA_DIR / f"digital_cases_{token}.csv"


def selected_history_files(source_month: str, latest_file: Path) -> list[Path]:
    source_period = parse_month(source_month)
    previous_periods = [source_period - 2, source_period - 1]
    files: list[Path] = []
    for period in previous_periods:
        monthly_file = monthly_extract_file(str(period))
        if monthly_file.exists():
            files.append(monthly_file)
    if latest_file.exists():
        files.append(latest_file)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle) > 1


def prediction_summary_path(predict_month: str) -> Path:
    return LOG_DIR / f"{predict_month.replace('-', '_').lower()}_prediction_summary.json"


def _count_top_labels(df: pd.DataFrame, day_columns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for day in day_columns:
        if day not in df.columns:
            continue
        for value in df[day].fillna("-").astype(str):
            for label in value.split("|"):
                label = label.strip()
                if not label or label == "-":
                    continue
                counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _count_daywise_labels(df: pd.DataFrame, day_columns: list[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for day in day_columns:
        if day not in df.columns:
            continue
        counts: dict[str, int] = {}
        for value in df[day].fillna("-").astype(str):
            for label in value.split("|"):
                label = label.strip()
                if not label or label == "-":
                    continue
                counts[label] = counts.get(label, 0) + 1
        output[day] = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return output


def _count_blank_predictions(df: pd.DataFrame, day_columns: list[str]) -> dict[str, object]:
    normalized = df[day_columns].fillna("-").astype(str).apply(lambda col: col.str.strip())
    blank_counts = {
        day: int((normalized[day] == "-").sum())
        for day in day_columns
        if day in normalized.columns
    }
    all_blank_rows = int(normalized.apply(lambda row: all(value == "-" for value in row), axis=1).sum())
    return {
        "blank_counts_by_day": blank_counts,
        "all_blank_rows": all_blank_rows,
    }


def build_prediction_summary(
    *,
    prediction_file: Path,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
    campaign_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> dict[str, object]:
    predictions = pd.read_csv(prediction_file)
    predictions = predictions[predictions["MONTH"] == prediction_month_label].copy()
    day_columns = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]

    source_risk_counts = (
        predictions["SOURCE_RISK"].fillna("UNKNOWN").astype(str).value_counts().sort_index().to_dict()
        if "SOURCE_RISK" in predictions.columns
        else {}
    )
    source_vertical_counts = (
        predictions["SOURCE_VERTICAL"].fillna("UNKNOWN").astype(str).value_counts().sort_index().to_dict()
        if "SOURCE_VERTICAL" in predictions.columns
        else {}
    )
    campaign_mode_counts = (
        campaign_df["mode"].fillna("UNKNOWN").astype(str).value_counts().sort_index().to_dict()
        if not campaign_df.empty
        else {}
    )
    campaign_vendor_counts = (
        campaign_df["vendor"].fillna("UNKNOWN").astype(str).value_counts().sort_index().to_dict()
        if not campaign_df.empty
        else {}
    )
    campaign_vertical_counts = (
        campaign_df["vertical"].fillna("UNKNOWN").astype(str).value_counts().sort_index().to_dict()
        if not campaign_df.empty
        else {}
    )
    mapping_counts_by_campaign = (
        mapping_df["campaign_name"].fillna("UNKNOWN").astype(str).value_counts().head(20).to_dict()
        if not mapping_df.empty
        else {}
    )

    blank_prediction_counts = _count_blank_predictions(predictions, day_columns)

    return {
        "source_month": source_month_label,
        "prediction_month": prediction_month_label,
        "model_name": model_name,
        "prediction_rows": int(len(predictions)),
        "campaign_rows": int(len(campaign_df)),
        "mapping_rows": int(len(mapping_df)),
        "source_risk_counts": source_risk_counts,
        "source_vertical_counts": source_vertical_counts,
        "predicted_label_counts": _count_top_labels(predictions, day_columns),
        "predicted_label_counts_by_day": _count_daywise_labels(predictions, day_columns),
        "blank_prediction_counts": blank_prediction_counts,
        "campaign_mode_counts": campaign_mode_counts,
        "campaign_vendor_counts": campaign_vendor_counts,
        "campaign_vertical_counts": campaign_vertical_counts,
        "top_campaign_mapping_counts": mapping_counts_by_campaign,
    }


def write_prediction_summary(
    *,
    summary: dict[str, object],
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved prediction summary | path=%s", output_path)
    logger.info(
        "Prediction summary counts | prediction_rows=%s campaign_rows=%s mapping_rows=%s",
        summary["prediction_rows"],
        summary["campaign_rows"],
        summary["mapping_rows"],
    )
    return output_path


def fetch_communication_extract(
    args: argparse.Namespace,
    output_file: Path,
    fetch_month: str,
    logger: logging.Logger,
) -> None:
    run_python_script(
        "fetch_month_from_postgres.py",
        "--schema",
        args.source_schema,
        "--table",
        args.source_table,
        "--fetch-month",
        fetch_month,
        "--output-file",
        str(output_file),
        logger=logger,
        env_updates={
            "PGHOST": args.host,
            "PGPORT": str(args.port),
            "PGDATABASE": args.dbname,
            "PGUSER": args.user,
            "PGPASSWORD": args.password,
        },
    )


def fetch_current_cases_extract(
    args: argparse.Namespace,
    output_file: Path,
    fetch_month: str,
    logger: logging.Logger,
) -> None:
    run_python_script(
        "fetch_cases_from_postgres.py",
        "--schema",
        args.source_schema,
        "--fetch-month",
        fetch_month,
        "--output-file",
        str(output_file),
        logger=logger,
        env_updates={
            "PGHOST": args.host,
            "PGPORT": str(args.port),
            "PGDATABASE": args.dbname,
            "PGUSER": args.user,
            "PGPASSWORD": args.password,
        },
    )


def run_python_script(
    script_name: str,
    *script_args: str,
    logger: logging.Logger | None = None,
    env_updates: dict[str, str] | None = None,
) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *script_args]
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    if logger:
        logger.info("Running child script | script=%s args=%s cwd=%s", script_name, list(script_args), SCRIPTS_DIR.parent)
    try:
        result = subprocess.run(cmd, check=True, cwd=SCRIPTS_DIR.parent, env=env, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if logger:
            logger.error("Child script failed | script=%s returncode=%s", script_name, exc.returncode)
            if isinstance(exc.stdout, str) and exc.stdout.strip():
                for line in exc.stdout.strip().splitlines():
                    logger.error("Child script stdout | script=%s | %s", script_name, line)
            else:
                logger.info("Child script stdout empty | script=%s", script_name)
            if isinstance(exc.stderr, str) and exc.stderr.strip():
                for line in exc.stderr.strip().splitlines():
                    logger.error("Child script stderr | script=%s | %s", script_name, line)
            else:
                logger.info("Child script stderr empty | script=%s", script_name)
        raise
    if logger:
        logger.info("Child script completed | script=%s returncode=%s", script_name, result.returncode)
        if isinstance(result.stdout, str) and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                logger.info("Child script stdout | script=%s | %s", script_name, line)
        else:
            logger.info("Child script stdout empty | script=%s", script_name)
        if isinstance(result.stderr, str) and result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                logger.info("Child script stderr | script=%s | %s", script_name, line)
        else:
            logger.info("Child script stderr empty | script=%s", script_name)


def download_mlflow_model_bundle(model_uri: str, target_file: Path, logger: logging.Logger) -> Path:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow model serving requires mlflow. Install project dependencies first.") from exc

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    logger.info("Downloading MLflow model bundle | model_uri=%s target_file=%s", model_uri, target_file)
    local_dir = Path(mlflow.artifacts.download_artifacts(model_uri))
    joblib_files = sorted(local_dir.rglob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"No .joblib model bundle found in downloaded MLflow model: {local_dir}")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(joblib_files[0], target_file)
    logger.info("Downloaded MLflow model bundle | source=%s target=%s", joblib_files[0], target_file)
    return target_file


def ensure_prediction_table(conn, schema: str, table: str) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_ref} (
                loan_number TEXT NOT NULL,
                source_month_used TEXT NOT NULL,
                prediction_month TEXT NOT NULL,
                model_name TEXT NOT NULL,
                source_risk TEXT,
                prediction_payload JSONB NOT NULL,
                prediction_reason JSONB,
                created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (loan_number, prediction_month, model_name)
            )
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )
    conn.execute(
        sql.SQL("ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS prediction_reason JSONB").format(
            table_ref=qualified_identifier(schema, table)
        )
    )


def ensure_campaign_table(conn, schema: str, table: str) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_ref} (
                name TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                template_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                vendor TEXT NOT NULL,
                active TEXT NOT NULL,
                source_month TEXT NOT NULL,
                prediction_month TEXT NOT NULL,
                model_name TEXT NOT NULL,
                emi_cycle INTEGER NOT NULL,
                risk TEXT NOT NULL,
                vertical TEXT NOT NULL,
                campaign_type TEXT NOT NULL,
                due_type TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                modified_at TIMESTAMPTZ NOT NULL,
                created_by TEXT NOT NULL,
                modified_by TEXT NOT NULL
            )
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )


def ensure_campaign_mapping_table(conn, schema: str, table: str) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_ref} (
                campaign_name TEXT NOT NULL,
                loan_number TEXT NOT NULL,
                mode TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                vendor TEXT NOT NULL,
                language TEXT NOT NULL,
                source_month TEXT NOT NULL,
                prediction_month TEXT NOT NULL,
                model_name TEXT NOT NULL,
                emi_cycle INTEGER NOT NULL,
                risk TEXT NOT NULL,
                vertical TEXT NOT NULL,
                campaign_type TEXT NOT NULL,
                due_type TEXT NOT NULL,
                prediction_reason TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                modified_at TIMESTAMPTZ NOT NULL,
                created_by TEXT NOT NULL,
                modified_by TEXT NOT NULL,
                PRIMARY KEY (campaign_name, loan_number)
            )
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )
    conn.execute(
        sql.SQL("ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS prediction_reason TEXT").format(
            table_ref=qualified_identifier(schema, table)
        )
    )
    conn.execute(
        sql.SQL("ALTER TABLE {table_ref} ADD COLUMN IF NOT EXISTS language TEXT").format(
            table_ref=qualified_identifier(schema, table)
        )
    )


def store_prediction_snapshots(
    conn,
    schema: str,
    table: str,
    prediction_file: Path,
    prediction_month_label: str,
    model_name: str,
) -> int:
    df = pd.read_csv(prediction_file)
    df = df[df["MONTH"] == prediction_month_label].copy()
    if df.empty:
        return 0

    ensure_prediction_table(conn, schema, table)
    now = datetime.now(timezone.utc)

    source_risk_col = "SOURCE_RISK" if "SOURCE_RISK" in df.columns else "RISK"
    rows = []
    for _, row in df.iterrows():
        payload = {
            day: row[day]
            for day in ["D-5", "D-4", "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4", "D+5"]
            if day in row.index
        }
        reason_payload = None
        if "PREDICTION_REASON" in row.index and not pd.isna(row["PREDICTION_REASON"]):
            raw_reason = str(row["PREDICTION_REASON"]).strip()
            if raw_reason:
                try:
                    reason_payload = json.loads(raw_reason)
                except json.JSONDecodeError:
                    reason_payload = {"raw": raw_reason}
        rows.append(
            (
                str(row["Loan_number"]),
                str(row["SOURCE_MONTH_USED"]),
                prediction_month_label,
                model_name,
                row.get(source_risk_col),
                Jsonb(payload),
                Jsonb(reason_payload) if reason_payload is not None else None,
                now,
            )
        )

    query = sql.SQL(
        """
        INSERT INTO {table_ref} (
            loan_number,
            source_month_used,
            prediction_month,
            model_name,
            source_risk,
            prediction_payload,
            prediction_reason,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (loan_number, prediction_month, model_name)
        DO UPDATE SET
            source_month_used = EXCLUDED.source_month_used,
            source_risk = EXCLUDED.source_risk,
            prediction_payload = EXCLUDED.prediction_payload,
            prediction_reason = EXCLUDED.prediction_reason,
            created_at = EXCLUDED.created_at
        """
    ).format(table_ref=qualified_identifier(schema, table))
    with conn.cursor() as cur:
        cur.executemany(query, rows)
    return len(rows)


def _format_scheduler_hour(hour_label: str) -> str:
    parsed = pd.to_datetime(hour_label.upper(), format="%I%p", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid strategy hour label: {hour_label!r}")
    return parsed.strftime("%H:00:00")


def _parse_strategy(strategy: str) -> tuple[str, str, str] | None:
    if not strategy or strategy == "-" or pd.isna(strategy):
        return None
    parts = str(strategy).split("-", 2)
    if len(parts) != 3:
        return None
    channel, hour_label, language = parts
    mode = MODE_BY_STRATEGY_CHANNEL.get(channel.upper())
    if mode is None:
        return None
    normalized_language = language.upper().strip()
    if normalized_language == "REGIONAL":
        return None
    return mode, _format_scheduler_hour(hour_label), normalized_language


def _due_bucket(day: str, *, configured_emi_dates: list[str] | None = None) -> tuple[str, str, str]:
    if day in PREDUE_DAYS:
        return "PRE", "PREDUE", day
    if day == "D":
        emi_dates = configured_emi_dates or []
        if emi_dates:
            return "DUE", "DUEDATE", ",".join(emi_dates)
        return "DUE", "DUEDATE", day
    if day in POSTDUE_DAYS:
        return "POST", "POSTDUE", day
    raise ValueError(f"Unsupported campaign day: {day}")


def _scheduler_name(
    *,
    due_type: str,
    mode: str,
    vertical: str,
    language: str,
    emi_cycle: int,
    vendor: str,
    risk_code: str,
    run_token: str,
) -> str:
    channel = NAME_CHANNEL_BY_MODE[mode]
    vendor_token = vendor.upper().replace("-", "_")
    return (
        f"{due_type}_AIML_{channel}_{vertical.upper()}_{language}_{risk_code}_"
        f"{emi_cycle}TH_{vendor_token}_{run_token}"
    )


def _template_name(*, due_type: str, mode: str, language: str) -> str:
    return f"{due_type}_AIML_{NAME_CHANNEL_BY_MODE[mode]}_{language}"


def _dataset_time_label(value: str) -> str:
    labels = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        labels.append(datetime.strptime(part, "%H:%M:%S").strftime("%H"))
    return ",".join(labels)


def _dataset_day_token(day_label: str) -> str:
    day_label = str(day_label).strip().upper()
    if day_label == "D":
        return "D"
    if day_label.startswith("D+"):
        return f"DP{day_label[2:]}"
    if day_label.startswith("D-"):
        return f"DM{day_label[2:]}"
    return day_label.replace(",", "-")


def _dataset_day_label(value: str) -> str:
    return "-".join(_dataset_day_token(part) for part in str(value).split(",") if part.strip())


def _dataset_name(*, due_type: str, mode: str, vertical: str, language: str, risk_code: str, emi_cycle: int, date_value: str, time_value: str) -> str:
    return (
        f"{due_type} AIML {NAME_CHANNEL_BY_MODE[mode]} {vertical.upper()} {language} {risk_code} "
        f"EMI {emi_cycle}TH {_dataset_day_label(date_value)} {_dataset_time_label(time_value)}"
    )


def _normalize_vendor_token(raw_value: object) -> str | None:
    if raw_value is None or pd.isna(raw_value):
        return None

    text = str(raw_value).strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text

    vendor = str(parsed).strip()
    if not vendor:
        return None

    token = vendor.replace("_", "-").split("-", 1)[0].strip().lower()
    if not token:
        return None

    alias_map = {
        "prutech": "prutech",
        "kaleyra": "kaleyra",
        "kaylera": "kaleyra",
    }
    return alias_map.get(token, token)


def _extract_campaign_vendors(raw_value: object) -> list[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        items = raw_value
    elif isinstance(raw_value, dict):
        items = []
        for key in ("vendor", "vendors", "name", "value"):
            if key in raw_value:
                items.append(raw_value[key])
    else:
        if pd.isna(raw_value):
            return []
        text = str(raw_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text

        if isinstance(parsed, (list, dict)):
            return _extract_campaign_vendors(parsed)
        items = [part.strip() for part in str(parsed).split(",") if part.strip()]

    vendors: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, (list, dict)):
            nested = _extract_campaign_vendors(item)
            for vendor in nested:
                if vendor not in seen:
                    seen.add(vendor)
                    vendors.append(vendor)
            continue

        normalized = _normalize_vendor_token(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            vendors.append(normalized)
    return vendors


def resolve_campaign_vendors(
    conn,
    *,
    source_schema: str,
    target_schema: str,
    fallback_vendor: str,
    logger: logging.Logger,
) -> list[str]:
    candidate_refs: list[sql.Composable] = [sql.Identifier("data_config")]
    seen = {"data_config"}
    for schema_name in (source_schema, target_schema):
        key = f"{schema_name}.data_config"
        if schema_name and key not in seen:
            seen.add(key)
            candidate_refs.append(sql.SQL("{}.{}").format(sql.Identifier(schema_name), sql.Identifier("data_config")))

    for table_ref in candidate_refs:
        try:
            row = conn.execute(
                sql.SQL("SELECT value FROM {} WHERE key_name = %s LIMIT 1").format(table_ref),
                (VENDOR_CONFIG_KEY,),
            ).fetchone()
        except errors.UndefinedTable:
            conn.rollback()
            continue
        except Exception:
            logger.exception("Vendor lookup failed. Using fallback vendor.")
            conn.rollback()
            break

        if not row:
            continue

        vendors = _extract_campaign_vendors(row[0])
        if vendors:
            logger.info("Resolved campaign vendors from data_config | key=%s vendors=%s", VENDOR_CONFIG_KEY, vendors)
            return vendors

    fallback_vendors = _extract_campaign_vendors(fallback_vendor)
    if fallback_vendors:
        logger.warning(
            "Using fallback campaign vendors | fallback_vendors=%s key=%s",
            fallback_vendors,
            VENDOR_CONFIG_KEY,
        )
        return fallback_vendors
    logger.warning(
        "Using raw fallback campaign vendor | fallback_vendor=%s key=%s",
        fallback_vendor,
        VENDOR_CONFIG_KEY,
    )
    return [fallback_vendor]


def _join_campaign_days(values: pd.Series) -> str:
    return ",".join(
        sorted(
            set(values),
            key=lambda day: CAMPAIGN_DAY_ORDER.get(day, len(CAMPAIGN_DAY_ORDER)),
        )
    )


def _join_campaign_times(values: pd.Series) -> str:
    return ",".join(sorted(set(values)))


def _coerce_emi_cycle_from_value(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        # Prediction EMI dates are emitted as DD/MM/YYYY; parse that first so
        # multi-cycle runs keep the actual EMI day instead of the month number.
        timestamp = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    except Exception:
        timestamp = pd.NaT
    if pd.isna(timestamp):
        try:
            timestamp = pd.to_datetime(text, dayfirst=True, errors="coerce")
        except Exception:
            timestamp = pd.NaT
    if not pd.isna(timestamp):
        return int(timestamp.day)
    match = re.search(r"(\d{1,2})", text)
    if match:
        day = int(match.group(1))
        if 1 <= day <= 31:
            return day
    return None


def _resolve_row_emi_cycle(row: pd.Series, default_cycles: list[int]) -> int:
    derived = _coerce_emi_cycle_from_value(row.get("EMI_DATE"))
    if derived is not None:
        return derived
    if len(default_cycles) == 1:
        return int(default_cycles[0])
    raise ValueError("Could not derive emi_cycle from prediction row EMI_DATE for multi-cycle config.")


def _load_prediction_context_lookup(prediction_file: Path, prediction_month_label: str) -> dict[str, dict[str, object]]:
    df = pd.read_csv(prediction_file)
    df = df[df["MONTH"] == prediction_month_label].copy()
    lookup: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        context: dict[str, object] = {}
        raw_reason = row.get("PREDICTION_REASON") if "PREDICTION_REASON" in row.index else None
        if raw_reason is not None and not pd.isna(raw_reason):
            try:
                parsed = json.loads(str(raw_reason))
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                context["reasons"] = {str(key): str(value) for key, value in parsed.items()}
        lookup[str(row["Loan_number"])] = context
    return lookup


def _mapping_prediction_reason(row: pd.Series, context_lookup: dict[str, dict[str, object]]) -> str:
    loan_reasons = context_lookup.get(str(row["loan_number"]), {}).get("reasons", {})
    parts: list[str] = []
    for day in str(row["date"]).split(","):
        day = day.strip()
        if not day:
            continue
        reason = loan_reasons.get(day)
        if reason:
            parts.append(f"{day}: {reason}")
    if parts:
        return " ".join(parts)
    return "Reason not available from prediction output for this campaign mapping."


def _extract_scheduler_emi_dates(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        items = raw_value
    else:
        if pd.isna(raw_value):
            return []
        text = str(raw_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [part.strip() for part in str(parsed).split(",") if part.strip()]

    dates: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        try:
            normalized = datetime.strptime(value, "%d-%m-%Y").strftime("%d-%m-%Y")
        except ValueError:
            continue
        if normalized not in seen:
            seen.add(normalized)
            dates.append(normalized)
    return dates


def resolve_scheduler_emi_dates(
    conn,
    *,
    source_schema: str,
    target_schema: str,
    logger: logging.Logger,
) -> list[str]:
    candidate_refs: list[sql.Composable] = [sql.Identifier("data_config")]
    seen = {"data_config"}
    for schema_name in (source_schema, target_schema):
        key = f"{schema_name}.data_config"
        if schema_name and key not in seen:
            seen.add(key)
            candidate_refs.append(sql.SQL("{}.{}").format(sql.Identifier(schema_name), sql.Identifier("data_config")))

    for table_ref in candidate_refs:
        try:
            row = conn.execute(
                sql.SQL("SELECT value FROM {} WHERE key_name = %s LIMIT 1").format(table_ref),
                (EMI_DATES_CONFIG_KEY,),
            ).fetchone()
        except errors.UndefinedTable:
            conn.rollback()
            continue
        except Exception:
            logger.exception("Scheduler EMI date lookup failed.")
            conn.rollback()
            break

        if not row:
            continue

        emi_dates = _extract_scheduler_emi_dates(row[0])
        if emi_dates:
            logger.info("Resolved scheduler EMI dates from data_config | key=%s emi_dates=%s", EMI_DATES_CONFIG_KEY, emi_dates)
            return emi_dates

    return []


def _build_campaign_assignment_groups(
    prediction_file: Path,
    *,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
    emi_cycles: list[int],
    vertical: str,
    vendors: list[str],
    configured_emi_dates: list[str],
    run_token: str,
) -> pd.DataFrame:
    df = pd.read_csv(prediction_file)
    df = df[df["MONTH"] == prediction_month_label].copy()
    rows: list[dict] = []

    for _, row in df.iterrows():
        loan_number = str(row["Loan_number"])
        risk = str(row.get("SOURCE_RISK", row.get("RISK", "LOW"))).upper()
        risk_code = RISK_CODES.get(risk, "LR")
        for day in [*PREDUE_DAYS, "D", *POSTDUE_DAYS]:
            for strategy in str(row.get(day, "")).split("|"):
                parsed = _parse_strategy(strategy.strip())
                if parsed is None:
                    continue
                mode, send_time, language = parsed
                campaign_type, due_type, campaign_dates = _due_bucket(day, configured_emi_dates=configured_emi_dates)
                vertical_value = str(row.get("SOURCE_VERTICAL", row.get("VERTICAL", vertical))).strip().upper()
                if not vertical_value or vertical_value == "UNKNOWN":
                    vertical_value = vertical.upper()
                template_name = _template_name(
                    due_type=due_type,
                    mode=mode,
                    language=language,
                )
                emi_cycle = _resolve_row_emi_cycle(row, emi_cycles)
                dataset_name = f"{due_type}|{NAME_CHANNEL_BY_MODE[mode]}|{vertical_value}|{language}|{risk_code}|{emi_cycle}"
                for vendor in vendors:
                    base_name = _scheduler_name(
                        due_type=due_type,
                        mode=mode,
                        vertical=vertical_value,
                        language=language,
                        emi_cycle=emi_cycle,
                        vendor=vendor,
                        risk_code=risk_code,
                        run_token=run_token,
                    )
                    rows.append(
                        {
                            "loan_number": loan_number,
                            "base_name": base_name,
                            "mode": mode,
                            "date": campaign_dates,
                            "time": send_time,
                            "template_name": template_name,
                            "dataset_name": dataset_name,
                            "vendor": vendor,
                            "language": language,
                            "active": "T",
                            "source_month": source_month_label,
                            "prediction_month": prediction_month_label,
                            "model_name": model_name,
                            "emi_cycle": emi_cycle,
                            "risk": risk_code,
                            "vertical": vertical_value,
                            "campaign_type": campaign_type,
                            "due_type": due_type,
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "loan_number",
                "base_name",
                "mode",
                "date",
                "time",
                "template_name",
                "dataset_name",
                "vendor",
                "language",
                "active",
                "source_month",
                "prediction_month",
                "model_name",
                "emi_cycle",
                "risk",
                "vertical",
                "campaign_type",
                "due_type",
                "prediction_reason",
            ]
        )

    result = pd.DataFrame(rows).drop_duplicates()
    group_cols = [col for col in result.columns if col != "date"]
    result = result.groupby(group_cols, as_index=False)["date"].agg(_join_campaign_days)

    group_cols = [col for col in result.columns if col not in {"base_name", "name", "time"}]
    result = result.groupby(group_cols, as_index=False).agg(
        {
            "base_name": "first",
            "time": _join_campaign_times,
        }
    )
    return result


def _prepare_campaign_outputs(
    prediction_file: Path,
    *,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
    emi_cycles: list[int],
    vertical: str,
    vendors: list[str],
    configured_emi_dates: list[str],
    run_date: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_token = (run_date or datetime.now(timezone.utc)).strftime("%d%m%y")
    assignment_groups = _build_campaign_assignment_groups(
        prediction_file,
        source_month_label=source_month_label,
        prediction_month_label=prediction_month_label,
        model_name=model_name,
        emi_cycles=emi_cycles,
        vertical=vertical,
        vendors=vendors,
        configured_emi_dates=configured_emi_dates,
        run_token=run_token,
    )
    if assignment_groups.empty:
        empty_campaigns = pd.DataFrame(
            columns=[
                "name",
                "mode",
                "date",
                "time",
                "template_name",
                "dataset_name",
                "vendor",
                "active",
                "source_month",
                "prediction_month",
                "model_name",
                "emi_cycle",
                "risk",
                "vertical",
                "campaign_type",
                "due_type",
            ]
        )
        empty_mappings = pd.DataFrame(
            columns=[
                "campaign_name",
                "loan_number",
                "mode",
                "date",
                "time",
                "vendor",
                "language",
                "source_month",
                "prediction_month",
                "model_name",
                "emi_cycle",
                "risk",
                "vertical",
                "campaign_type",
                "due_type",
                "prediction_reason",
            ]
        )
        return empty_campaigns, empty_mappings

    campaigns = assignment_groups.drop(columns=["loan_number"]).drop_duplicates()
    campaigns = campaigns.sort_values(["base_name", "time", "date"]).reset_index(drop=True)
    duplicate_index = campaigns.groupby("base_name").cumcount() + 1
    duplicate_count = campaigns.groupby("base_name")["base_name"].transform("size")
    campaigns["name"] = campaigns["base_name"]
    campaigns.loc[duplicate_count > 1, "name"] = (
        campaigns.loc[duplicate_count > 1, "base_name"] + "_" + duplicate_index.loc[duplicate_count > 1].astype(str)
    )
    campaigns["dataset_name"] = campaigns.apply(
        lambda row: _dataset_name(
            due_type=str(row["due_type"]),
            mode=str(row["mode"]),
            vertical=str(row["vertical"]),
            language=str(row["template_name"]).rsplit("_", 1)[-1],
            risk_code=str(row["risk"]),
            emi_cycle=int(row["emi_cycle"]),
            date_value=str(row["date"]),
            time_value=str(row["time"]),
        ),
        axis=1,
    )

    join_cols = [
        "base_name",
        "mode",
        "date",
        "time",
        "template_name",
        "vendor",
        "language",
        "active",
        "source_month",
        "prediction_month",
        "model_name",
        "emi_cycle",
        "risk",
        "vertical",
        "campaign_type",
        "due_type",
    ]
    mappings = assignment_groups.merge(campaigns[["name", *join_cols]], on=join_cols, how="left")
    mappings = mappings.rename(columns={"name": "campaign_name"})
    mappings = mappings[
        [
            "campaign_name",
            "loan_number",
            "mode",
            "date",
            "time",
            "vendor",
            "language",
            "source_month",
            "prediction_month",
            "model_name",
            "emi_cycle",
            "risk",
            "vertical",
            "campaign_type",
            "due_type",
        ]
    ].drop_duplicates()
    context_lookup = _load_prediction_context_lookup(prediction_file, prediction_month_label)
    mappings["prediction_reason"] = mappings.apply(
        lambda row: _mapping_prediction_reason(row, context_lookup),
        axis=1,
    )

    campaign_output = campaigns[
        [
            "name",
            "mode",
            "date",
            "time",
            "template_name",
            "dataset_name",
            "vendor",
            "active",
            "source_month",
            "prediction_month",
            "model_name",
            "emi_cycle",
            "risk",
            "vertical",
            "campaign_type",
            "due_type",
        ]
    ]
    return campaign_output, mappings


def build_campaign_recommendations(
    prediction_file: Path,
    *,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
    emi_cycles: list[int],
    vertical: str,
    vendors: list[str],
    run_date: datetime | None = None,
) -> pd.DataFrame:
    campaign_rows, _ = _prepare_campaign_outputs(
        prediction_file,
        source_month_label=source_month_label,
        prediction_month_label=prediction_month_label,
        model_name=model_name,
        emi_cycles=emi_cycles,
        vertical=vertical,
        vendors=vendors,
        run_date=run_date,
    )
    return campaign_rows


def build_campaign_mappings(
    prediction_file: Path,
    *,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
    emi_cycles: list[int],
    vertical: str,
    vendors: list[str],
    run_date: datetime | None = None,
) -> pd.DataFrame:
    _, mapping_rows = _prepare_campaign_outputs(
        prediction_file,
        source_month_label=source_month_label,
        prediction_month_label=prediction_month_label,
        model_name=model_name,
        emi_cycles=emi_cycles,
        vertical=vertical,
        vendors=vendors,
        run_date=run_date,
    )
    return mapping_rows


def store_campaign_recommendations(
    conn,
    schema: str,
    table: str,
    campaign_rows: pd.DataFrame,
    *,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
) -> int:
    ensure_campaign_table(conn, schema, table)
    conn.execute(
        sql.SQL(
            """
            DELETE FROM {table_ref}
            WHERE source_month = %s
              AND prediction_month = %s
              AND model_name = %s
            """
        ).format(table_ref=qualified_identifier(schema, table)),
        (source_month_label, prediction_month_label, model_name),
    )
    if campaign_rows.empty:
        return 0

    now = datetime.now(timezone.utc)
    actor = "campaign-model"
    rows = []
    for _, row in campaign_rows.iterrows():
        rows.append(
            (
                row["name"],
                row["mode"],
                row["date"],
                row["time"],
                row["template_name"],
                row["dataset_name"],
                row["vendor"],
                row["active"],
                row["source_month"],
                row["prediction_month"],
                row["model_name"],
                int(row["emi_cycle"]),
                row["risk"],
                row["vertical"],
                row["campaign_type"],
                row["due_type"],
                now,
                now,
                actor,
                actor,
            )
        )

    query = sql.SQL(
        """
        INSERT INTO {table_ref} (
            name,
            mode,
            date,
            time,
            template_name,
            dataset_name,
            vendor,
            active,
            source_month,
            prediction_month,
            model_name,
            emi_cycle,
            risk,
            vertical,
            campaign_type,
            due_type,
            created_at,
            modified_at,
            created_by,
            modified_by
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name)
        DO UPDATE SET
            mode = EXCLUDED.mode,
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            template_name = EXCLUDED.template_name,
            dataset_name = EXCLUDED.dataset_name,
            vendor = EXCLUDED.vendor,
            active = EXCLUDED.active,
            source_month = EXCLUDED.source_month,
            prediction_month = EXCLUDED.prediction_month,
            model_name = EXCLUDED.model_name,
            emi_cycle = EXCLUDED.emi_cycle,
            risk = EXCLUDED.risk,
            vertical = EXCLUDED.vertical,
            campaign_type = EXCLUDED.campaign_type,
            due_type = EXCLUDED.due_type,
            modified_at = EXCLUDED.modified_at,
            modified_by = EXCLUDED.modified_by
        """
    ).format(table_ref=qualified_identifier(schema, table))
    with conn.cursor() as cur:
        cur.executemany(query, rows)
    return len(rows)


def store_campaign_mappings(
    conn,
    schema: str,
    table: str,
    mapping_rows: pd.DataFrame,
    *,
    source_month_label: str,
    prediction_month_label: str,
    model_name: str,
) -> int:
    ensure_campaign_mapping_table(conn, schema, table)
    conn.execute(
        sql.SQL(
            """
            DELETE FROM {table_ref}
            WHERE source_month = %s
              AND prediction_month = %s
              AND model_name = %s
            """
        ).format(table_ref=qualified_identifier(schema, table)),
        (source_month_label, prediction_month_label, model_name),
    )
    if mapping_rows.empty:
        return 0

    now = datetime.now(timezone.utc)
    actor = "campaign-model"
    rows = []
    for _, row in mapping_rows.iterrows():
        rows.append(
            (
                row["campaign_name"],
                row["loan_number"],
                row["mode"],
                row["date"],
                row["time"],
                row["vendor"],
                row["language"],
                row["source_month"],
                row["prediction_month"],
                row["model_name"],
                int(row["emi_cycle"]),
                row["risk"],
                row["vertical"],
                row["campaign_type"],
                row["due_type"],
                row.get("prediction_reason"),
                now,
                now,
                actor,
                actor,
            )
        )

    query = sql.SQL(
        """
        INSERT INTO {table_ref} (
            campaign_name,
            loan_number,
            mode,
            date,
            time,
            vendor,
            language,
            source_month,
            prediction_month,
            model_name,
            emi_cycle,
            risk,
            vertical,
            campaign_type,
            due_type,
            prediction_reason,
            created_at,
            modified_at,
            created_by,
            modified_by
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (campaign_name, loan_number)
        DO UPDATE SET
            mode = EXCLUDED.mode,
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            vendor = EXCLUDED.vendor,
            language = EXCLUDED.language,
            source_month = EXCLUDED.source_month,
            prediction_month = EXCLUDED.prediction_month,
            model_name = EXCLUDED.model_name,
            emi_cycle = EXCLUDED.emi_cycle,
            risk = EXCLUDED.risk,
            vertical = EXCLUDED.vertical,
            campaign_type = EXCLUDED.campaign_type,
            due_type = EXCLUDED.due_type,
            prediction_reason = EXCLUDED.prediction_reason,
            modified_at = EXCLUDED.modified_at,
            modified_by = EXCLUDED.modified_by
        """
    ).format(table_ref=qualified_identifier(schema, table))
    with conn.cursor() as cur:
        cur.executemany(query, rows)
    return len(rows)

def _load_metrics_metadata(metrics_file: Path) -> dict[str, object]:
    if not metrics_file.exists():
        return {}
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_model_drift_metrics(
    *,
    feature_file: Path,
    schedule_file: Path,
    base_population_file: Path,
    model_file: Path,
    metrics_file: Path,
    source_month_label: str,
) -> dict[str, object]:
    bundle = joblib.load(model_file)
    target_offset_months = int(bundle.get("target_offset_months", 1))
    history_window_months = int(bundle.get("history_window_months", 1))

    dataset = prepare_next_month_dataset(
        feature_file=feature_file,
        schedule_file=schedule_file,
        target_offset_months=target_offset_months,
        history_window_months=history_window_months,
    )
    base_population = load_base_population(base_population_file, source_month_label)
    prediction_rows, blank_rows = build_prediction_population(
        dataset=dataset,
        base_population=base_population,
        prediction_source_month=source_month_label,
    )

    metrics_metadata = _load_metrics_metadata(metrics_file)
    baseline_source_months = [str(value) for value in metrics_metadata.get("train_source_months", []) if str(value).strip()]
    if baseline_source_months:
        baseline_rows = split_by_source_month(dataset, baseline_source_months, require_target=True)
    else:
        baseline_rows = dataset[dataset["TARGET_MONTH"].notna()].copy()
        baseline_source_months = sorted(baseline_rows["SOURCE_MONTH"].dropna().astype(str).unique().tolist())

    baseline_matrix = build_feature_matrix(baseline_rows).reindex(columns=bundle["feature_columns"], fill_value=0)
    inference_matrix = build_feature_matrix(prediction_rows).reindex(columns=bundle["feature_columns"], fill_value=0)
    report = compute_drift_report(
        baseline_df=baseline_matrix,
        inference_df=inference_matrix,
        feature_columns=bundle["feature_columns"],
        blank_inference_rows=len(blank_rows),
    )
    report["baseline_source_months"] = baseline_source_months
    report["history_window_months"] = history_window_months
    return report





def main() -> None:
    args = parse_args()
    run_started_at = datetime.now(timezone.utc)
    prediction_file: Path | None = None
    prediction_rows = 0
    drift_report: dict[str, object] | None = None
    logger = setup_logging(args.log_file, "monthly_inference_pipeline")
    logger.info(
        "Monthly inference args: %s",
        {
            key: ("***" if key == "password" else value)
            for key, value in vars(args).items()
        },
    )
    source_month_label = month_label(args.source_month)
    prediction_month_label = month_label(args.predict_month)
    source_extract_file = latest_extract_file(args.source_month)
    source_cases_file = current_cases_file(args.source_month)

    try:
        source_period = parse_month(args.source_month)
        history_fetches = [
            (str(source_period - 2), monthly_extract_file(str(source_period - 2))),
            (str(source_period - 1), monthly_extract_file(str(source_period - 1))),
            (args.source_month, source_extract_file),
        ]
        with log_step(logger, "fetch_communication_history", source_month=args.source_month):
            for fetch_month, output_file in history_fetches:
                fetch_communication_extract(args, output_file, fetch_month, logger)
        with log_step(logger, "fetch_current_cases", source_month=args.source_month):
            fetch_current_cases_extract(args, source_cases_file, args.source_month, logger)

        if not csv_has_rows(source_extract_file):
            logger.warning("No latest communication rows found. Skipping prediction run.")
            print(f"No latest communication rows found in {source_extract_file}; skipped prediction run.")
            return

        history_files = selected_history_files(args.source_month, source_extract_file)
        if len(history_files) < 3:
            raise FileNotFoundError(
                "Need latest communication file plus previous two month files. "
                f"Found {len(history_files)} files: {[str(path) for path in history_files]}"
            )
        logger.info("Selected communication history files: %s", [str(path) for path in history_files])

        with log_step(logger, "prepare_inference_features"):
            run_python_script(
                "generate_strategy_dataset.py",
                "--output-file",
                str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
                "--month-source",
                args.feature_month_source,
                "--input-files",
                *[str(path) for path in history_files],
                logger=logger,
            )
        with log_step(logger, "build_schedule_dataset"):
            run_python_script(
                "build_strategy_schedule_dataset.py",
                "--input-file",
                str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
                "--output-file",
                str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
                logger=logger,
            )
        with log_step(logger, "build_monthly_features"):
            run_python_script(
                "build_monthly_feature_dataset.py",
                "--input-file",
                str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
                "--output-file",
                str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
                logger=logger,
            )

        if args.model in {"catboost", "catboost_3m"}:
            model_suffix = "catboost_3m" if args.model == "catboost_3m" else "catboost"
            prediction_file = (
                PREDICTIONS_DIR
                / f"{args.predict_month.replace('-', '_').lower()}_strategy_predictions_{model_suffix}.csv"
            )
            model_file = MODEL_DIR / f"next_month_strategy_{model_suffix}.joblib"
            metrics_file = METRICS_DIR / f"next_month_strategy_{model_suffix}_metrics.json"
            if args.model_serving == "mlflow":
                with log_step(
                    logger,
                    "download_mlflow_model_bundle",
                    model_uri=args.mlflow_model_uri,
                    model_file=model_file,
                ):
                    download_mlflow_model_bundle(args.mlflow_model_uri, model_file, logger)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"CatBoost model bundle not found: {model_file}. "
                    "Train the CatBoost model once or set MODEL_SERVING=mlflow and MLFLOW_MODEL_URI."
                )
            with log_step(
                logger,
                "run_catboost_inference",
                model_file=model_file,
                prediction_file=prediction_file,
            ):
                run_python_script(
                    "predict_next_month_strategy_catboost.py",
                    "--model-file",
                    str(model_file),
                    "--prediction-file",
                    str(prediction_file),
                    "--base-population-file",
                    str(source_cases_file),
                    "--prediction-source-months",
                    source_month_label,
                    logger=logger,
                )
        else:
            prediction_file = (
                PREDICTIONS_DIR
                / f"{args.predict_month.replace('-', '_').lower()}_strategy_predictions_logistic.csv"
            )
            model_file = MODEL_DIR / "next_month_strategy_logistic.joblib"
            metrics_file = METRICS_DIR / "next_month_strategy_logistic_metrics.json"
            with log_step(logger, "run_logistic_training_and_inference", prediction_file=prediction_file):
                run_python_script(
                    "train_next_month_strategy_model_logistic.py",
                    "--model-file",
                    str(model_file),
                    "--metrics-file",
                    str(metrics_file),
                    "--prediction-file",
                    str(prediction_file),
                    "--train-source-months",
                    "NOV-2025",
                    "DEC-2025",
                    "JAN-2026",
                    "--validation-source-months",
                    "FEB-2026",
                    "--test-source-months",
                    "--prediction-source-months",
                    source_month_label,
                    logger=logger,
                )

        with log_step(logger, "compute_model_drift", model_file=model_file, metrics_file=metrics_file):
            drift_report = compute_model_drift_metrics(
                feature_file=FEATURE_DATA_DIR / "strategy_monthly_features.csv",
                schedule_file=SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv",
                base_population_file=source_cases_file,
                model_file=model_file,
                metrics_file=metrics_file,
                source_month_label=source_month_label,
            )
            metrics_payload = _load_metrics_metadata(metrics_file)
            accuracy_value = None
            validation_metrics = metrics_payload.get("validation_metrics") if isinstance(metrics_payload, dict) else None
            train_metrics = metrics_payload.get("train_metrics") if isinstance(metrics_payload, dict) else None
            accuracy_source = validation_metrics if isinstance(validation_metrics, dict) and validation_metrics.get("average_day_accuracy") is not None else train_metrics
            if isinstance(accuracy_source, dict) and accuracy_source.get("average_day_accuracy") is not None:
                accuracy_value = round(float(accuracy_source["average_day_accuracy"]) * 100, 2)
            drift_report["current_accuracy"] = accuracy_value
            logger.info(
                "Drift summary | baseline_rows=%s inference_rows=%s blank_inference_rows=%s feature_count=%s drift_percentage=%s overall_psi=%s max_feature_psi=%s status=%s",
                drift_report["baseline_rows"],
                drift_report["inference_rows"],
                drift_report["blank_inference_rows"],
                drift_report["feature_count"],
                drift_report["drift_percentage"],
                drift_report["overall_psi"],
                drift_report["max_feature_psi"],
                drift_report["status"],
            )

        if not args.skip_db_store:
            with log_step(logger, "store_snapshots", target_schema=args.target_schema):
                config = PostgresConfig(
                    host=args.host,
                    port=args.port,
                    dbname=args.dbname,
                    user=args.user,
                    password=args.password,
                )
                with connect_db(config) as conn:
                    campaign_vendors = resolve_campaign_vendors(
                        conn,
                        source_schema=args.source_schema,
                        target_schema=args.target_schema,
                        fallback_vendor=args.campaign_vendor,
                        logger=logger,
                    )
                    configured_emi_dates = resolve_scheduler_emi_dates(
                        conn,
                        source_schema=args.source_schema,
                        target_schema=args.target_schema,
                        logger=logger,
                    )
                    prediction_rows = store_prediction_snapshots(
                        conn,
                        args.target_schema,
                        args.prediction_table,
                        prediction_file,
                        prediction_month_label,
                        args.model,
                    )
                    campaign_df, mapping_df = _prepare_campaign_outputs(
                        prediction_file,
                        source_month_label=source_month_label,
                        prediction_month_label=prediction_month_label,
                        model_name=args.model,
                        emi_cycles=resolve_emi_cycle(args.config_file, os.getenv("EMI_CYCLE", "")),
                        vertical=args.campaign_vertical,
                        vendors=campaign_vendors,
                        configured_emi_dates=configured_emi_dates,
                    )
                    campaign_rows = store_campaign_recommendations(
                        conn,
                        args.target_schema,
                        args.campaign_table,
                        campaign_df,
                        source_month_label=source_month_label,
                        prediction_month_label=prediction_month_label,
                        model_name=args.model,
                    )
                    mapping_rows = store_campaign_mappings(
                        conn,
                        args.target_schema,
                        args.campaign_mapping_table,
                        mapping_df,
                        source_month_label=source_month_label,
                        prediction_month_label=prediction_month_label,
                        model_name=args.model,
                    )
                    conn.commit()
                print(f"Stored {prediction_rows:,} prediction snapshots in Postgres")
                print(f"Stored {campaign_rows:,} campaign recommendation snapshots in Postgres")
                print(f"Stored {mapping_rows:,} campaign mapping snapshots in Postgres")
        else:
            campaign_vendors = []
            configured_emi_dates = []
            campaign_df, mapping_df = _prepare_campaign_outputs(
                prediction_file,
                source_month_label=source_month_label,
                prediction_month_label=prediction_month_label,
                model_name=args.model,
                emi_cycles=resolve_emi_cycle(args.config_file, os.getenv("EMI_CYCLE", "")),
                vertical=args.campaign_vertical,
                vendors=_extract_campaign_vendors(args.campaign_vendor) or [args.campaign_vendor],
                configured_emi_dates=configured_emi_dates,
            )

        summary = build_prediction_summary(
            prediction_file=prediction_file,
            source_month_label=source_month_label,
            prediction_month_label=prediction_month_label,
            model_name=args.model,
            campaign_df=campaign_df,
            mapping_df=mapping_df,
        )
        if drift_report is not None:
            summary["drift"] = {
                key: value
                for key, value in drift_report.items()
                if key != "feature_metrics"
            }
        summary_path = write_prediction_summary(
            summary=summary,
            output_path=prediction_summary_path(args.predict_month),
            logger=logger,
        )

        print(f"Prediction file: {prediction_file}")
        print(f"Metrics file: {metrics_file}")
        print(f"Prediction summary: {summary_path}")
        logger.info("Monthly inference completed successfully.")
    except Exception as exc:
        logger.exception("Monthly inference failed.")
        raise


if __name__ == "__main__":
    main()
