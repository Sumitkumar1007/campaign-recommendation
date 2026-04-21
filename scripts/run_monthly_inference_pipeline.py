from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from psycopg import sql
from psycopg.types.json import Jsonb

from app_logging import log_step, setup_logging
from env_utils import load_dotenv
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import (
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


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=(
            "Incremental monthly inference pipeline: fetch one source month from "
            "Postgres, rebuild local processed data, run inference, and store "
            "feature/prediction snapshots back to Postgres."
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
        help="Schema used for storing processed features and predictions.",
    )
    parser.add_argument(
        "--feature-table",
        default=os.getenv("FEATURE_TABLE", "ai_ml_recommendations_feature"),
        help="Target table for processed source-month feature snapshots.",
    )
    parser.add_argument(
        "--prediction-table",
        default=os.getenv("PREDICTION_TABLE", "ai_ml_recommendations_data"),
        help="Target table for model prediction snapshots.",
    )
    parser.add_argument(
        "--audit-table",
        default=os.getenv("AUDIT_TABLE", "ai_ml_audit_table"),
        help="Target table for pipeline audit records.",
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
        "--feature-month-source",
        choices=["emi_date", "created_date"],
        default=os.getenv("FEATURE_MONTH_SOURCE", "created_date"),
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


def selected_history_files(source_month: str, latest_file: Path) -> list[Path]:
    source_period = parse_month(source_month)
    previous_periods = [source_period - 2, source_period - 1]
    files: list[Path] = []
    for period in previous_periods:
        token = month_file_token(period)
        files.extend(sorted(COMMUNICATION_DATA_DIR.glob(f"*{token}*.csv")))
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
        logger.info("Running child script | script=%s args=%s", script_name, list(script_args))
    subprocess.run(cmd, check=True, cwd=SCRIPTS_DIR.parent, env=env)


def ensure_feature_table(conn, schema: str, table: str) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_ref} (
                apac_card_number TEXT NOT NULL,
                source_month TEXT NOT NULL,
                risk TEXT,
                feature_payload JSONB NOT NULL,
                pipeline_version TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (apac_card_number, source_month)
            )
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )


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
                created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (loan_number, prediction_month, model_name)
            )
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )


def ensure_audit_table(conn, schema: str, table: str) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_ref} (
                audit_id BIGSERIAL PRIMARY KEY,
                audit_key TEXT NOT NULL,
                audit_value TEXT NOT NULL,
                model_name TEXT NOT NULL,
                source_month TEXT,
                prediction_month TEXT,
                status TEXT NOT NULL,
                prediction_completed_count INTEGER NOT NULL DEFAULT 0,
                prediction_failed_count INTEGER NOT NULL DEFAULT 0,
                failed_reason TEXT,
                duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                feature_table TEXT,
                prediction_table TEXT,
                prediction_file TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                modified_at TIMESTAMPTZ NOT NULL,
                created_by TEXT NOT NULL,
                modified_by TEXT NOT NULL
            )
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )
    conn.execute(
        sql.SQL(
            """
            DELETE FROM {table_ref} older
            USING {table_ref} newer
            WHERE older.audit_id < newer.audit_id
              AND older.audit_key = newer.audit_key
              AND older.audit_value = newer.audit_value
              AND older.model_name = newer.model_name
              AND older.source_month = newer.source_month
              AND older.prediction_month = newer.prediction_month
            """
        ).format(table_ref=qualified_identifier(schema, table))
    )
    conn.execute(
        sql.SQL(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
            ON {table_ref} (
                audit_key,
                audit_value,
                model_name,
                source_month,
                prediction_month
            )
            """
        ).format(
            index_name=sql.Identifier(f"{table}_audit_unique_idx"),
            table_ref=qualified_identifier(schema, table),
        )
    )


def store_audit_record(
    conn,
    schema: str,
    table: str,
    *,
    model_name: str,
    source_month: str,
    prediction_month: str,
    status: str,
    prediction_completed_count: int,
    prediction_failed_count: int,
    failed_reason: str | None,
    duration_seconds: float,
    feature_table: str,
    prediction_table: str,
    prediction_file: Path | None,
) -> None:
    ensure_audit_table(conn, schema, table)
    now = datetime.now(timezone.utc)
    actor = "campaign-model"
    conn.execute(
        sql.SQL(
            """
            INSERT INTO {table_ref} (
                audit_key,
                audit_value,
                model_name,
                source_month,
                prediction_month,
                status,
                prediction_completed_count,
                prediction_failed_count,
                failed_reason,
                duration_seconds,
                feature_table,
                prediction_table,
                prediction_file,
                created_at,
                modified_at,
                created_by,
                modified_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (
                audit_key,
                audit_value,
                model_name,
                source_month,
                prediction_month
            )
            DO UPDATE SET
                status = EXCLUDED.status,
                prediction_completed_count = EXCLUDED.prediction_completed_count,
                prediction_failed_count = EXCLUDED.prediction_failed_count,
                failed_reason = EXCLUDED.failed_reason,
                duration_seconds = EXCLUDED.duration_seconds,
                feature_table = EXCLUDED.feature_table,
                prediction_table = EXCLUDED.prediction_table,
                prediction_file = EXCLUDED.prediction_file,
                modified_at = EXCLUDED.modified_at,
                modified_by = EXCLUDED.modified_by
            """
        ).format(table_ref=qualified_identifier(schema, table)),
        (
            "model",
            "recommendation",
            model_name,
            source_month,
            prediction_month,
            status,
            int(prediction_completed_count),
            int(prediction_failed_count),
            failed_reason,
            float(duration_seconds),
            feature_table,
            prediction_table,
            str(prediction_file) if prediction_file else None,
            now,
            now,
            actor,
            actor,
        ),
    )


def write_pipeline_audit(
    args: argparse.Namespace,
    *,
    status: str,
    prediction_completed_count: int,
    prediction_failed_count: int,
    failed_reason: str | None,
    duration_seconds: float,
    prediction_file: Path | None,
    logger: logging.Logger,
) -> None:
    try:
        config = PostgresConfig(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
        )
        with connect_db(config) as conn:
            store_audit_record(
                conn,
                args.target_schema,
                args.audit_table,
                model_name=args.model,
                source_month=month_label(args.source_month),
                prediction_month=month_label(args.predict_month),
                status=status,
                prediction_completed_count=prediction_completed_count,
                prediction_failed_count=prediction_failed_count,
                failed_reason=failed_reason,
                duration_seconds=duration_seconds,
                feature_table=args.feature_table,
                prediction_table=args.prediction_table,
                prediction_file=prediction_file,
            )
            conn.commit()
    except Exception:
        logger.exception("Failed to write pipeline audit record.")


def store_feature_snapshots(
    conn,
    schema: str,
    table: str,
    source_month_label: str,
    pipeline_version: str,
) -> int:
    feature_file = FEATURE_DATA_DIR / "strategy_monthly_features.csv"
    df = pd.read_csv(feature_file)
    df = df[df["MONTH"] == source_month_label].copy()
    if df.empty:
        return 0

    ensure_feature_table(conn, schema, table)
    now = datetime.now(timezone.utc)
    rows = []
    for _, row in df.iterrows():
        payload = row.drop(labels=["APAC_CARD_NUMBER", "MONTH", "RISK"]).to_dict()
        rows.append(
            (
                str(row["APAC_CARD_NUMBER"]),
                source_month_label,
                row.get("RISK"),
                Jsonb(payload),
                pipeline_version,
                now,
            )
        )

    query = sql.SQL(
        """
        INSERT INTO {table_ref} (
            apac_card_number,
            source_month,
            risk,
            feature_payload,
            pipeline_version,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (apac_card_number, source_month)
        DO UPDATE SET
            risk = EXCLUDED.risk,
            feature_payload = EXCLUDED.feature_payload,
            pipeline_version = EXCLUDED.pipeline_version,
            created_at = EXCLUDED.created_at
        """
    ).format(table_ref=qualified_identifier(schema, table))
    with conn.cursor() as cur:
        cur.executemany(query, rows)
    return len(rows)


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
        rows.append(
            (
                str(row["Loan_number"]),
                str(row["SOURCE_MONTH_USED"]),
                prediction_month_label,
                model_name,
                row.get(source_risk_col),
                Jsonb(payload),
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
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (loan_number, prediction_month, model_name)
        DO UPDATE SET
            source_month_used = EXCLUDED.source_month_used,
            source_risk = EXCLUDED.source_risk,
            prediction_payload = EXCLUDED.prediction_payload,
            created_at = EXCLUDED.created_at
        """
    ).format(table_ref=qualified_identifier(schema, table))
    with conn.cursor() as cur:
        cur.executemany(query, rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    run_started_at = datetime.now(timezone.utc)
    prediction_file: Path | None = None
    prediction_rows = 0
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

        if not csv_has_rows(source_extract_file):
            logger.warning("No latest communication rows found. Skipping prediction run.")
            print(f"No latest communication rows found in {source_extract_file}; skipped prediction run.")
            write_pipeline_audit(
                args,
                status="SKIPPED",
                prediction_completed_count=0,
                prediction_failed_count=0,
                failed_reason=f"No latest communication rows found in {source_extract_file}",
                duration_seconds=(datetime.now(timezone.utc) - run_started_at).total_seconds(),
                prediction_file=None,
                logger=logger,
            )
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
            if not model_file.exists():
                raise FileNotFoundError(
                    f"CatBoost model bundle not found: {model_file}. "
                    "Train the CatBoost model once before running monthly inference."
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
                    feature_rows = store_feature_snapshots(
                        conn,
                        args.target_schema,
                        args.feature_table,
                        source_month_label,
                        pipeline_version="v1",
                    )
                    prediction_rows = store_prediction_snapshots(
                        conn,
                        args.target_schema,
                        args.prediction_table,
                        prediction_file,
                        prediction_month_label,
                        args.model,
                    )
                    conn.commit()
                print(f"Stored {feature_rows:,} feature snapshots in Postgres")
                print(f"Stored {prediction_rows:,} prediction snapshots in Postgres")

        print(f"Prediction file: {prediction_file}")
        print(f"Metrics file: {metrics_file}")
        write_pipeline_audit(
            args,
            status="SUCCESS",
            prediction_completed_count=prediction_rows,
            prediction_failed_count=0,
            failed_reason=None,
            duration_seconds=(datetime.now(timezone.utc) - run_started_at).total_seconds(),
            prediction_file=prediction_file,
            logger=logger,
        )
        logger.info("Monthly inference completed successfully.")
    except Exception as exc:
        logger.exception("Monthly inference failed.")
        write_pipeline_audit(
            args,
            status="FAILED",
            prediction_completed_count=prediction_rows,
            prediction_failed_count=1,
            failed_reason=str(exc),
            duration_seconds=(datetime.now(timezone.utc) - run_started_at).total_seconds(),
            prediction_file=prediction_file,
            logger=logger,
        )
        raise


if __name__ == "__main__":
    main()
