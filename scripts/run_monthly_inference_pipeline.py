from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from psycopg import sql
from psycopg.types.json import Jsonb

from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import (
    FEATURE_DATA_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SCHEDULE_DATA_DIR,
    SCRIPTS_DIR,
    TRAINING_DATA_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incremental monthly inference pipeline: fetch one source month from "
            "Postgres, rebuild local processed data, run inference, and store "
            "feature/prediction snapshots back to Postgres."
        )
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--source-schema",
        default="digital_collections",
        help="Schema containing the source communications table.",
    )
    parser.add_argument(
        "--source-table",
        default="communications",
        help="Source communications table.",
    )
    parser.add_argument(
        "--target-schema",
        default="digital_collections",
        help="Schema used for storing processed features and predictions.",
    )
    parser.add_argument(
        "--feature-table",
        default="recommendation_feature_snapshots",
        help="Target table for processed source-month feature snapshots.",
    )
    parser.add_argument(
        "--prediction-table",
        default="recommendation_prediction_snapshots",
        help="Target table for model prediction snapshots.",
    )
    parser.add_argument(
        "--source-month",
        required=True,
        help="Source month to fetch and process in YYYY-MM format.",
    )
    parser.add_argument(
        "--predict-month",
        required=True,
        help="Target month to predict in YYYY-MM format.",
    )
    parser.add_argument(
        "--model",
        choices=["catboost", "logistic"],
        default="catboost",
        help="Which next-month model pipeline to run.",
    )
    parser.add_argument(
        "--filter-on",
        choices=["emi_date", "created_date"],
        default="emi_date",
        help="Which date field defines the source month extract window.",
    )
    parser.add_argument(
        "--skip-db-store",
        action="store_true",
        help="Run the local fetch/process/predict flow without storing snapshots back to Postgres.",
    )
    return parser.parse_args()


def month_label(yyyy_mm: str) -> str:
    return pd.Timestamp(f"{yyyy_mm}-01").strftime("%b-%Y").upper()


def run_python_script(script_name: str, *script_args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *script_args]
    subprocess.run(cmd, check=True, cwd=SCRIPTS_DIR.parent)


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

    conn.executemany(
        sql.SQL(
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
        ).format(table_ref=qualified_identifier(schema, table)),
        rows,
    )
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

    conn.executemany(
        sql.SQL(
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
        ).format(table_ref=qualified_identifier(schema, table)),
        rows,
    )
    return len(rows)


def main() -> None:
    args = parse_args()
    source_month_label = month_label(args.source_month)
    prediction_month_label = month_label(args.predict_month)

    run_python_script(
        "fetch_month_from_postgres.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dbname",
        args.dbname,
        "--user",
        args.user,
        "--password",
        args.password,
        "--schema",
        args.source_schema,
        "--table",
        args.source_table,
        "--source-month",
        args.source_month,
        "--filter-on",
        args.filter_on,
    )

    run_python_script(
        "generate_strategy_dataset.py",
        "--output-file",
        str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
    )
    run_python_script(
        "build_strategy_schedule_dataset.py",
        "--input-file",
        str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
        "--output-file",
        str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"),
    )
    run_python_script(
        "build_monthly_feature_dataset.py",
        "--input-file",
        str(TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"),
        "--output-file",
        str(FEATURE_DATA_DIR / "strategy_monthly_features.csv"),
    )

    if args.model == "catboost":
        prediction_file = PREDICTIONS_DIR / f"{args.predict_month.replace('-', '_').lower()}_strategy_predictions_catboost.csv"
        model_file = MODEL_DIR / "next_month_strategy_catboost.joblib"
        metrics_file = METRICS_DIR / "next_month_strategy_catboost_metrics.json"
        if not model_file.exists():
            raise FileNotFoundError(
                f"CatBoost model bundle not found: {model_file}. "
                "Train the CatBoost model once before running monthly inference."
            )
        run_python_script(
            "predict_next_month_strategy_catboost.py",
            "--model-file",
            str(model_file),
            "--prediction-file",
            str(prediction_file),
            "--prediction-source-months",
            source_month_label,
        )
    else:
        prediction_file = PREDICTIONS_DIR / f"{args.predict_month.replace('-', '_').lower()}_strategy_predictions_logistic.csv"
        model_file = MODEL_DIR / "next_month_strategy_logistic.joblib"
        metrics_file = METRICS_DIR / "next_month_strategy_logistic_metrics.json"
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
        )

    if not args.skip_db_store:
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


if __name__ == "__main__":
    main()
