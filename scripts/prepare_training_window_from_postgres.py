from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from psycopg import sql

from app_logging import log_step, setup_logging
from env_utils import load_dotenv
from fetch_month_from_postgres import (
    build_query,
    default_output_file,
    parse_args as _unused_fetch_parse_args,
    write_query_to_csv,
)
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import FEATURE_DATA_DIR, REPO_ROOT, SCHEDULE_DATA_DIR, TRAINING_DATA_DIR
from run_monthly_inference_pipeline import run_python_script


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=(
            "Prepare communication extracts and derived training datasets from Postgres "
            "for the requested historical month window."
        )
    )
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--schema", default=os.getenv("SOURCE_SCHEMA", "digital_collections"))
    parser.add_argument("--table", default=os.getenv("SOURCE_TABLE", "communications"))
    parser.add_argument("--months", type=int, required=True, help="Requested historical source-month window for training.")
    parser.add_argument("--month-source", choices=["emi_date", "created_date"], default=os.getenv("FEATURE_MONTH_SOURCE", "emi_date"))
    parser.add_argument("--config-file", default=os.getenv("CONFIG_FILE", str(REPO_ROOT / "config" / "default_config.json")))
    parser.add_argument("--fetch-size", type=int, default=100_000)
    parser.add_argument("--log-file", default=None)
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
    if args.months < 2:
        parser.error("--months must be at least 2")
    if args.fetch_size <= 0:
        parser.error("--fetch-size must be greater than 0")
    return args


def latest_available_source_months(conn, schema: str, table: str, raw_month_count: int) -> list[str]:
    table_ref = qualified_identifier(schema, table)
    query = sql.SQL(
        """
        SELECT DISTINCT TO_CHAR(DATE_TRUNC('month', TO_DATE(c.emi_date, 'DD/MM/YYYY')), 'YYYY-MM') AS source_month
        FROM {table_ref} c
        WHERE c.emi_date IS NOT NULL
        ORDER BY source_month DESC
        LIMIT %s
        """
    ).format(table_ref=table_ref)
    rows = conn.execute(query, (raw_month_count,)).fetchall()
    months_desc = [row[0] for row in rows if row and row[0]]
    months_desc.sort()
    return months_desc


def fetch_month_extracts(args: argparse.Namespace, logger: logging.Logger, months: list[str]) -> list[Path]:
    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    files: list[Path] = []
    query = build_query(args.schema, args.table)
    logger.info(
        "Opening source DB connection for training extract | host=%s port=%s dbname=%s schema=%s table=%s months=%s",
        args.host,
        args.port,
        args.dbname,
        args.schema,
        args.table,
        months,
    )
    with connect_db(config) as conn:
        logger.info("Source DB connection established for training extract")
        for month in months:
            output_file = default_output_file(month)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # Reuse the existing month fetch SQL by substituting the configured EMI-cycle day dates.
            # The fetch utility itself is still the source of truth for monthly communication snapshots.
            from fetch_month_from_postgres import emi_cycle_dates, resolve_emi_cycle  # local import to avoid CLI coupling
            emi_cycle = resolve_emi_cycle(args.config_file, os.getenv("EMI_CYCLE", ""))
            params = {"emi_dates": [date.date() for date in emi_cycle_dates(emi_cycle, fetch_month=month)]}
            logger.info(
                "Resolved EMI dates for training month | month=%s emi_cycle=%s emi_dates=%s output_file=%s",
                month,
                emi_cycle,
                params["emi_dates"],
                output_file,
            )
            with log_step(logger, "fetch_training_month", month=month, output_file=output_file):
                row_count = write_query_to_csv(conn, query, params, output_file, args.fetch_size)
                file_size = output_file.stat().st_size if output_file.exists() else 0
                logger.info(
                    "Fetched training month | month=%s rows=%s output_file=%s file_size_bytes=%s",
                    month,
                    row_count,
                    output_file,
                    file_size,
                )
            files.append(output_file)
    logger.info("Completed monthly extract fetch | file_count=%s files=%s", len(files), files)
    return files


def build_training_artifacts(input_files: list[Path], month_source: str, logger: logging.Logger) -> None:
    training_output = TRAINING_DATA_DIR / "strategy_training_dataset_all_months.csv"
    schedule_output = SCHEDULE_DATA_DIR / "strategy_schedule_dataset_all_months.csv"
    feature_output = FEATURE_DATA_DIR / "strategy_monthly_features.csv"
    with log_step(logger, "generate_strategy_dataset", input_files=','.join(str(path) for path in input_files)):
        run_python_script(
            "generate_strategy_dataset.py",
            "--output-file",
            str(training_output),
            "--month-source",
            month_source,
            "--input-files",
            *[str(path) for path in input_files],
            logger=logger,
        )
        logger.info("Generated training dataset | output_file=%s exists=%s", training_output, training_output.exists())
    with log_step(logger, "build_schedule_dataset"):
        run_python_script(
            "build_strategy_schedule_dataset.py",
            "--input-file",
            str(training_output),
            "--output-file",
            str(schedule_output),
            logger=logger,
        )
        logger.info("Generated schedule dataset | output_file=%s exists=%s", schedule_output, schedule_output.exists())
    with log_step(logger, "build_monthly_features"):
        run_python_script(
            "build_monthly_feature_dataset.py",
            "--input-file",
            str(training_output),
            "--output-file",
            str(feature_output),
            logger=logger,
        )
        logger.info("Generated feature dataset | output_file=%s exists=%s", feature_output, feature_output.exists())


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file, "prepare_training_window")
    try:
        raw_month_count = args.months + 1
        config = PostgresConfig(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
        )
        logger.info(
            "Preparing training window | months=%s host=%s dbname=%s schema=%s table=%s month_source=%s",
            args.months,
            args.host,
            args.dbname,
            args.schema,
            args.table,
            args.month_source,
        )
        logger.info("Connecting to source DB to resolve latest available source months")
        with connect_db(config) as conn:
            months = latest_available_source_months(conn, args.schema, args.table, raw_month_count)
        logger.info("Resolved latest available source months from DB | months=%s", months)
        if len(months) < 3:
            raise ValueError(
                f"Need at least 3 source months to prepare training data, but found only {len(months)} month(s): {months}"
            )
        logger.info(
            "Resolved training month window | requested_months=%s raw_month_count=%s fetched_source_months=%s",
            args.months,
            raw_month_count,
            months,
        )
        input_files = fetch_month_extracts(args, logger, months)
        logger.info("Fetched all month extracts successfully | input_files=%s", input_files)
        build_training_artifacts(input_files, args.month_source, logger)
        logger.info("Training window preparation completed successfully | requested_months=%s raw_source_months=%s", args.months, months)
        print(f"Prepared training data for requested months={args.months} using raw source months={months}")
    except Exception:
        logger.exception("Training window preparation failed.")
        raise


if __name__ == "__main__":
    main()
