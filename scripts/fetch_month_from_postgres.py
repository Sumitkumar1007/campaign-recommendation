from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from psycopg import sql

from env_utils import load_dotenv
from pipeline_common import resolve_emi_cycle
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import COMMUNICATION_DATA_DIR, REPO_ROOT, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Fetch one source month of communication data from Postgres."
    )
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--schema",
        default=os.getenv("SOURCE_SCHEMA", "digital_collections"),
        help="Schema containing the communications source table.",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("SOURCE_TABLE", "communications"),
        help="Source communications table.",
    )
    parser.add_argument(
        "--config-file",
        default=os.getenv("CONFIG_FILE", str(REPO_ROOT / "config" / "default_config.json")),
        help="JSON config file containing emi_cycle.",
    )
    parser.add_argument(
        "--emi-cycle",
        default=os.getenv("EMI_CYCLE", ""),
        help="Comma-separated EMI cycle days. Overrides config emi_cycle when provided.",
    )
    parser.add_argument(
        "--fetch-month",
        default="",
        help="Month/year used to build EMI cycle dates in YYYY-MM. Defaults to current month.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional explicit output CSV path.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=100_000,
        help="Rows fetched from Postgres per batch while writing the CSV.",
    )
    args = parser.parse_args()
    args.emi_cycle = resolve_emi_cycle(args.config_file, args.emi_cycle)
    missing = [
        name
        for name, value in {
            "PGHOST/--host": args.host,
            "PGDATABASE/--dbname": args.dbname,
            "PGUSER/--user": args.user,
            "PGPASSWORD/--password": args.password,
            "emi_cycle in config/default_config.json or EMI_CYCLE/--emi-cycle": args.emi_cycle,
        }.items()
        if not value
    ]
    if missing:
        parser.error("Missing required environment variables or CLI args: " + ", ".join(missing))
    if args.fetch_size <= 0:
        parser.error("--fetch-size must be greater than 0")
    return args


def month_bounds(source_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{source_month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return start.normalize(), end.normalize()


def default_output_file(source_month: str) -> Path:
    month_token = pd.Timestamp(f"{source_month}-01").strftime("%b%Y").upper()
    return COMMUNICATION_DATA_DIR / f"mfl_recomm_model_{month_token}_comm_data.csv"


def current_month(today: pd.Timestamp | None = None) -> str:
    return (today or pd.Timestamp.today()).strftime("%Y-%m")


def emi_cycle_dates(
    emi_cycle: list[int],
    today: pd.Timestamp | None = None,
    fetch_month: str | None = None,
) -> list[pd.Timestamp]:
    base = pd.Timestamp(f"{fetch_month}-01") if fetch_month else (today or pd.Timestamp.today()).normalize()
    month_start = base.replace(day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)
    days_in_month = (month_end - pd.Timedelta(days=1)).day
    return [month_start.replace(day=day) for day in emi_cycle if day <= days_in_month]


def build_query(schema: str, table: str) -> sql.Composed:
    table_ref = qualified_identifier(schema, table)
    where_sql = sql.SQL("TO_DATE(c.emi_date, 'DD/MM/YYYY') = ANY(%(emi_dates)s)")

    return sql.SQL(
        """
        SELECT
            c.id,
            c.apac_card_number,
            c.comm_status,
            c.communication_type,
            c.verbiage_language,
            c.risk,
            TO_DATE(c.emi_date, 'DD/MM/YYYY') AS emi_date,
            EXTRACT(HOUR FROM date_trunc('hour', c.created_date)) AS hr,
            CAST(c.created_date AS DATE) AS date,
            c.collectable_amount,
            c.created_date,
            c.last_modified_date
        FROM {table_ref} c
        WHERE {where_sql}
        """
    ).format(table_ref=table_ref, where_sql=where_sql)


def write_query_to_csv(
    conn,
    query: sql.Composed,
    params: dict,
    output_file: Path,
    fetch_size: int,
) -> int:
    total_rows = 0
    wrote_header = False
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [desc.name for desc in cur.description]
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(output_file, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
            wrote_header = True
            total_rows += len(df)

    if not wrote_header:
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
    return total_rows


def main() -> None:
    args = parse_args()
    fetch_month = args.fetch_month or current_month()
    emi_dates = emi_cycle_dates(args.emi_cycle, fetch_month=fetch_month)
    output_file = (
        ensure_parent_dir(args.output_file)
        if args.output_file
        else ensure_parent_dir(default_output_file(fetch_month))
    )

    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    query = build_query(args.schema, args.table)
    params = {"emi_dates": [date.date() for date in emi_dates]}

    with connect_db(config) as conn:
        row_count = write_query_to_csv(
            conn,
            query,
            params,
            output_file,
            args.fetch_size,
        )

    formatted_dates = ", ".join(date.strftime("%d/%m/%Y") for date in emi_dates)
    print(f"Fetched {row_count:,} rows for EMI dates: {formatted_dates}")
    print(f"Saved raw month extract to {output_file}")


if __name__ == "__main__":
    main()
