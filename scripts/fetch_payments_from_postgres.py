from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from psycopg import sql

from env_utils import load_dotenv
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import PAYMENT_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Fetch payment rows for one source month from Postgres.")
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--schema", default=os.getenv("SOURCE_SCHEMA", "digital_collections"))
    parser.add_argument("--table", default=os.getenv("PAYMENT_TABLE_NAME", "payment"))
    parser.add_argument("--source-month", required=True, help="Payment month to fetch in YYYY-MM format.")
    parser.add_argument("--output-file", default="", help="Optional output CSV path.")
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
    return args


def default_output_file(source_month: str) -> Path:
    month_token = pd.Timestamp(f"{source_month}-01").strftime("%b%Y").upper()
    return PAYMENT_DATA_DIR / f"payment_data_{month_token}.csv"


def build_query(schema: str, table: str) -> sql.Composed:
    table_ref = qualified_identifier(schema, table)
    return sql.SQL(
        """
        SELECT
            apac_card_number,
            amount,
            payment_datetime
        FROM {table_ref}
        WHERE payment_datetime::timestamp >= %(month_start)s
            AND payment_datetime::timestamp < %(next_month_start)s
            AND apac_card_number IS NOT NULL
        """
    ).format(table_ref=table_ref)


def main() -> None:
    args = parse_args()
    source_period = pd.Period(args.source_month, freq="M")
    month_start = source_period.to_timestamp().to_pydatetime()
    next_month_start = (source_period + 1).to_timestamp().to_pydatetime()
    output_file = ensure_parent_dir(args.output_file or default_output_file(args.source_month))
    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    query = build_query(args.schema, args.table)
    with connect_db(config) as conn:
        cursor = conn.execute(
            query,
            {"month_start": month_start, "next_month_start": next_month_start},
        )
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(
        "Fetched payment rows | "
        f"source_month={args.source_month} schema={args.schema} table={args.table} "
        f"rows={len(df)} output_file={output_file}"
    )


if __name__ == "__main__":
    main()
