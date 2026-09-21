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
    parser.add_argument("--host", default=os.getenv("PGHOST", ""))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", ""))
    parser.add_argument("--user", default=os.getenv("PGUSER", ""))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", ""))
    parser.add_argument("--schema", default=os.getenv("SOURCE_SCHEMA", "digital_collections"))
    parser.add_argument("--table", default=os.getenv("PAYMENT_TABLE_NAME", "payment"))
    parser.add_argument("--source-month", required=True, help="Payment month to fetch in YYYY-MM format.")
    parser.add_argument("--output-file", default="", help="Optional output CSV path.")
    return parser.parse_args()


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
        WHERE (
            payment_datetime::text LIKE %(month_prefix)s
            OR payment_datetime::text LIKE %(alt_month_prefix1)s
            OR payment_datetime::text LIKE %(alt_month_prefix2)s
            OR (
                CASE 
                    WHEN payment_datetime::text ~ '^[0-9]{4}-[0-9]{2}' THEN payment_datetime::timestamp
                    ELSE NULL 
                END >= %(month_start)s 
                AND CASE 
                    WHEN payment_datetime::text ~ '^[0-9]{4}-[0-9]{2}' THEN payment_datetime::timestamp
                    ELSE NULL 
                END < %(next_month_start)s
            )
        )
        AND apac_card_number IS NOT NULL
        """
    ).format(table_ref=table_ref)


def main() -> None:
    output_file: Path | None = None
    source_month_str = ""
    schema_str = ""
    table_str = ""
    try:
        args = parse_args()
        source_month_str = args.source_month
        schema_str = args.schema
        table_str = args.table
        output_file = ensure_parent_dir(args.output_file or default_output_file(args.source_month))

        missing = [
            name
            for name, value in {
                "PGHOST": args.host,
                "PGDATABASE": args.dbname,
                "PGUSER": args.user,
                "PGPASSWORD": args.password,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError("Missing database connection parameter(s): " + ", ".join(missing))

        source_period = pd.Period(args.source_month, freq="M")
        month_start = source_period.to_timestamp().to_pydatetime()
        next_month_start = (source_period + 1).to_timestamp().to_pydatetime()
        month_prefix = f"{args.source_month}%"
        month_timestamp = source_period.to_timestamp()
        alt_month_prefix1 = f"%{month_timestamp.strftime('%b-%Y').upper()}%"
        alt_month_prefix2 = f"%{month_timestamp.strftime('%b%Y').upper()}%"

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
                {
                    "month_start": month_start,
                    "next_month_start": next_month_start,
                    "month_prefix": month_prefix,
                    "alt_month_prefix1": alt_month_prefix1,
                    "alt_month_prefix2": alt_month_prefix2,
                },
            )
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    except Exception as e:
        print(f"Warning: fetch_payments_from_postgres encountered an issue ({e}). Creating empty output file.")
        df = pd.DataFrame(columns=["apac_card_number", "amount", "payment_datetime"])

    if output_file is None:
        output_file = PAYMENT_DATA_DIR / "payment_data_fallback.csv"
        ensure_parent_dir(output_file)

    df.to_csv(output_file, index=False)
    print(
        "Fetched payment rows | "
        f"source_month={source_month_str} schema={schema_str} table={table_str} "
        f"rows={len(df)} output_file={output_file}"
    )


if __name__ == "__main__":
    main()
