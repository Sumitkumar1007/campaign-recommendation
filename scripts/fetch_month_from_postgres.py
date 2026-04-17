from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from psycopg import sql

from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import COMMUNICATION_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one source month of communication data from Postgres."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--schema",
        default="digital_collections",
        help="Schema containing the communications source table.",
    )
    parser.add_argument(
        "--table",
        default="communications",
        help="Source communications table.",
    )
    parser.add_argument(
        "--source-month",
        required=True,
        help="Month to fetch in YYYY-MM format, for example 2026-04.",
    )
    parser.add_argument(
        "--filter-on",
        choices=["emi_date", "created_date"],
        default="emi_date",
        help="Which date field defines the source month window.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional explicit output CSV path.",
    )
    return parser.parse_args()


def month_bounds(source_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{source_month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return start.normalize(), end.normalize()


def default_output_file(source_month: str) -> Path:
    month_token = pd.Timestamp(f"{source_month}-01").strftime("%b%Y").upper()
    return COMMUNICATION_DATA_DIR / f"mfl_recomm_model_{month_token}_comm_data.csv"


def build_query(schema: str, table: str, filter_on: str) -> sql.Composed:
    table_ref = qualified_identifier(schema, table)
    if filter_on == "emi_date":
        where_sql = sql.SQL(
            "TO_DATE(c.emi_date, 'DD/MM/YYYY') >= %(start_date)s "
            "AND TO_DATE(c.emi_date, 'DD/MM/YYYY') < %(end_date)s"
        )
    else:
        where_sql = sql.SQL(
            "c.created_date >= %(start_date)s "
            "AND c.created_date < %(end_date)s"
        )

    return sql.SQL(
        """
        SELECT
            c.id,
            c.apac_card_number,
            c.comm_status,
            c.communication_type,
            c.verbiage_language,
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


def main() -> None:
    args = parse_args()
    start_date, end_date = month_bounds(args.source_month)
    output_file = (
        ensure_parent_dir(args.output_file)
        if args.output_file
        else ensure_parent_dir(default_output_file(args.source_month))
    )

    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    query = build_query(args.schema, args.table, args.filter_on)

    with connect_db(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                {"start_date": start_date.date(), "end_date": end_date.date()},
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)

    df.to_csv(output_file, index=False)
    print(f"Fetched {len(df):,} rows for {args.source_month}")
    print(f"Saved raw month extract to {output_file}")


if __name__ == "__main__":
    main()
