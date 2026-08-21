from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from psycopg import sql

from entity_keys import strategy_use_party_id
from env_utils import load_dotenv
from fetch_month_from_postgres import configured_prediction_month_from_emi_dates, resolve_scheduler_emi_dates
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import CASE_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Fetch one source month of digital case data from Postgres.")
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--schema",
        default=os.getenv("SOURCE_SCHEMA", "digital_collections"),
        help="Schema containing the digital cases source table.",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("CASES_TABLE", "digital_cases"),
        help="Source digital cases table.",
    )
    parser.add_argument(
        "--vertical-column",
        default=os.getenv("CASES_VERTICAL_COLUMN", "vertical"),
        help="Source column name containing the business vertical for digital cases.",
    )
    parser.add_argument(
        "--party-id-column",
        default=os.getenv("CASES_PARTY_ID_COLUMN", "party_id"),
        help="Optional source column name containing party_id for multi-loan history grouping.",
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


EMI_DATES_CONFIG_KEY = "upload.scheduler.emi-dates"


def current_month(today: pd.Timestamp | None = None) -> str:
    return (today or pd.Timestamp.today()).strftime("%Y-%m")


def _extract_scheduler_emi_cycle(raw_value: object) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        items = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [part.strip() for part in str(parsed).split(",") if part.strip()]
    cycles: list[int] = []
    seen: set[int] = set()
    for item in items:
        try:
            day = pd.to_datetime(str(item).strip(), dayfirst=True, errors="raise").day
        except Exception:
            continue
        if day not in seen:
            seen.add(day)
            cycles.append(day)
    return sorted(cycles)


def resolve_scheduler_emi_cycle(conn, *, schema: str) -> list[int]:
    candidate_refs = [sql.Identifier("data_config")]
    if schema:
        candidate_refs.append(qualified_identifier(schema, "data_config"))
    for table_ref in candidate_refs:
        try:
            row = conn.execute(sql.SQL("SELECT value FROM {} WHERE key_name = %s LIMIT 1").format(table_ref), (EMI_DATES_CONFIG_KEY,)).fetchone()
        except Exception:
            conn.rollback()
            continue
        if not row:
            continue
        cycles = _extract_scheduler_emi_cycle(row[0])
        if cycles:
            return cycles
    raise ValueError(f"Could not resolve EMI cycle from data_config key {EMI_DATES_CONFIG_KEY!r}.")


def emi_cycle_dates(
    emi_cycle: list[int],
    today: pd.Timestamp | None = None,
    fetch_month: str | None = None,
) -> list[str]:
    base = pd.Timestamp(f"{fetch_month}-01") if fetch_month else (today or pd.Timestamp.today()).normalize()
    month_start = base.replace(day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)
    days_in_month = (month_end - pd.Timedelta(days=1)).day
    return [month_start.replace(day=day).strftime("%d/%m/%Y") for day in emi_cycle if day <= days_in_month]


def default_output_file(source_month: str) -> Path:
    month_token = pd.Timestamp(f"{source_month}-01").strftime("%b%Y").upper()
    return CASE_DATA_DIR / f"digital_cases_{month_token}.csv"


def build_query(schema: str, table: str, vertical_column: str, party_id_column: str | None = None) -> sql.Composed:
    table_ref = qualified_identifier(schema, table)
    where_sql = sql.SQL("d.emi_date = ANY(%(emi_dates)s)")
    vertical_identifier = sql.Identifier(vertical_column)
    if party_id_column:
        party_id_select = sql.SQL('d.{} AS party_id').format(sql.Identifier(party_id_column))
    else:
        party_id_select = sql.SQL('NULL::TEXT AS party_id')
    return sql.SQL(
        """
        SELECT
            d.apac_card_number,
            d.risk,
            d.{vertical_column} AS vertical,
            d.outstanding_balance AS collectable_amount,
            d.emi_date,
            {party_id_select}
        FROM {table_ref} d
        WHERE {where_sql}
        """
    ).format(table_ref=table_ref, where_sql=where_sql, vertical_column=vertical_identifier, party_id_select=party_id_select)


def main() -> None:
    args = parse_args()

    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    query = build_query(args.schema, args.table, args.vertical_column, args.party_id_column if strategy_use_party_id() else None)

    with connect_db(config) as conn:
        configured_emi_dates = resolve_scheduler_emi_dates(conn, schema=args.schema)
        fetch_month = args.fetch_month or configured_prediction_month_from_emi_dates(configured_emi_dates)
        output_file = ensure_parent_dir(args.output_file) if args.output_file else ensure_parent_dir(default_output_file(fetch_month))
        emi_cycle = [int(emi_date.day) for emi_date in configured_emi_dates]
        emi_dates = emi_cycle_dates(emi_cycle, fetch_month=fetch_month)
        params = {"emi_dates": emi_dates}
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc.name for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)
    df.to_csv(output_file, index=False)

    formatted_dates = ", ".join(emi_dates)
    print(f"Fetched {len(df):,} case rows for EMI dates: {formatted_dates}")
    print(f"Saved case extract to {output_file}")


if __name__ == "__main__":
    main()
