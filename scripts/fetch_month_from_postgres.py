from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from psycopg import sql

from entity_keys import strategy_use_party_id
from env_utils import load_dotenv
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from project_paths import COMMUNICATION_DATA_DIR, ensure_parent_dir


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
        "--vertical-column",
        default=os.getenv("COMMUNICATION_VERTICAL_COLUMN", "vertical"),
        help="Source column name containing the business vertical for communications.",
    )
    parser.add_argument(
        "--party-id-column",
        default=os.getenv("COMMUNICATION_PARTY_ID_COLUMN", "party_id"),
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
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=100_000,
        help="Rows fetched from Postgres per batch while writing the CSV.",
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
    if args.fetch_size <= 0:
        parser.error("--fetch-size must be greater than 0")
    return args


def month_bounds(source_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{source_month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return start.normalize(), end.normalize()


def default_output_file(source_month: str, cycle_day: int | None = None) -> Path:
    month_token = pd.Timestamp(f"{source_month}-01").strftime("%b%Y").upper()
    if cycle_day is None:
        return COMMUNICATION_DATA_DIR / f"comm_data_{month_token}.csv"
    return COMMUNICATION_DATA_DIR / f"comm_data_{month_token}_{int(cycle_day)}.csv"


def output_file_for_cycle(source_month: str, cycle_day: int, explicit_output_file: str = "") -> Path:
    if not explicit_output_file:
        return default_output_file(source_month, cycle_day)
    base_output_file = Path(explicit_output_file)
    suffix = base_output_file.suffix or ".csv"
    stem = base_output_file.stem
    cycle_suffix = f"_{int(cycle_day)}"
    if stem.endswith(cycle_suffix):
        return base_output_file.with_suffix(suffix)
    return base_output_file.with_name(f"{stem}{cycle_suffix}{suffix}")


EMI_DATES_CONFIG_KEY = "upload.scheduler.emi-dates"


def safe_emi_date_sql(alias: str = "c") -> sql.Composed:
    alias_identifier = sql.Identifier(alias)
    return sql.SQL(
        """
        CASE
            WHEN {alias}.emi_date ~ '^[0-9]{{2}}/[0-9]{{2}}/[0-9]{{4}}$'
             AND split_part({alias}.emi_date, '/', 2)::int BETWEEN 1 AND 12
             AND split_part({alias}.emi_date, '/', 1)::int BETWEEN 1 AND 31
             AND split_part({alias}.emi_date, '/', 3)::int BETWEEN 1900 AND 2999
             AND split_part({alias}.emi_date, '/', 1)::int <= EXTRACT(
                   DAY FROM (
                       make_date(
                           split_part({alias}.emi_date, '/', 3)::int,
                           split_part({alias}.emi_date, '/', 2)::int,
                           1
                       ) + INTERVAL '1 month - 1 day'
                   )
               )
            THEN make_date(
                split_part({alias}.emi_date, '/', 3)::int,
                split_part({alias}.emi_date, '/', 2)::int,
                split_part({alias}.emi_date, '/', 1)::int
            )
            ELSE NULL
        END
        """
    ).format(alias=alias_identifier)


def current_month(today: pd.Timestamp | None = None) -> str:
    return (today or pd.Timestamp.today()).strftime("%Y-%m")


def _extract_scheduler_emi_dates(raw_value: object) -> list[pd.Timestamp]:
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
    emi_dates: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for item in items:
        try:
            emi_date = pd.to_datetime(str(item).strip(), dayfirst=True, errors="raise").normalize()
        except Exception:
            continue
        if emi_date not in seen:
            seen.add(emi_date)
            emi_dates.append(emi_date)
    emi_dates.sort()
    return emi_dates


def _extract_scheduler_emi_cycle(raw_value: object) -> list[int]:
    cycles = {int(emi_date.day) for emi_date in _extract_scheduler_emi_dates(raw_value)}
    return sorted(cycles)


def resolve_scheduler_emi_dates(conn, *, schema: str) -> list[pd.Timestamp]:
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
        emi_dates = _extract_scheduler_emi_dates(row[0])
        if emi_dates:
            return emi_dates
    raise ValueError(f"Could not resolve EMI dates from data_config key {EMI_DATES_CONFIG_KEY!r}.")


def configured_prediction_month_from_emi_dates(emi_dates: list[pd.Timestamp]) -> str:
    if not emi_dates:
        raise ValueError("No EMI dates configured in data_config.")
    return emi_dates[-1].strftime("%Y-%m")


def configured_source_month_from_emi_dates(emi_dates: list[pd.Timestamp]) -> str:
    predict_period = pd.Period(configured_prediction_month_from_emi_dates(emi_dates), freq="M")
    return str(predict_period - 1)


def resolve_scheduler_emi_cycle(conn, *, schema: str) -> list[int]:
    return [int(emi_date.day) for emi_date in resolve_scheduler_emi_dates(conn, schema=schema)]


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


def build_query(schema: str, table: str, vertical_column: str, party_id_column: str | None = None) -> sql.Composed:
    table_ref = qualified_identifier(schema, table)
    parsed_emi_date = safe_emi_date_sql("src")
    vertical_identifier = sql.Identifier(vertical_column)
    if party_id_column:
        party_id_select = sql.SQL('src.{} AS party_id').format(sql.Identifier(party_id_column))
    else:
        party_id_select = sql.SQL('NULL::TEXT AS party_id')

    return sql.SQL(
        """
        WITH parsed AS (
            SELECT
                src.id,
                src.apac_card_number,
                src.comm_status,
                src.communication_type,
                src.verbiage_language,
                src.{vertical_column} AS vertical,
                src.risk,
                src.disposition,
                {party_id_select},
                {parsed_emi_date} AS emi_date,
                EXTRACT(HOUR FROM date_trunc('hour', src.created_date)) AS hr,
                CAST(src.created_date AS DATE) AS date,
                src.collectable_amount,
                src.created_date,
                src.last_modified_date
            FROM {table_ref} src
        )
        SELECT
            id,
            apac_card_number,
            comm_status,
            communication_type,
            verbiage_language,
            vertical,
            risk,
            disposition,
            party_id,
            emi_date,
            hr,
            date,
            collectable_amount,
            created_date,
            last_modified_date
        FROM parsed
        WHERE emi_date = ANY(%(emi_dates)s)
            AND LOWER(apac_card_number) NOT LIKE '%%test%%'
            AND apac_card_number NOT LIKE 'PRVCMP%%'
            AND apac_card_number NOT IN ('1234', '2345', '3453')
        """
    ).format(
        table_ref=table_ref,
        parsed_emi_date=parsed_emi_date,
        vertical_column=vertical_identifier,
        party_id_select=party_id_select,
    )


def build_audit_summary(conn, *, schema: str, table: str, emi_dates: list[pd.Timestamp]) -> dict[str, object]:
    table_ref = qualified_identifier(schema, table)
    parsed_emi_date = safe_emi_date_sql("src")
    params = {"emi_dates": [date.date() for date in emi_dates]}

    aggregate_query = sql.SQL(
        """
        WITH parsed AS (
            SELECT
                NULLIF(BTRIM(src.emi_date), '') AS raw_emi_date,
                NULLIF(BTRIM(src.communication_type), '') AS communication_type,
                {parsed_emi_date} AS parsed_emi_date
            FROM {table_ref} src
        )
        SELECT
            COUNT(*) AS total_source_rows,
            COUNT(*) FILTER (WHERE parsed_emi_date IS NULL) AS invalid_emi_date_rows,
            COUNT(*) FILTER (WHERE parsed_emi_date IS NOT NULL) AS valid_emi_date_rows,
            COUNT(*) FILTER (WHERE parsed_emi_date = ANY(%(emi_dates)s)) AS matched_emi_date_rows,
            COUNT(*) FILTER (WHERE parsed_emi_date IS NOT NULL AND parsed_emi_date <> ALL(%(emi_dates)s)) AS excluded_emi_date_rows
        FROM parsed
        """
    ).format(table_ref=table_ref, parsed_emi_date=parsed_emi_date)
    aggregate_row = conn.execute(aggregate_query, params).fetchone()

    invalid_values_query = sql.SQL(
        """
        WITH parsed AS (
            SELECT
                NULLIF(BTRIM(src.emi_date), '') AS raw_emi_date,
                {parsed_emi_date} AS parsed_emi_date
            FROM {table_ref} src
        )
        SELECT raw_emi_date, COUNT(*) AS row_count
        FROM parsed
        WHERE parsed_emi_date IS NULL
          AND raw_emi_date IS NOT NULL
        GROUP BY raw_emi_date
        ORDER BY row_count DESC, raw_emi_date
        LIMIT 20
        """
    ).format(table_ref=table_ref, parsed_emi_date=parsed_emi_date)
    invalid_rows = conn.execute(invalid_values_query).fetchall()

    excluded_values_query = sql.SQL(
        """
        WITH parsed AS (
            SELECT
                NULLIF(BTRIM(src.emi_date), '') AS raw_emi_date,
                {parsed_emi_date} AS parsed_emi_date
            FROM {table_ref} src
        )
        SELECT raw_emi_date, COUNT(*) AS row_count
        FROM parsed
        WHERE parsed_emi_date IS NOT NULL
          AND parsed_emi_date <> ALL(%(emi_dates)s)
          AND raw_emi_date IS NOT NULL
        GROUP BY raw_emi_date
        ORDER BY row_count DESC, raw_emi_date
        LIMIT 20
        """
    ).format(table_ref=table_ref, parsed_emi_date=parsed_emi_date)
    excluded_rows = conn.execute(excluded_values_query, params).fetchall()

    matched_type_query = sql.SQL(
        """
        WITH parsed AS (
            SELECT
                NULLIF(BTRIM(src.communication_type), '') AS communication_type,
                {parsed_emi_date} AS parsed_emi_date
            FROM {table_ref} src
        )
        SELECT COALESCE(communication_type, 'NULL') AS communication_type, COUNT(*) AS row_count
        FROM parsed
        WHERE parsed_emi_date = ANY(%(emi_dates)s)
        GROUP BY COALESCE(communication_type, 'NULL')
        ORDER BY row_count DESC, communication_type
        """
    ).format(table_ref=table_ref, parsed_emi_date=parsed_emi_date)
    matched_type_rows = conn.execute(matched_type_query, params).fetchall()

    return {
        "row_counts": {
            "total_source_rows": int(aggregate_row[0] or 0),
            "invalid_emi_date_rows": int(aggregate_row[1] or 0),
            "valid_emi_date_rows": int(aggregate_row[2] or 0),
            "matched_emi_date_rows": int(aggregate_row[3] or 0),
            "excluded_emi_date_rows": int(aggregate_row[4] or 0),
        },
        "matched_communication_type_counts": {
            str(row[0]): int(row[1]) for row in matched_type_rows
        },
        "top_invalid_emi_date_values": {
            str(row[0]): int(row[1]) for row in invalid_rows
        },
        "top_excluded_emi_date_values": {
            str(row[0]): int(row[1]) for row in excluded_rows
        },
    }


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

    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    
    # Note: Using the new build_query that includes args.vertical_column
    query = build_query(args.schema, args.table, args.vertical_column, args.party_id_column if strategy_use_party_id() else None)

    with connect_db(config) as conn:
        configured_emi_dates = resolve_scheduler_emi_dates(conn, schema=args.schema)
        fetch_month = args.fetch_month or configured_source_month_from_emi_dates(configured_emi_dates)
        
        # Determine output file (single file per month, no cycle suffix)
        if args.output_file:
            output_file = ensure_parent_dir(Path(args.output_file))
        else:
            output_file = ensure_parent_dir(default_output_file(fetch_month))
            
        emi_cycle = [int(emi_date.day) for emi_date in configured_emi_dates]
        emi_dates = emi_cycle_dates(emi_cycle, fetch_month=fetch_month)
        audit_summary = build_audit_summary(conn, schema=args.schema, table=args.table, emi_dates=emi_dates)
        
        # Pass all dates at once
        params = {"emi_dates": [date.date() for date in emi_dates]}
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
    print(
        "Fetch audit summary | "
        + json.dumps(
            {
                "fetch_month": fetch_month,
                "emi_dates": [date.strftime("%d/%m/%Y") for date in emi_dates],
                "output_file": str(output_file),
                **audit_summary,
            },
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    main()