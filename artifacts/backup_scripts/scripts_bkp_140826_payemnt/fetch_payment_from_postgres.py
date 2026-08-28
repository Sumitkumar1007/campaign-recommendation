from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


import pandas as pd
from psycopg import sql

# Reusing the exact same imports from your project architecture
# pyrefly: ignore [missing-import]
from env_utils import load_dotenv
# pyrefly: ignore [missing-import]
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
# pyrefly: ignore [missing-import]
from project_paths import PAYMENT_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Fetch past 3 months of payment data from Postgres, split into monthly files."
    )
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument(
        "--schema",
        default=os.getenv("SOURCE_SCHEMA", "digital_collections"),
        help="Schema containing the payments source table.",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("PAYMENTS_SOURCE_TABLE", "payment"),
        help="Source payments table.",
    )
    parser.add_argument(
        "--fetch-month",
        default="",
        help="Target end month in YYYY-MM. Defaults to current month.",
    )
    parser.add_argument(
        "--history-months",
        type=int,
        default=1,
        help="Number of months of history to fetch backwards from target month.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional base output CSV path (month token will be appended).",
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
    if args.history_months <= 0:
        parser.error("--history-months must be greater than 0")
    
    return args


def get_monthly_cycles(fetch_month: str | None = None, history_months: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Generates billing cycles starting from the target fetch_month backwards.
    Each cycle: 27th of previous month (00:00:00) to 28th of target month (23:59:59).
    """
    today = pd.Timestamp.today().normalize()
    base = pd.Timestamp(f"{fetch_month}-01") if fetch_month else today.replace(day=1)
    
    cycles = []
    # Loop history_months times to get target month and prior months
    for i in range(history_months):
        current_target = base - pd.DateOffset(months=i)
        prev_target = current_target - pd.DateOffset(months=1)
        
        start_date = prev_target.replace(day=27, hour=0, minute=0, second=0)
        end_date = current_target.replace(day=28, hour=23, minute=59, second=59)
        
        month_token = current_target.strftime("%b%Y").upper() # e.g., JUL2026
        cycles.append((start_date, end_date, month_token))
        
    return cycles


def default_output_file(month_token: str) -> Path:
    """
    Saves in a 'payments' subdirectory inside the default communication directory.
    """
    payments_dir = PAYMENT_DATA_DIR / "payments"
    return payments_dir / f"payment_data_{month_token}.csv"


def build_payment_query(schema: str, table: str) -> sql.Composed:
    table_ref = qualified_identifier(schema, table)

    return sql.SQL(
        """
        SELECT 
            apac_card_number, 
            reference_number, 
            created_date, 
            last_modified_date, 
            status,
            payment_datetime,
            EXTRACT(HOUR FROM payment_datetime::timestamp) AS hr,
            CAST(payment_datetime AS DATE) AS date
        FROM {table_ref}
        WHERE payment_datetime::date >= %s AND payment_datetime::date <= %s
        """
    ).format(table_ref=table_ref)


def write_query_to_csv(
    conn,
    query: sql.Composed,
    params: tuple,
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

    # In case there were 0 rows, we still want an empty CSV with headers
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
    
    query = build_payment_query(args.schema, args.table)
    fetch_month = args.fetch_month or pd.Timestamp.today().strftime("%Y-%m")
    cycles = get_monthly_cycles(fetch_month, args.history_months)

    audit_summaries = []

    with connect_db(config) as conn:
        for start_date, end_date, month_token in cycles:
            
            # Determine output file path for this specific month
            if args.output_file:
                base_path = Path(args.output_file)
                output_file = ensure_parent_dir(
                    base_path.parent / f"{base_path.stem}_{month_token}{base_path.suffix}"
                )
            else:
                output_file = ensure_parent_dir(default_output_file(month_token))

            params = (start_date, end_date)
            row_count = write_query_to_csv(
                conn,
                query,
                params,
                output_file,
                args.fetch_size,
            )
            
            # Log successful extraction for this month
            print(f"Fetched {row_count:,} rows for cycle: {start_date.date()} to {end_date.date()}")
            print(f"Saved to {output_file}")
            
            audit_summaries.append({
                "month_token": month_token,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "total_rows_fetched": row_count,
                "output_file": str(output_file)
            })

    # Print final JSON audit log
    print("Fetch audit summary | " + json.dumps(
        {
            "base_fetch_month": fetch_month,
            "cycles_processed": audit_summaries
        }, 
        sort_keys=True
    ))


if __name__ == "__main__":
    main()