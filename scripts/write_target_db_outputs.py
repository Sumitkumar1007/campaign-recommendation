from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from env_utils import load_dotenv
from pipeline_common import resolve_emi_cycle
from postgres_utils import PostgresConfig, connect_db
from run_monthly_inference_pipeline import (
    _extract_campaign_vendors,
    _prepare_campaign_outputs,
    month_label,
    store_api_audit_log,
    store_campaign_mappings,
    store_campaign_recommendations,
    store_prediction_snapshots,
)


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=(
            "Write existing local inference output into target Postgres tables "
            "without re-running source fetch or model inference."
        )
    )
    parser.add_argument("--host", default=os.getenv("TARGET_PGHOST") or os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TARGET_PGPORT") or os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("TARGET_PGDATABASE") or os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("TARGET_PGUSER") or os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("TARGET_PGPASSWORD") or os.getenv("PGPASSWORD"))
    parser.add_argument("--target-schema", default=os.getenv("TARGET_SCHEMA", "digital_collections"))
    parser.add_argument("--prediction-table", default=os.getenv("PREDICTION_TABLE", "ai_ml_recommendations_data"))
    parser.add_argument("--audit-table", default=os.getenv("AUDIT_TABLE", "api_audit_log"))
    parser.add_argument("--campaign-table", default=os.getenv("CAMPAIGN_TABLE", "ai_ml_campaign_recommendations"))
    parser.add_argument("--campaign-mapping-table", default=os.getenv("CAMPAIGN_MAPPING_TABLE", "ai_ml_campaign_mapping"))
    parser.add_argument("--prediction-file", required=True, help="Local prediction CSV produced by inference.")
    parser.add_argument("--source-month", required=True, help="Source month as YYYY-MM, e.g. 2026-04.")
    parser.add_argument("--predict-month", required=True, help="Prediction month as YYYY-MM, e.g. 2026-05.")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "catboost_3m"))
    parser.add_argument("--config-file", default=os.getenv("CONFIG_FILE", str(Path(__file__).resolve().parents[1] / "config" / "default_config.json")))
    parser.add_argument("--campaign-vertical", default=os.getenv("CAMPAIGN_VERTICAL", "LAP"))
    parser.add_argument("--campaign-vendor", default=os.getenv("CAMPAIGN_VENDOR", "prutech-cpass"))
    parser.add_argument("--connect-timeout", type=int, default=int(os.getenv("TARGET_PGCONNECT_TIMEOUT", "10")))
    args = parser.parse_args()
    missing = [
        name
        for name, value in {
            "TARGET_PGHOST/--host": args.host,
            "TARGET_PGDATABASE/--dbname": args.dbname,
            "TARGET_PGUSER/--user": args.user,
            "TARGET_PGPASSWORD/--password": args.password,
        }.items()
        if not value
    ]
    if missing:
        parser.error("Missing required target DB args: " + ", ".join(missing))
    return args


def main() -> None:
    args = parse_args()
    prediction_file = Path(args.prediction_file).resolve()
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    os.environ["PGCONNECT_TIMEOUT"] = str(args.connect_timeout)

    source_month_label = month_label(args.source_month)
    prediction_month_label = month_label(args.predict_month)
    model_name = args.model
    emi_cycles = resolve_emi_cycle(args.config_file, os.getenv("EMI_CYCLE", ""))
    vendors = _extract_campaign_vendors(args.campaign_vendor) or [args.campaign_vendor]
    started = time.time()

    campaign_df, mapping_df = _prepare_campaign_outputs(
        prediction_file,
        source_month_label=source_month_label,
        prediction_month_label=prediction_month_label,
        model_name=model_name,
        emi_cycles=emi_cycles,
        vertical=args.campaign_vertical,
        vendors=vendors,
    )

    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )

    with connect_db(config) as conn:
        prediction_rows = store_prediction_snapshots(
            conn,
            args.target_schema,
            args.prediction_table,
            prediction_file,
            prediction_month_label,
            model_name,
        )
        campaign_rows = store_campaign_recommendations(
            conn,
            args.target_schema,
            args.campaign_table,
            campaign_df,
            source_month_label=source_month_label,
            prediction_month_label=prediction_month_label,
            model_name=model_name,
        )
        mapping_rows = store_campaign_mappings(
            conn,
            args.target_schema,
            args.campaign_mapping_table,
            mapping_df,
            source_month_label=source_month_label,
            prediction_month_label=prediction_month_label,
            model_name=model_name,
        )
        store_api_audit_log(
            conn,
            args.target_schema,
            args.audit_table,
            model_name=model_name,
            source_month=source_month_label,
            prediction_month=prediction_month_label,
            status="SUCCESS",
            prediction_completed_count=prediction_rows,
            prediction_failed_count=0,
            failed_reason=None,
            duration_seconds=time.time() - started,
            prediction_table=args.prediction_table,
            prediction_file=prediction_file,
        )
        conn.commit()

    print(f"Target DB write complete: {args.host}:{args.port}/{args.dbname}")
    print(f"Prediction rows: {prediction_rows}")
    print(f"Campaign rows: {campaign_rows}")
    print(f"Mapping rows: {mapping_rows}")
    print(f"Audit table: {args.target_schema}.{args.audit_table}")


if __name__ == "__main__":
    main()
