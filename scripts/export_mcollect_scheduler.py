from __future__ import annotations

import argparse
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from psycopg import sql

from env_utils import load_dotenv
from postgres_utils import PostgresConfig, connect_db, qualified_identifier
from quartz_job_data import build_mcollect_job_data
from run_monthly_inference_pipeline import month_label


SCHED_NAME = "quartzScheduler"
TRIGGER_GROUP = "DEFAULT"
TIME_ZONE_ID = "Asia/Kolkata"
ACTOR = "campaign-model"

JOB_CLASS_BY_MODE = {
    "SMS": "com.company.api_test.jobs.BatchSMSJob",
    "WHATSAPP": "com.company.api_test.jobs.BatchWhatsappJob",
    "VOICE": "com.company.api_test.jobs.BatchVoiceJob",
}
JOB_GROUP_BY_MODE = {
    "SMS": "SMS",
    "WHATSAPP": "WHATSAPP",
    "VOICE": "VOICE",
}
DIGITAL_RULE_MODE_BY_MODE = {
    "SMS": "SMS",
    "WHATSAPP": "WHATSAPP",
    "VOICE": "VOICE",
}


@dataclass(frozen=True)
class CronTriggerSpec:
    trigger_name: str
    cron_expression: str
    next_fire_time_ms: int
    start_time_ms: int
    end_time_ms: int
    trigger_state: str


@dataclass(frozen=True)
class ExportSummary:
    campaigns: int
    datasets: int
    templates: int
    jobs: int
    triggers: int


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=(
            "Export ai_ml_campaign_recommendations into MCollect Digital dataset, "
            "digital_rules, and Quartz scheduler tables. Dry-run by default."
        )
    )
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--schema", default=os.getenv("TARGET_SCHEMA", "digital_collections"))
    parser.add_argument("--campaign-table", default=os.getenv("CAMPAIGN_TABLE", "ai_ml_campaign_recommendations"))
    parser.add_argument("--mapping-table", default=os.getenv("CAMPAIGN_MAPPING_TABLE", "ai_ml_campaign_mapping"))
    parser.add_argument("--source-month", required=True, help="Source month as YYYY-MM or MON-YYYY, e.g. 2026-04/APR-2026.")
    parser.add_argument("--prediction-month", required=True, help="Prediction month as YYYY-MM or MON-YYYY, e.g. 2026-05/MAY-2026.")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "catboost_3m"))
    parser.add_argument("--trigger-state", choices=["WAITING", "PAUSED"], default="WAITING")
    parser.add_argument("--write", action="store_true", help="Actually write to MCollect tables. Without this, only prints counts.")
    args = parser.parse_args()
    missing = [
        key
        for key, value in {
            "PGHOST/--host": args.host,
            "PGDATABASE/--dbname": args.dbname,
            "PGUSER/--user": args.user,
            "PGPASSWORD/--password": args.password,
        }.items()
        if not value
    ]
    if missing:
        parser.error("Missing required environment variables or CLI args: " + ", ".join(missing))
    args.source_month = normalize_month_label(args.source_month)
    args.prediction_month = normalize_month_label(args.prediction_month)
    return args


def normalize_month_label(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return month_label(value)
    match = re.fullmatch(r"[A-Za-z]{3}-\d{4}", value)
    if not match:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM or MON-YYYY.")
    mon, year = value.split("-", 1)
    return f"{mon.upper()}-{year}"


def month_start_from_label(month: str) -> datetime:
    mon, year_text = normalize_month_label(month).split("-", 1)
    month_number = datetime.strptime(mon, "%b").month
    return datetime(int(year_text), month_number, 1)


def relative_campaign_day_to_date(day_label: str, prediction_month_label: str, emi_cycle: int) -> datetime:
    day_label = day_label.strip().upper()
    if day_label == "D":
        offset = 0
    else:
        match = re.fullmatch(r"D([+-]\d+)", day_label)
        if not match:
            raise ValueError(f"Invalid campaign day label {day_label!r}.")
        offset = int(match.group(1))
    prediction_month_start = month_start_from_label(prediction_month_label)
    emi_date = prediction_month_start.replace(day=int(emi_cycle))
    return emi_date + timedelta(days=offset)


def parse_campaign_times(time_value: str) -> list[time]:
    times: list[time] = []
    for part in str(time_value).split(","):
        part = part.strip()
        if not part:
            continue
        times.append(datetime.strptime(part, "%H:%M:%S").time())
    if not times:
        raise ValueError(f"No valid campaign times in {time_value!r}.")
    return sorted(set(times))


def build_cron_trigger_specs(row: pd.Series, *, trigger_state: str) -> list[CronTriggerSpec]:
    zone = ZoneInfo(TIME_ZONE_ID)
    now_ms = int(datetime.now(tz=zone).timestamp() * 1000)
    campaign_dates = [
        relative_campaign_day_to_date(day, str(row["prediction_month"]), int(row["emi_cycle"]))
        for day in str(row["date"]).split(",")
        if day.strip()
    ]
    campaign_times = parse_campaign_times(str(row["time"]))
    if not campaign_dates:
        raise ValueError(f"No valid campaign dates for {row['name']!r}.")

    datetimes_by_month: dict[tuple[int, int], list[datetime]] = defaultdict(list)
    for campaign_date in campaign_dates:
        for campaign_time in campaign_times:
            scheduled = datetime.combine(campaign_date.date(), campaign_time, tzinfo=zone)
            datetimes_by_month[(scheduled.year, scheduled.month)].append(scheduled)

    specs: list[CronTriggerSpec] = []
    for (_, month_number), scheduled_datetimes in sorted(datetimes_by_month.items()):
        day_values = sorted({scheduled.day for scheduled in scheduled_datetimes})
        hour_values = sorted({scheduled.hour for scheduled in scheduled_datetimes})
        minute_values = sorted({scheduled.minute for scheduled in scheduled_datetimes})
        first_fire = min(scheduled_datetimes)
        last_fire = max(scheduled_datetimes)
        cron = (
            f"0 {','.join(str(value) for value in minute_values)} "
            f"{','.join(str(value) for value in hour_values)} "
            f"{','.join(str(value) for value in day_values)} {month_number} ?"
        )
        specs.append(
            CronTriggerSpec(
                trigger_name=str(uuid.uuid4()),
                cron_expression=cron,
                next_fire_time_ms=int(first_fire.timestamp() * 1000),
                start_time_ms=now_ms,
                end_time_ms=int((last_fire + timedelta(days=1)).timestamp() * 1000),
                trigger_state=trigger_state,
            )
        )
    return specs


def language_from_template(template_name: str) -> str:
    tokenized = re.split(r"[ _]+", template_name.strip())
    if not tokenized:
        return "English"
    language = tokenized[-1].title()
    if language.upper() == "ENGLISH":
        return "English"
    return language


def placeholder_verbiage(row: pd.Series) -> str:
    mode = str(row["mode"]).upper()
    language = language_from_template(str(row["template_name"]))
    if mode == "VOICE":
        return "NA"
    return (
        f"AIML generated {str(row['due_type']).lower()} {mode.lower()} template "
        f"for {str(row['vertical']).title()} in {language}."
    )


def dataset_query_for(row: pd.Series, *, schema: str, campaign_table: str, mapping_table: str) -> str:
    dataset_name = str(row["dataset_name"]).replace("'", "''")
    source_month = str(row["source_month"]).replace("'", "''")
    prediction_month = str(row["prediction_month"]).replace("'", "''")
    model_name = str(row["model_name"]).replace("'", "''")
    schema_prefix = f"{schema}." if schema else ""
    return (
        "select distinct dc.* "
        f"from {schema_prefix}digital_cases dc "
        f"join {schema_prefix}{mapping_table} aiml_map on aiml_map.loan_number = dc.apac_card_number "
        f"join {schema_prefix}{campaign_table} aiml_campaign on aiml_campaign.name = aiml_map.campaign_name "
        f"where aiml_campaign.dataset_name = '{dataset_name}' "
        f"and aiml_campaign.source_month = '{source_month}' "
        f"and aiml_campaign.prediction_month = '{prediction_month}' "
        f"and aiml_campaign.model_name = '{model_name}'"
    )


def fetch_campaign_rows(conn, *, schema: str, campaign_table: str, source_month: str, prediction_month: str, model_name: str) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT name, mode, date, time, template_name, dataset_name, vendor, active,
               source_month, prediction_month, model_name, emi_cycle, risk, vertical,
               campaign_type, due_type
        FROM {table_ref}
        WHERE source_month = %s
          AND prediction_month = %s
          AND model_name = %s
        ORDER BY name
        """
    ).format(table_ref=qualified_identifier(schema, campaign_table))
    with conn.cursor() as cur:
        cur.execute(query, (source_month, prediction_month, model_name))
        columns = [description.name for description in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


class DirectDatabaseMcollectPublisher:
    """Temporary direct-table publisher until MCollect exposes a scheduler API."""

    def __init__(self, conn, *, schema: str, campaign_table: str, mapping_table: str, trigger_state: str):
        self.conn = conn
        self.schema = schema
        self.campaign_table = campaign_table
        self.mapping_table = mapping_table
        self.trigger_state = trigger_state

    def publish(self, campaigns: pd.DataFrame) -> ExportSummary:
        if campaigns.empty:
            return ExportSummary(campaigns=0, datasets=0, templates=0, jobs=0, triggers=0)

        now = datetime.now(timezone.utc)
        self._sync_sequences()
        self._upsert_datasets(campaigns, now=now)
        self._upsert_templates(campaigns, now=now)
        trigger_count = self._replace_quartz_jobs(campaigns)
        return ExportSummary(
            campaigns=len(campaigns),
            datasets=int(campaigns["dataset_name"].nunique()),
            templates=int(campaigns["template_name"].nunique()),
            jobs=len(campaigns),
            triggers=trigger_count,
        )

    def _sync_sequences(self) -> None:
        for table_name in ("dataset", "digital_rules"):
            self.conn.execute(
                sql.SQL(
                    """
                    SELECT setval(
                        pg_get_serial_sequence(%s, 'id'),
                        COALESCE((SELECT max(id) FROM {table_ref}), 0) + 1,
                        false
                    )
                    """
                ).format(table_ref=qualified_identifier(self.schema, table_name)),
                (f"{self.schema}.{table_name}",),
            )

    def _upsert_datasets(self, campaigns: pd.DataFrame, *, now: datetime) -> None:
        dataset_rows = []
        for dataset_name, group in campaigns.groupby("dataset_name", sort=True):
            row = group.iloc[0]
            dataset_rows.append(
                (
                    dataset_name,
                    dataset_query_for(
                        row,
                        schema=self.schema,
                        campaign_table=self.campaign_table,
                        mapping_table=self.mapping_table,
                    ),
                    now,
                    ACTOR,
                    now,
                    ACTOR,
                    None,
                    "AIML",
                )
            )
        query = sql.SQL(
            """
            INSERT INTO {table_ref} (
                name, query, created_date, created_by, last_modified_date,
                last_modified_by, wizard_config_json, created_mode
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name)
            DO UPDATE SET
                query = EXCLUDED.query,
                last_modified_date = EXCLUDED.last_modified_date,
                last_modified_by = EXCLUDED.last_modified_by,
                created_mode = EXCLUDED.created_mode
            """
        ).format(table_ref=qualified_identifier(self.schema, "dataset"))
        with self.conn.cursor() as cur:
            cur.executemany(query, dataset_rows)

    def _upsert_templates(self, campaigns: pd.DataFrame, *, now: datetime) -> None:
        template_rows = []
        for template_name, group in campaigns.groupby("template_name", sort=True):
            row = group.iloc[0]
            template_rows.append(
                (
                    template_name,
                    DIGITAL_RULE_MODE_BY_MODE.get(str(row["mode"]).upper(), str(row["mode"]).upper()),
                    "",
                    "TEST_DLT_TEMPLATE_ID",
                    placeholder_verbiage(row),
                    "",
                    language_from_template(str(template_name)),
                    True,
                    ACTOR,
                    now,
                    ACTOR,
                    now,
                    None,
                    None,
                    False,
                    "",
                    str(row["due_type"]),
                    False,
                    False,
                )
            )
        query = sql.SQL(
            """
            INSERT INTO {table_ref} (
                iteration, mode_, sender_id, dlt_temp_id, verbiage, params,
                language, active, created_by, created_date, last_modified_by,
                last_modified_date, deleted_by, deleted_date,
                collectable_amount_active, collectable_amount_column,
                campaign_identifier, media_attachment_active, multi_apac
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (iteration)
            DO UPDATE SET
                mode_ = EXCLUDED.mode_,
                verbiage = EXCLUDED.verbiage,
                language = EXCLUDED.language,
                active = EXCLUDED.active,
                last_modified_by = EXCLUDED.last_modified_by,
                last_modified_date = EXCLUDED.last_modified_date,
                campaign_identifier = EXCLUDED.campaign_identifier
            """
        ).format(table_ref=qualified_identifier(self.schema, "digital_rules"))
        with self.conn.cursor() as cur:
            cur.executemany(query, template_rows)

    def _replace_quartz_jobs(self, campaigns: pd.DataFrame) -> int:
        trigger_count = 0
        with self.conn.cursor() as cur:
            for _, row in campaigns.iterrows():
                job_name = str(row["name"])
                mode = str(row["mode"]).upper()
                job_group = JOB_GROUP_BY_MODE.get(mode, mode)
                job_class = JOB_CLASS_BY_MODE.get(mode)
                if job_class is None:
                    raise ValueError(f"Unsupported campaign mode {mode!r} for {job_name!r}.")
                self._delete_existing_triggers(cur, job_name=job_name, job_group=job_group)
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {table_ref} (
                            sched_name, job_name, job_group, description, job_class_name,
                            is_durable, is_nonconcurrent, is_update_data, requests_recovery, job_data
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sched_name, job_name, job_group)
                        DO UPDATE SET
                            description = EXCLUDED.description,
                            job_class_name = EXCLUDED.job_class_name,
                            is_durable = EXCLUDED.is_durable,
                            is_nonconcurrent = EXCLUDED.is_nonconcurrent,
                            is_update_data = EXCLUDED.is_update_data,
                            requests_recovery = EXCLUDED.requests_recovery,
                            job_data = EXCLUDED.job_data
                        """
                    ).format(table_ref=qualified_identifier(self.schema, "qrtz_job_details")),
                    (
                        SCHED_NAME,
                        job_name,
                        job_group,
                        ACTOR,
                        job_class,
                        True,
                        True,
                        False,
                        False,
                        build_mcollect_job_data(
                            dataset_name=str(row["dataset_name"]),
                            template_name=str(row["template_name"]),
                            vendor=str(row["vendor"]),
                        ),
                    ),
                )
                for trigger in build_cron_trigger_specs(row, trigger_state=self.trigger_state):
                    trigger_count += 1
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {table_ref} (
                                sched_name, trigger_name, trigger_group, job_name, job_group,
                                description, next_fire_time, prev_fire_time, priority,
                                trigger_state, trigger_type, start_time, end_time,
                                calendar_name, misfire_instr, job_data
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                        ).format(table_ref=qualified_identifier(self.schema, "qrtz_triggers")),
                        (
                            SCHED_NAME,
                            trigger.trigger_name,
                            TRIGGER_GROUP,
                            job_name,
                            job_group,
                            "",
                            trigger.next_fire_time_ms,
                            -1,
                            5,
                            trigger.trigger_state,
                            "CRON",
                            trigger.start_time_ms,
                            trigger.end_time_ms,
                            None,
                            0,
                            None,
                        ),
                    )
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {table_ref} (
                                sched_name, trigger_name, trigger_group, cron_expression, time_zone_id
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            """
                        ).format(table_ref=qualified_identifier(self.schema, "qrtz_cron_triggers")),
                        (
                            SCHED_NAME,
                            trigger.trigger_name,
                            TRIGGER_GROUP,
                            trigger.cron_expression,
                            TIME_ZONE_ID,
                        ),
                    )
        return trigger_count

    def _delete_existing_triggers(self, cur, *, job_name: str, job_group: str) -> None:
        cur.execute(
            sql.SQL(
                """
                DELETE FROM {cron_ref} cron
                USING {trigger_ref} trigger
                WHERE cron.sched_name = trigger.sched_name
                  AND cron.trigger_name = trigger.trigger_name
                  AND cron.trigger_group = trigger.trigger_group
                  AND trigger.sched_name = %s
                  AND trigger.job_name = %s
                  AND trigger.job_group = %s
                """
            ).format(
                cron_ref=qualified_identifier(self.schema, "qrtz_cron_triggers"),
                trigger_ref=qualified_identifier(self.schema, "qrtz_triggers"),
            ),
            (SCHED_NAME, job_name, job_group),
        )
        cur.execute(
            sql.SQL(
                """
                DELETE FROM {trigger_ref}
                WHERE sched_name = %s
                  AND job_name = %s
                  AND job_group = %s
                """
            ).format(trigger_ref=qualified_identifier(self.schema, "qrtz_triggers")),
            (SCHED_NAME, job_name, job_group),
        )


def main() -> None:
    args = parse_args()
    config = PostgresConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    with connect_db(config) as conn:
        campaigns = fetch_campaign_rows(
            conn,
            schema=args.schema,
            campaign_table=args.campaign_table,
            source_month=args.source_month,
            prediction_month=args.prediction_month,
            model_name=args.model,
        )
        trigger_count = sum(
            len(build_cron_trigger_specs(row, trigger_state=args.trigger_state)) for _, row in campaigns.iterrows()
        )
        summary = ExportSummary(
            campaigns=len(campaigns),
            datasets=int(campaigns["dataset_name"].nunique()) if not campaigns.empty else 0,
            templates=int(campaigns["template_name"].nunique()) if not campaigns.empty else 0,
            jobs=len(campaigns),
            triggers=trigger_count,
        )
        if not args.write:
            print("Dry run only. Add --write to insert/update MCollect tables.")
            print(summary)
            return
        publisher = DirectDatabaseMcollectPublisher(
            conn,
            schema=args.schema,
            campaign_table=args.campaign_table,
            mapping_table=args.mapping_table,
            trigger_state=args.trigger_state,
        )
        written = publisher.publish(campaigns)
        conn.commit()
        print(written)


if __name__ == "__main__":
    main()
