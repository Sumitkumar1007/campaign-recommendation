from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from entity_keys import annotate_entity_key, entity_key_column_name
from env_utils import load_dotenv
from pipeline_common import candidate_hours
from project_paths import COMMUNICATION_DATA_DIR, REPO_ROOT, TRAINING_DATA_DIR, ensure_parent_dir
from append_payment_flags import load_paid_keys


TIME_BUCKETS = {
    9: "9AM",
    10: "10AM",
    11: "11AM",
    12: "12PM",
    13: "1PM",
    14: "2PM",
    15: "3PM",
    16: "4PM",
    17: "5PM",
    18: "6PM",
}

COMM_TYPE_MAP = {
    "SMS": "SMS",
    "WHATSAPP": "WH",
    "VOICE": "VOICE",
    "VOICE_BOT": "VOICE_BOT",
}

SUCCESS_STATUS_MAP = {
    "SMS": {"DELIVERED", "CLICKED", "SENT"},
    "WH": {"DELIVERED", "READ", "CLICKED", "SENT"},
    "VOICE": {"CONNECTED", "CALL_CONNECTED"},
    "VOICE_BOT": {"CONNECTED", "CALL_CONNECTED"},
}

DEFAULT_SUCCESS_SCORES = {
    "SMS": {"DELIVERED": 0.4, "READ": 0.8, "CLICKED": 1.0},
    "WH": {"DELIVERED": 0.4, "READ": 0.8, "CLICKED": 1.0},
    "VOICE": {"CONNECTED": 1.0, "CALL_CONNECTED": 1.0},
    "VOICE_BOT": {"CONNECTED": 1.0, "CALL_CONNECTED": 1.0},
}

DEFAULT_VOICE_BOT_DISPOSITION_SCORES: dict[str, float] = {}
WEIGHTS_CONFIG_FILE = REPO_ROOT / "config" / "weights_config.json"

TARGET_SAMPLE_WEIGHT_COLUMN = "TARGET_SAMPLE_WEIGHT"

USECOLS = [
    "apac_card_number",
    "party_id",
    "comm_status",
    "communication_type",
    "disposition",
    "verbiage_language",
    "vertical",
    "risk",
    "emi_date",
    "date",
    "created_date",
]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_config_key(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").upper().split())


def normalize_channel_key(value: object) -> str:
    raw = str(value or "").strip().upper()
    return COMM_TYPE_MAP.get(raw, raw)


def load_weight_config() -> tuple[bool, dict[str, set[str]], dict[str, dict[str, float]], dict[str, float], float, Path, float, float]:
    enabled = env_flag("STRATEGY_USE_STATUS_WEIGHTS", default=False)
    config_file = Path(os.getenv("STRATEGY_WEIGHTS_CONFIG_FILE", str(WEIGHTS_CONFIG_FILE))).expanduser()
    success_status_map = {channel: set(statuses) for channel, statuses in SUCCESS_STATUS_MAP.items()}
    score_map = DEFAULT_SUCCESS_SCORES
    disposition_scores = DEFAULT_VOICE_BOT_DISPOSITION_SCORES

    if config_file.exists():
        parsed = json.loads(config_file.read_text())
        raw_statuses = parsed.get("success_statuses", {})
        if raw_statuses:
            success_status_map = {
                normalize_channel_key(channel): {str(status).upper().strip() for status in statuses}
                for channel, statuses in raw_statuses.items()
            }
        raw_scores = parsed.get("strategy_success_scores", {})
        if raw_scores:
            score_map = {
                normalize_channel_key(channel): {
                    str(status).upper().strip(): float(score)
                    for status, score in statuses.items()
                }
                for channel, statuses in raw_scores.items()
            }
        raw_disposition_scores = parsed.get("voice_bot_disposition_success_scores", {})
        if raw_disposition_scores:
            disposition_scores = {
                normalize_config_key(disposition): float(score)
                for disposition, score in raw_disposition_scores.items()
            }

    positive_boost = float(os.getenv("STRATEGY_SAMPLE_WEIGHT_POSITIVE_BOOST", "1.0"))
    payment_multipliers = parsed.get("payment_multipliers", {}) if config_file.exists() else {}
    default_link_mult = float(payment_multipliers.get("link_payment_multiplier", 5.0))
    default_ext_mult = float(payment_multipliers.get("external_payment_multiplier", 2.5))

    link_payment_multiplier = float(
        os.getenv("STRATEGY_LINK_PAYMENT_MULTIPLIER")
        or os.getenv("STRATEGY_PAYMENT_MULTIPLIER")
        or default_link_mult
    )
    external_payment_multiplier = float(
        os.getenv("STRATEGY_EXTERNAL_PAYMENT_MULTIPLIER")
        or default_ext_mult
    )
    return (
        enabled,
        success_status_map,
        score_map,
        disposition_scores,
        positive_boost,
        config_file,
        link_payment_multiplier,
        external_payment_multiplier,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a wide strategy dataset from MFL communication CSV files "
            "for D-5 to D+20, excluding D."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(COMMUNICATION_DATA_DIR),
        help="Directory containing monthly communication CSV files.",
    )
    # parser.add_argument(
    #     "--output-file",
    #     default=str(TRAINING_DATA_DIR / "strategy_training_dataset.csv"),
    #     help="Path to the final wide CSV output.",
    # )
    parser.add_argument(
        "--output-file",
        default=str(TRAINING_DATA_DIR / "strategy_training_dataset_train.csv"),
        help="Path to the final wide CSV output.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for streaming reads.",
    )
    parser.add_argument(
        "--sample-cards",
        type=int,
        default=0,
        help="If set, generate output only for the first N APAC_CARD_NUMBER values found.",
    )
    parser.add_argument(
        "--exclude-months",
        nargs="*",
        default=[],
        help=(
            "Optional month tokens to exclude based on filename, for example "
            "MAR2026 FEB2026."
        ),
    )
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=[],
        help="Optional explicit CSV files. When set, --input-dir and --exclude-months are ignored.",
    )
    parser.add_argument(
        "--month-source",
        choices=["emi_date", "created_date"],
        default="emi_date",
        help="Date field used to assign the MONTH feature label.",
    )
    parser.add_argument(
        "--payment-files",
        nargs="*",
        default=[],
        help="Optional payment CSV files to apply 'True Success' multiplier to SMS and WH channels.",
    )
    parser.add_argument(
        "--config-file",
        default=str(REPO_ROOT / "config" / "default_config.json"),
        help="JSON config file containing send_hour_window.",
    )
    return parser.parse_args()


def day_label(offset: int) -> str:
    if offset < 0:
        return f"D{offset}"
    return f"D+{offset}"


def normalize_language(series: pd.Series) -> pd.Series:
    return (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .replace({"": "UNKNOWN"})
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
    )


def normalize_status(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def normalize_comm_type(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .map(COMM_TYPE_MAP)
    )


def normalize_risk(series: pd.Series) -> pd.Series:
    return (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .replace({"": "UNKNOWN"})
        .str.upper()
    )


def normalize_vertical(series: pd.Series) -> pd.Series:
    return (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .replace({"": "UNKNOWN"})
        .str.upper()
    )


def load_send_hour_window(config_file: str | Path) -> dict[str, int]:
    with Path(config_file).open("r", encoding="utf-8") as handle:
        raw_window = json.load(handle).get("send_hour_window", {})
    return {
        "start_hour": int(raw_window.get("start_hour", 9)),
        "end_hour": int(raw_window.get("end_hour", 18)),
        "step_hours": int(raw_window.get("step_hours", 1)),
    }


def bucket_send_hour(hour: object, send_hour_window: dict[str, int] | None = None) -> int | pd.NA:
    if pd.isna(hour):
        return pd.NA
    window = send_hour_window or {"start_hour": 9, "end_hour": 18, "step_hours": 1}
    start_hour = int(window["start_hour"])
    end_hour = int(window["end_hour"])
    hour_int = int(hour)
    if hour_int < start_hour:
        return start_hour
    if hour_int >= end_hour:
        return end_hour
    allowed_hours = candidate_hours(window)
    return max(allowed_hour for allowed_hour in allowed_hours if allowed_hour <= hour_int)


def process_chunk(
    chunk: pd.DataFrame,
    sample_cards: set[str] | None = None,
    month_source: str = "emi_date",
    send_hour_window: dict[str, int] | None = None,
    weighting_enabled: bool = False,
    success_statuses: dict[str, set[str]] | None = None,
    success_scores: dict[str, dict[str, float]] | None = None,
    voice_bot_disposition_scores: dict[str, float] | None = None,
    paid_keys: set[tuple[str, str]] | None = None,
    payment_multiplier: float = 5.0,
    link_ref_numbers: set[str] | None = None,
    external_payment_dates: set[tuple[str, str]] | None = None,
    link_payment_multiplier: float = 5.0,
    external_payment_multiplier: float = 2.5,
    link_pay_counts: dict[tuple[str, str], int] | None = None,
    ext_pay_counts: dict[tuple[str, str], int] | None = None,
    tot_pay_counts: dict[tuple[str, str], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = chunk.copy()
    if "risk" not in df.columns:
        raise ValueError("Missing required risk column in communication data.")
    df = annotate_entity_key(df, loan_column="apac_card_number", party_column="party_id", output_column=entity_key_column_name())
    df["APAC_CARD_NUMBER"] = df["apac_card_number"].astype(str).str.strip()
    key_col = entity_key_column_name()
    if sample_cards is not None:
        df = df[df["APAC_CARD_NUMBER"].isin(sample_cards)].copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df["COMM_TYPE"] = normalize_comm_type(df["communication_type"])
    df["STATUS"] = normalize_status(df["comm_status"])
    df["DISPOSITION"] = df["disposition"].map(normalize_config_key) if "disposition" in df.columns else ""
    df["LANGUAGE"] = normalize_language(df["verbiage_language"])
    df["VERTICAL"] = normalize_vertical(df["vertical"]) if "vertical" in df.columns else "UNKNOWN"
    df["RISK"] = normalize_risk(df["risk"])
    df["emi_date"] = pd.to_datetime(df["emi_date"], errors="coerce").dt.normalize()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    created_ts = pd.to_datetime(df["created_date"], errors="coerce")
    df["hr"] = created_ts.dt.hour.astype("Int64").map(
        lambda hour: bucket_send_hour(hour, send_hour_window)
    ).astype("Int64")

    df = df[
        df[key_col].ne("")
        & df["RISK"].ne("UNKNOWN")
        & df["COMM_TYPE"].notna()
        & df["emi_date"].notna()
        & df["date"].notna()
        & created_ts.notna()
    ].copy()

    df["offset"] = (df["date"] - df["emi_date"]).dt.days
    df = df[df["offset"].between(-5, 20) & df["offset"].ne(0)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    month_dates = df["emi_date"] if month_source == "emi_date" else created_ts.dt.normalize()
    df["MONTH"] = month_dates.dt.strftime("%b-%Y").str.upper()
    df["DAY"] = df["offset"].map(day_label)
    status_lookup = success_statuses or SUCCESS_STATUS_MAP
    disposition_lookup = voice_bot_disposition_scores or {}

    def is_success(row: pd.Series) -> bool:
        connected_success = row["STATUS"] in status_lookup.get(row["COMM_TYPE"], set())
        if weighting_enabled and row["COMM_TYPE"] == "VOICE_BOT" and disposition_lookup:
            return connected_success and row["DISPOSITION"] in disposition_lookup
        return connected_success

    df["IS_SUCCESS"] = df.apply(is_success, axis=1)
    score_lookup = {
        normalize_channel_key(channel): {
            str(status).upper().strip(): float(score)
            for status, score in statuses.items()
        }
        for channel, statuses in (success_scores or {}).items()
    }

    def success_score(row: pd.Series) -> float:
        if not row["IS_SUCCESS"]:
            return 0.0
            
        base_score = 1.0
        if weighting_enabled:
            if row["COMM_TYPE"] == "VOICE_BOT" and disposition_lookup:
                base_score = float(disposition_lookup.get(row["DISPOSITION"], 0.0))
            else:
                base_score = float(score_lookup.get(row["COMM_TYPE"], {}).get(row["STATUS"], 1.0))

        if paid_keys and row["COMM_TYPE"] in {"SMS", "WH", "WHATSAPP", "VOICE", "VOICE_BOT"}:
            entity = str(row[key_col]).strip()
            month = str(row["MONTH"]).strip().upper()
            pay_unique_id = str(row.get("payment_unique_id", "") or "").strip()
            comm_date_val = row.get("date")
            comm_date_str = comm_date_val.strftime("%Y-%m-%d") if pd.notna(comm_date_val) else ""

            # Priority 1: Link Payment Match (c.payment_unique_id == p.reference_number)
            if pay_unique_id and link_ref_numbers and pay_unique_id in link_ref_numbers:
                return base_score * link_payment_multiplier

            # Priority 2: External Payment Match on communication date
            if comm_date_str and external_payment_dates and (entity, comm_date_str) in external_payment_dates:
                return base_score * external_payment_multiplier

            # Fallback: Monthly account payment match
            if (entity, month) in paid_keys:
                return base_score * external_payment_multiplier

        return base_score

    def get_payment_flags(row: pd.Series) -> tuple[int, int, int]:
        if not paid_keys or row["COMM_TYPE"] not in {"SMS", "WH", "WHATSAPP", "VOICE", "VOICE_BOT"}:
            return 0, 0, 0
        entity = str(row[key_col]).strip()
        month = str(row["MONTH"]).strip().upper()
        pay_unique_id = str(row.get("payment_unique_id", "") or "").strip()
        comm_date_val = row.get("date")
        comm_date_str = comm_date_val.strftime("%Y-%m-%d") if pd.notna(comm_date_val) else ""

        is_link = 1 if (pay_unique_id and link_ref_numbers and pay_unique_id in link_ref_numbers) else 0
        
        is_ext = 0
        # Only assign external payment flag if there was actually an external payment for this account
        if not is_link and ext_pay_counts and ext_pay_counts.get((entity, month), 0) > 0:
            if comm_date_str and external_payment_dates and (entity, comm_date_str) in external_payment_dates:
                is_ext = 1
            else:
                # Fallback: if there was an external payment this month, but we couldn't match the exact date,
                # we don't blindly flag EVERY communication as an external payment for the daily counts.
                # Since we want accurate daily counts, we only flag it if the date matches.
                # (The monthly total is handled safely downstream in build_monthly_feature_dataset.py)
                pass

        is_any = 1 if (is_link or is_ext) else 0
        return is_link, is_ext, is_any

    df["SUCCESS_SCORE"] = df.apply(success_score, axis=1)
    p_flags = df.apply(get_payment_flags, axis=1)
    df["IS_LINK_PAYMENT"] = [p[0] for p in p_flags]
    df["IS_EXT_PAYMENT"] = [p[1] for p in p_flags]
    df["IS_ANY_PAYMENT"] = [p[2] for p in p_flags]

    totals = (
        df.groupby([key_col, "MONTH", "DAY", "COMM_TYPE"], sort=False)
        .size()
        .reset_index(name="count")
    )
    totals["feature"] = totals["COMM_TYPE"] + "_TOTAL_INTENSITY"

    # Payment feature aggregations per (key_col, MONTH, DAY)
    link_pay_feats = (
        df.groupby([key_col, "MONTH", "DAY"], sort=False)["IS_LINK_PAYMENT"]
        .sum()
        .reset_index(name="count")
    )
    link_pay_feats["feature"] = "LINK_PAYMENT_COUNT"

    ext_pay_feats = (
        df.groupby([key_col, "MONTH", "DAY"], sort=False)["IS_EXT_PAYMENT"]
        .sum()
        .reset_index(name="count")
    )
    ext_pay_feats["feature"] = "EXT_PAYMENT_COUNT"

    tot_pay_feats = (
        df.groupby([key_col, "MONTH", "DAY"], sort=False)["IS_ANY_PAYMENT"]
        .sum()
        .reset_index(name="count")
    )
    tot_pay_feats["feature"] = "TOTAL_PAYMENT_COUNT"

    if link_pay_counts or ext_pay_counts or tot_pay_counts:
        for idx, r in link_pay_feats.iterrows():
            entity = str(r[key_col]).strip()
            month = str(r["MONTH"]).strip().upper()
            k = (entity, month)
            link_pay_feats.at[idx, "count"] = float((link_pay_counts or {}).get(k, 0))
        for idx, r in ext_pay_feats.iterrows():
            entity = str(r[key_col]).strip()
            month = str(r["MONTH"]).strip().upper()
            k = (entity, month)
            ext_pay_feats.at[idx, "count"] = float((ext_pay_counts or {}).get(k, 0))
        for idx, r in tot_pay_feats.iterrows():
            entity = str(r[key_col]).strip()
            month = str(r["MONTH"]).strip().upper()
            k = (entity, month)
            tot_pay_feats.at[idx, "count"] = float((tot_pay_counts or {}).get(k, 0))

    failed = (
        df.loc[~df["IS_SUCCESS"]]
        .groupby(
            [key_col, "MONTH", "DAY", "COMM_TYPE", "LANGUAGE"],
            sort=False,
        )
        .size()
        .reset_index(name="count")
    )
    if not failed.empty:
        failed["feature"] = (
            failed["COMM_TYPE"] + "_FAILED_" + failed["LANGUAGE"]
        )

    success = df.loc[df["IS_SUCCESS"] & df["hr"].isin(TIME_BUCKETS)].copy()
    if not success.empty:
        success["TIME_LABEL"] = success["hr"].map(TIME_BUCKETS)
        success["feature"] = (
            success["COMM_TYPE"]
            + "_SUCCESS_"
            + success["TIME_LABEL"]
            + "_"
            + success["LANGUAGE"]
        )
        success_features = (
            success.groupby(
                [key_col, "MONTH", "DAY", "feature"],
                sort=False,
            )["SUCCESS_SCORE"]
            .sum()
            .reset_index(name="count")
        )
        success["strategy"] = (
            success["COMM_TYPE"]
            + "-"
            + success["TIME_LABEL"]
            + "-"
            + success["LANGUAGE"]
        )
        strategy_counts = (
            success.groupby(
                [key_col, "MONTH", "DAY", "strategy"],
                sort=False,
            )["SUCCESS_SCORE"]
            .sum()
            .reset_index(name="count")
            .rename(columns={"strategy": "feature"})
        )
    else:
        success_features = pd.DataFrame()
        strategy_counts = pd.DataFrame()

    feature_frames = [
        totals[[key_col, "MONTH", "DAY", "feature", "count"]],
        link_pay_feats[[key_col, "MONTH", "DAY", "feature", "count"]],
        ext_pay_feats[[key_col, "MONTH", "DAY", "feature", "count"]],
        tot_pay_feats[[key_col, "MONTH", "DAY", "feature", "count"]],
    ]
    if not failed.empty:
        feature_frames.append(
            failed[[key_col, "MONTH", "DAY", "feature", "count"]]
        )
    if not success_features.empty:
        feature_frames.append(
            success_features[[key_col, "MONTH", "DAY", "feature", "count"]]
        )

    feature_counts = pd.concat(feature_frames, ignore_index=True)
    risk_counts = (
        df.groupby([key_col, "MONTH", "DAY"], sort=False)[["RISK", "VERTICAL"]]
        .last()
        .reset_index()
    )
    return feature_counts, strategy_counts, risk_counts


def summarize_chunk_audit(
    chunk: pd.DataFrame,
    sample_cards: set[str] | None = None,
) -> dict[str, object]:
    df = chunk.copy()
    df = annotate_entity_key(df, loan_column="apac_card_number", party_column="party_id", output_column=entity_key_column_name())
    df["APAC_CARD_NUMBER"] = df["apac_card_number"].fillna("").astype(str).str.strip()
    key_col = entity_key_column_name()
    sample_mask = df["APAC_CARD_NUMBER"].isin(sample_cards) if sample_cards is not None else pd.Series(True, index=df.index)
    scoped = df.loc[sample_mask].copy()

    raw_comm_type = scoped["communication_type"].fillna("").astype(str).str.strip().str.upper()
    normalized_comm_type = raw_comm_type.map(COMM_TYPE_MAP)
    risk = normalize_risk(scoped["risk"]) if "risk" in scoped.columns else pd.Series("UNKNOWN", index=scoped.index)
    emi_date = pd.to_datetime(scoped["emi_date"], errors="coerce").dt.normalize()
    event_date = pd.to_datetime(scoped["date"], errors="coerce").dt.normalize()
    created_ts = pd.to_datetime(scoped["created_date"], errors="coerce")

    required_mask = (
        scoped[key_col].ne("")
        & risk.ne("UNKNOWN")
        & normalized_comm_type.notna()
        & emi_date.notna()
        & event_date.notna()
        & created_ts.notna()
    )
    offsets = (event_date - emi_date).dt.days
    within_window = offsets.between(-5, 5, inclusive="both")
    day_zero = offsets.eq(0)
    kept_mask = required_mask & within_window & ~day_zero

    unsupported_counts = raw_comm_type[normalized_comm_type.isna()].replace({"": "EMPTY"}).value_counts().to_dict()
    normalized_counts = normalized_comm_type.fillna("UNSUPPORTED").value_counts().to_dict()
    return {
        "total_rows": int(len(df)),
        "sample_filtered_rows": int((~sample_mask).sum()) if sample_cards is not None else 0,
        "blank_apac_rows": int(scoped["APAC_CARD_NUMBER"].eq("").sum()),
        "unknown_risk_rows": int(risk.eq("UNKNOWN").sum()),
        "unsupported_comm_type_rows": int(normalized_comm_type.isna().sum()),
        "invalid_emi_date_rows": int(emi_date.isna().sum()),
        "invalid_event_date_rows": int(event_date.isna().sum()),
        "invalid_created_date_rows": int(created_ts.isna().sum()),
        "outside_day_window_rows": int((required_mask & ~within_window).sum()),
        "day_zero_rows": int((required_mask & day_zero).sum()),
        "kept_model_rows": int(kept_mask.sum()),
        "unsupported_comm_type_counts": {str(key): int(value) for key, value in unsupported_counts.items()},
        "normalized_comm_type_counts": {str(key): int(value) for key, value in normalized_counts.items()},
    }


def merge_audit_counts(target: dict[str, object], current: dict[str, object]) -> dict[str, object]:
    for key, value in current.items():
        if isinstance(value, dict):
            existing = target.setdefault(key, {})
            for sub_key, sub_value in value.items():
                existing[sub_key] = int(existing.get(sub_key, 0)) + int(sub_value)
        else:
            target[key] = int(target.get(key, 0)) + int(value)
    return target


def log_dataset_audit(audit_counts: dict[str, object], *, csv_files: list[Path], output_file: Path) -> None:
    payload = {
        "input_files": [path.name for path in csv_files],
        "output_file": str(output_file),
        **audit_counts,
    }
    print("Strategy dataset audit summary | " + json.dumps(payload, sort_keys=True))
    normalized_counts = payload.get("normalized_comm_type_counts", {})
    if int(normalized_counts.get("VOICE_BOT", 0)) == 0:
        print("Strategy dataset warning | VOICE_BOT channel was not present in the training input data.")


def collect_sample_cards(
    csv_files: list[Path],
    chunksize: int,
    sample_size: int,
) -> set[str]:
    cards: list[str] = []
    seen: set[str] = set()

    for csv_file in csv_files:
        for chunk in pd.read_csv(csv_file, usecols=["apac_card_number"], chunksize=chunksize):
            values = (
                chunk["apac_card_number"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            for card in values:
                if not card or card in seen:
                    continue
                seen.add(card)
                cards.append(card)
                if len(cards) >= sample_size:
                    return set(cards)
    return set(cards)


def build_dataset(
    input_dir: Path,
    output_file: Path,
    chunksize: int,
    sample_size: int = 0,
    exclude_months: list[str] | None = None,
    input_files: list[Path] | None = None,
    month_source: str = "emi_date",
    send_hour_window: dict[str, int] | None = None,
    payment_files: list[Path] | None = None,
) -> None:
    (
        weighting_enabled,
        success_statuses,
        success_scores,
        voice_bot_disposition_scores,
        positive_boost,
        weights_config_file,
        link_payment_multiplier,
        external_payment_multiplier,
    ) = load_weight_config()
    output_file = ensure_parent_dir(output_file)
    exclude_tokens = {token.upper() for token in (exclude_months or [])}
    if input_files:
        csv_files = [Path(csv_file) for csv_file in input_files]
    else:
        csv_files = sorted(
            csv_file
            for csv_file in input_dir.glob("*.csv")
            if not any(token in csv_file.name.upper() for token in exclude_tokens)
        )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    feature_parts: list[pd.DataFrame] = []
    strategy_parts: list[pd.DataFrame] = []
    risk_parts: list[pd.DataFrame] = []
    audit_counts: dict[str, object] = {}
    
    paid_keys: set[tuple[str, str]] | None = None
    link_ref_numbers: set[str] | None = None
    external_payment_dates: set[tuple[str, str]] | None = None
    link_pay_counts: dict[tuple[str, str], int] | None = None
    ext_pay_counts: dict[tuple[str, str], int] | None = None
    tot_pay_counts: dict[tuple[str, str], int] | None = None
    if payment_files:
        print(f"Loading payment data (Link Multiplier={link_payment_multiplier}x, External Multiplier={external_payment_multiplier}x)...")
        paid_keys, _, link_ref_numbers, external_payment_dates, link_pay_counts, ext_pay_counts, tot_pay_counts = load_paid_keys(payment_files)
        print(f"Loaded {len(paid_keys)} unique payment events ({len(link_ref_numbers)} link refs, {len(external_payment_dates)} date refs).")

    sample_cards = (
        collect_sample_cards(csv_files, chunksize, sample_size)
        if sample_size > 0
        else None
    )

    for csv_file in csv_files:
        print(f"Processing {csv_file.name} ...")
        header_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()
        available_usecols = [column for column in USECOLS if column in header_columns]
        for chunk in pd.read_csv(csv_file, usecols=available_usecols, chunksize=chunksize):
            for column in USECOLS:
                if column not in chunk.columns:
                    chunk[column] = ""
            merge_audit_counts(audit_counts, summarize_chunk_audit(chunk, sample_cards=sample_cards))
            feature_counts, strategy_counts, risk_counts = process_chunk(
                chunk,
                sample_cards=sample_cards,
                month_source=month_source,
                send_hour_window=send_hour_window,
                weighting_enabled=weighting_enabled,
                success_statuses=success_statuses,
                success_scores=success_scores,
                voice_bot_disposition_scores=voice_bot_disposition_scores,
                paid_keys=paid_keys,
                link_ref_numbers=link_ref_numbers,
                external_payment_dates=external_payment_dates,
                link_payment_multiplier=link_payment_multiplier,
                external_payment_multiplier=external_payment_multiplier,
                link_pay_counts=link_pay_counts,
                ext_pay_counts=ext_pay_counts,
                tot_pay_counts=tot_pay_counts,
            )
            if not feature_counts.empty:
                feature_parts.append(feature_counts)
            if not strategy_counts.empty:
                strategy_parts.append(strategy_counts)
            if not risk_counts.empty:
                risk_parts.append(risk_counts)

    if not feature_parts:
        raise ValueError("No rows matched the D-5 to D+20 window.")
    if not risk_parts:
        raise ValueError("No risk rows found in communication data.")

    key_col = entity_key_column_name()
    features = (
        pd.concat(feature_parts, ignore_index=True)
        .groupby([key_col, "MONTH", "DAY", "feature"], as_index=False)["count"]
        .sum()
        .rename(
            columns={
                key_col: "entity_key",
                "MONTH": "month",
                "DAY": "day",
            }
        )
    )

    if strategy_parts:
        strategies = (
            pd.concat(strategy_parts, ignore_index=True)
            .groupby([key_col, "MONTH", "DAY", "feature"], as_index=False)["count"]
            .sum()
            .rename(
                columns={
                    key_col: "entity_key",
                    "MONTH": "month",
                    "DAY": "day",
                }
            )
        )
    else:
        strategies = pd.DataFrame(
            columns=["entity_key", "month", "day", "feature", "count"]
        )

    wide = (
        features.pivot_table(
            index=["entity_key", "month", "day"],
            columns="feature",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(
            columns={
                "entity_key": "ENTITY_KEY",
                "month": "MONTH",
                "day": "DAY",
            }
        )
    )
    risk_df = (
        pd.concat(risk_parts, ignore_index=True)
        .groupby([key_col, "MONTH", "DAY"], as_index=False)[["RISK", "VERTICAL"]]
        .last()
        .rename(columns={key_col: "ENTITY_KEY"})
    )
    wide["ENTITY_KEY"] = wide["ENTITY_KEY"].astype(str).str.strip()
    risk_df["ENTITY_KEY"] = risk_df["ENTITY_KEY"].astype(str).str.strip()
    wide = wide.merge(risk_df, on=["ENTITY_KEY", "MONTH", "DAY"], how="left")

    if not strategies.empty:
        strategies = strategies.sort_values(
            ["entity_key", "month", "day", "count", "feature"],
            ascending=[True, True, True, False, True],
        )
        predicted = strategies.drop_duplicates(
            subset=["entity_key", "month", "day"],
            keep="first",
        ).rename(
            columns={
                "entity_key": "ENTITY_KEY",
                "month": "MONTH",
                "day": "DAY",
                "feature": "PREDICTED_STRATEGY",
            }
        )
        predicted["ENTITY_KEY"] = predicted["ENTITY_KEY"].astype(str).str.strip()
        if weighting_enabled:
            predicted[TARGET_SAMPLE_WEIGHT_COLUMN] = 1.0 + (
                predicted["count"].astype(float) * positive_boost
            )
        else:
            predicted[TARGET_SAMPLE_WEIGHT_COLUMN] = 1.0
        wide = wide.merge(
            predicted[[
                "ENTITY_KEY",
                "MONTH",
                "DAY",
                "PREDICTED_STRATEGY",
                TARGET_SAMPLE_WEIGHT_COLUMN,
            ]],
            on=["ENTITY_KEY", "MONTH", "DAY"],
            how="left",
        )
    else:
        wide["PREDICTED_STRATEGY"] = pd.NA
        wide[TARGET_SAMPLE_WEIGHT_COLUMN] = 1.0

    wide[TARGET_SAMPLE_WEIGHT_COLUMN] = pd.to_numeric(
        wide[TARGET_SAMPLE_WEIGHT_COLUMN],
        errors="coerce",
    ).fillna(1.0)

    feature_columns = sorted(
        column
        for column in wide.columns
        if column not in {"ENTITY_KEY", "MONTH", "DAY", "RISK", "VERTICAL", "PREDICTED_STRATEGY", TARGET_SAMPLE_WEIGHT_COLUMN}
    )
    wide = wide[
        [
            "ENTITY_KEY",
            "MONTH",
            "DAY",
            "RISK",
            "VERTICAL",
            *feature_columns,
            TARGET_SAMPLE_WEIGHT_COLUMN,
            "PREDICTED_STRATEGY",
        ]
    ]
    wide.to_csv(output_file, index=False)
    print(f"Saved {len(wide):,} rows to {output_file}")
    log_dataset_audit(audit_counts, csv_files=csv_files, output_file=output_file)
    print(
        "Strategy dataset weighting | "
        + json.dumps(
            {
                "enabled": weighting_enabled,
                "positive_boost": positive_boost,
                "weights_config_file": str(weights_config_file),
                "weighted_status_channels": sorted(success_scores),
                "voice_bot_disposition_weight_count": len(voice_bot_disposition_scores),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    send_hour_window = load_send_hour_window(args.config_file)
    build_dataset(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output_file),
        chunksize=args.chunksize,
        sample_size=args.sample_cards,
        exclude_months=args.exclude_months,
        input_files=[Path(path) for path in args.input_files],
        month_source=args.month_source,
        send_hour_window=send_hour_window,
        payment_files=[Path(path) for path in args.payment_files],
    )


if __name__ == "__main__":
    main()
