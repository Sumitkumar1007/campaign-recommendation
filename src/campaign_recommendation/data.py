from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .labeling import apply_success_label


HISTORY_FEATURE_COLUMNS = [
    "prev_month_total_comms",
    "prev_month_success_count",
    "prev_month_payment_count",
    "prev_month_sms_count",
    "prev_month_whatsapp_count",
    "prev_month_voice_count",
    "prev_month_sms_success_count",
    "prev_month_whatsapp_success_count",
    "prev_month_voice_success_count",
    "prev_month_success_rate",
    "prev_month_payment_rate",
]

DAY_OFFSET_SUFFIX = {
    -5: "m5",
    -4: "m4",
    -3: "m3",
    -2: "m2",
    -1: "m1",
    1: "p1",
    2: "p2",
    3: "p3",
    4: "p4",
    5: "p5",
}

DAY_AVAILABILITY_COLUMNS = [
    f"prev_month_day_{suffix}_available" for suffix in DAY_OFFSET_SUFFIX.values()
]


def _prepare_chunk(
    chunk: pd.DataFrame,
    allowed_day_offsets: set[int],
    success_statuses: dict[str, list[str]],
    success_scores: dict[str, dict[str, float]],
    positive_boost: float,
    emi_cycles: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = chunk.copy()
    prepared["created_date"] = pd.to_datetime(prepared["created_date"], errors="coerce")
    prepared["emi_date"] = pd.to_datetime(prepared["emi_date"], format="%d/%m/%Y", errors="coerce")
    prepared = prepared.dropna(
        subset=[
            "apac_card_number",
            "created_date",
            "emi_date",
            "communication_type",
            "comm_status",
            "risk",
            "collectable_amount",
        ]
    )
    emi_cycle_set = {int(day) for day in emi_cycles}
    prepared = prepared[prepared["emi_date"].dt.day.isin(emi_cycle_set)].copy()
    if prepared.empty:
        return prepared, prepared

    prepared["day_offset"] = (prepared["created_date"].dt.normalize() - prepared["emi_date"]).dt.days
    prepared["send_hour"] = prepared["created_date"].dt.hour
    prepared["emi_month"] = prepared["emi_date"].dt.to_period("M").astype(str)
    prepared["created_month"] = prepared["created_date"].dt.to_period("M").astype(str)
    prepared["has_payment"] = prepared["payment_unique_id"].notna().astype(int)
    prepared = apply_success_label(
        prepared,
        success_statuses=success_statuses,
        success_scores=success_scores,
        positive_boost=positive_boost,
    )

    history = prepared.copy()
    current = prepared[prepared["day_offset"].isin(allowed_day_offsets)].copy()
    return history, current


def _aggregate_history(history_frame: pd.DataFrame) -> pd.DataFrame:
    if history_frame.empty:
        return pd.DataFrame(
            columns=["apac_card_number", "history_month", *HISTORY_FEATURE_COLUMNS, *DAY_AVAILABILITY_COLUMNS]
        )

    history = history_frame.copy()
    history["is_sms"] = (history["communication_type"] == "SMS").astype(int)
    history["is_whatsapp"] = (history["communication_type"] == "WHATSAPP").astype(int)
    history["is_voice"] = (history["communication_type"] == "VOICE").astype(int)
    history["sms_success"] = history["is_sms"] * history["target_success"]
    history["whatsapp_success"] = history["is_whatsapp"] * history["target_success"]
    history["voice_success"] = history["is_voice"] * history["target_success"]

    aggregated = (
        history.groupby(["apac_card_number", "created_month"], as_index=False)
        .agg(
            prev_month_total_comms=("communication_type", "size"),
            prev_month_success_count=("target_success", "sum"),
            prev_month_payment_count=("has_payment", "sum"),
            prev_month_sms_count=("is_sms", "sum"),
            prev_month_whatsapp_count=("is_whatsapp", "sum"),
            prev_month_voice_count=("is_voice", "sum"),
            prev_month_sms_success_count=("sms_success", "sum"),
            prev_month_whatsapp_success_count=("whatsapp_success", "sum"),
            prev_month_voice_success_count=("voice_success", "sum"),
        )
        .rename(columns={"created_month": "history_month"})
    )

    total = aggregated["prev_month_total_comms"].replace(0, np.nan)
    aggregated["prev_month_success_rate"] = (
        aggregated["prev_month_success_count"] / total
    ).fillna(0.0)
    aggregated["prev_month_payment_rate"] = (
        aggregated["prev_month_payment_count"] / total
    ).fillna(0.0)

    availability = (
        history[history["day_offset"].isin(DAY_OFFSET_SUFFIX)]
        .assign(day_suffix=lambda frame: frame["day_offset"].map(DAY_OFFSET_SUFFIX))
        .drop_duplicates(["apac_card_number", "created_month", "day_suffix"])
    )
    if not availability.empty:
        availability["available"] = 1
        day_matrix = (
            availability.pivot_table(
                index=["apac_card_number", "created_month"],
                columns="day_suffix",
                values="available",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
            .rename(columns={"created_month": "history_month"})
        )
        day_matrix.columns = [
            column
            if column in {"apac_card_number", "history_month"}
            else f"prev_month_day_{column}_available"
            for column in day_matrix.columns
        ]
        aggregated = aggregated.merge(day_matrix, on=["apac_card_number", "history_month"], how="left")

    for column in DAY_AVAILABILITY_COLUMNS:
        if column not in aggregated.columns:
            aggregated[column] = 0
        aggregated[column] = pd.to_numeric(aggregated[column], errors="coerce").fillna(0).astype(int)
    return aggregated


def get_day_availability_column(day_offset: int) -> str:
    suffix = DAY_OFFSET_SUFFIX[day_offset]
    return f"prev_month_day_{suffix}_available"


def _assign_month_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    months = sorted(df["emi_month"].dropna().astype(str).unique().tolist())
    if len(months) < 3:
        raise ValueError(
            f"Need at least 3 EMI months for train/validation/test split, but found only {months}."
        )

    train_months = months[:-2]
    validation_month = months[-2]
    test_month = months[-1]

    split_map = {month: "train" for month in train_months}
    split_map[validation_month] = "validation"
    split_map[test_month] = "test"

    assigned = df.copy()
    assigned["dataset_split"] = assigned["emi_month"].map(split_map)
    meta = {
        "train_months": train_months,
        "validation_months": [validation_month],
        "test_months": [test_month],
    }
    return assigned, meta


def _sample_rows_by_month(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df

    months = sorted(df["emi_month"].unique().tolist())
    per_month_limit = max(1, max_rows // max(1, len(months)))
    sampled = (
        df.sort_values(["emi_month", "emi_date", "created_date"])
        .groupby("emi_month", group_keys=False)
        .head(per_month_limit)
    )
    return sampled.reset_index(drop=True)


def build_modeling_dataset(
    csv_path: str | Path,
    usecols: list[str],
    chunk_size: int,
    max_rows: int | None,
    allowed_day_offsets: list[int],
    success_statuses: dict[str, list[str]],
    success_scores: dict[str, dict[str, float]],
    positive_boost: float,
    emi_cycles: list[int],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    path = Path(csv_path)
    history_chunks: list[pd.DataFrame] = []
    current_chunks: list[pd.DataFrame] = []
    allowed = set(allowed_day_offsets)

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        history, current = _prepare_chunk(
            chunk=chunk,
            allowed_day_offsets=allowed,
            success_statuses=success_statuses,
            success_scores=success_scores,
            positive_boost=positive_boost,
            emi_cycles=emi_cycles,
        )
        if not history.empty:
            history_chunks.append(history)
        if not current.empty:
            current_chunks.append(current)

    if not current_chunks:
        raise ValueError("No current-window training rows were prepared from the source CSV.")

    history_frame = pd.concat(history_chunks, ignore_index=True)
    current_frame = pd.concat(current_chunks, ignore_index=True)

    history_agg = _aggregate_history(history_frame)
    current_frame["history_month"] = (
        current_frame["emi_date"].dt.to_period("M") - 1
    ).astype(str)
    modeling_frame = current_frame.merge(
        history_agg,
        on=["apac_card_number", "history_month"],
        how="left",
    )
    for column in HISTORY_FEATURE_COLUMNS:
        modeling_frame[column] = pd.to_numeric(modeling_frame[column], errors="coerce").fillna(0.0)
    for column in DAY_AVAILABILITY_COLUMNS:
        modeling_frame[column] = pd.to_numeric(modeling_frame[column], errors="coerce").fillna(0).astype(int)
    modeling_frame["prev_month_same_day_available"] = modeling_frame.apply(
        lambda row: row[get_day_availability_column(int(row["day_offset"]))],
        axis=1,
    )

    modeling_frame, split_meta = _assign_month_split(modeling_frame)
    modeling_frame = _sample_rows_by_month(modeling_frame, max_rows=max_rows)
    return modeling_frame.reset_index(drop=True), split_meta


def build_base_population(df: pd.DataFrame) -> pd.DataFrame:
    latest_month = sorted(df["emi_month"].dropna().astype(str).unique().tolist())[-1]
    latest = df[df["emi_month"] == latest_month].copy()
    sort_cols = ["apac_card_number", "emi_date", "created_date"]
    ordered = latest.sort_values(sort_cols).copy()
    deduped = ordered.groupby(["apac_card_number", "emi_date"], as_index=False).tail(1)
    base_columns = [
        "apac_card_number",
        "risk",
        "collectable_amount",
        "campaign_identifier",
        "emi_date",
        *HISTORY_FEATURE_COLUMNS,
        *DAY_AVAILABILITY_COLUMNS,
    ]
    return deduped[base_columns].reset_index(drop=True)
