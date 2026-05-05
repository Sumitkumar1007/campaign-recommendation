from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import candidate_hours
from project_paths import COMMUNICATION_DATA_DIR, REPO_ROOT, TRAINING_DATA_DIR, ensure_parent_dir


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
}

SUCCESS_STATUS_MAP = {
    "SMS": {"DELIVERED", "CLICKED", "SENT"},
    "WH": {"DELIVERED", "READ", "CLICKED", "SENT"},
    "VOICE": {"CONNECTED", "CALL_CONNECTED"},
}

USECOLS = [
    "apac_card_number",
    "comm_status",
    "communication_type",
    "verbiage_language",
    "vertical",
    "risk",
    "emi_date",
    "date",
    "created_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a wide strategy dataset from MFL communication CSV files "
            "for D-5 to D+5, excluding D."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(COMMUNICATION_DATA_DIR),
        help="Directory containing monthly communication CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default=str(TRAINING_DATA_DIR / "strategy_training_dataset.csv"),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = chunk.copy()
    if "risk" not in df.columns:
        raise ValueError("Missing required risk column in communication data.")
    df["APAC_CARD_NUMBER"] = df["apac_card_number"].astype(str).str.strip()
    if sample_cards is not None:
        df = df[df["APAC_CARD_NUMBER"].isin(sample_cards)].copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df["COMM_TYPE"] = normalize_comm_type(df["communication_type"])
    df["STATUS"] = normalize_status(df["comm_status"])
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
        df["APAC_CARD_NUMBER"].ne("")
        & df["RISK"].ne("UNKNOWN")
        & df["COMM_TYPE"].notna()
        & df["emi_date"].notna()
        & df["date"].notna()
        & created_ts.notna()
    ].copy()

    df["offset"] = (df["date"] - df["emi_date"]).dt.days
    df = df[df["offset"].between(-5, 5) & df["offset"].ne(0)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    month_dates = df["emi_date"] if month_source == "emi_date" else created_ts.dt.normalize()
    df["MONTH"] = month_dates.dt.strftime("%b-%Y").str.upper()
    df["DAY"] = df["offset"].map(day_label)
    status_map = {key: set(values) for key, values in SUCCESS_STATUS_MAP.items()}
    df["IS_SUCCESS"] = False
    for comm_type, success_statuses in status_map.items():
        type_mask = df["COMM_TYPE"].eq(comm_type)
        if type_mask.any():
            df.loc[type_mask, "IS_SUCCESS"] = df.loc[type_mask, "STATUS"].isin(success_statuses)

    unknown_status = (
        df["COMM_TYPE"].isin(status_map)
        & df["STATUS"].ne("")
        & ~df["IS_SUCCESS"]
    )
    if unknown_status.any():
        unknown_counts = (
            df.loc[unknown_status]
            .groupby(["COMM_TYPE", "STATUS"], sort=False)
            .size()
            .reset_index(name="count")
        )
        print(
            "Observed non-success statuses in chunk: "
            + ", ".join(
                f"{row.COMM_TYPE}:{row.STATUS}={row.count}"
                for row in unknown_counts.itertuples(index=False)
            )
        )

    totals = (
        df.groupby(["APAC_CARD_NUMBER", "MONTH", "DAY", "COMM_TYPE"], sort=False)
        .size()
        .reset_index(name="count")
    )
    totals["feature"] = totals["COMM_TYPE"] + "_TOTAL_INTENSITY"

    failed = (
        df.loc[~df["IS_SUCCESS"]]
        .groupby(
            ["APAC_CARD_NUMBER", "MONTH", "DAY", "COMM_TYPE", "LANGUAGE"],
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
                ["APAC_CARD_NUMBER", "MONTH", "DAY", "feature"],
                sort=False,
            )
            .size()
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
                ["APAC_CARD_NUMBER", "MONTH", "DAY", "strategy"],
                sort=False,
            )
            .size()
            .reset_index(name="count")
            .rename(columns={"strategy": "feature"})
        )
    else:
        success_features = pd.DataFrame()
        strategy_counts = pd.DataFrame()

    feature_frames = [
        totals[["APAC_CARD_NUMBER", "MONTH", "DAY", "feature", "count"]],
    ]
    if not failed.empty:
        feature_frames.append(
            failed[["APAC_CARD_NUMBER", "MONTH", "DAY", "feature", "count"]]
        )
    if not success_features.empty:
        feature_frames.append(
            success_features[["APAC_CARD_NUMBER", "MONTH", "DAY", "feature", "count"]]
        )

    feature_counts = pd.concat(feature_frames, ignore_index=True)
    risk_counts = (
        df.groupby(["APAC_CARD_NUMBER", "MONTH", "DAY"], sort=False)[["RISK", "VERTICAL"]]
        .last()
        .reset_index()
    )
    return feature_counts, strategy_counts, risk_counts


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
) -> None:
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
    sample_cards = (
        collect_sample_cards(csv_files, chunksize, sample_size)
        if sample_size > 0
        else None
    )

    for csv_file in csv_files:
        print(f"Processing {csv_file.name} ...")
        for chunk in pd.read_csv(csv_file, usecols=USECOLS, chunksize=chunksize):
            feature_counts, strategy_counts, risk_counts = process_chunk(
                chunk,
                sample_cards=sample_cards,
                month_source=month_source,
                send_hour_window=send_hour_window,
            )
            if not feature_counts.empty:
                feature_parts.append(feature_counts)
            if not strategy_counts.empty:
                strategy_parts.append(strategy_counts)
            if not risk_counts.empty:
                risk_parts.append(risk_counts)

    if not feature_parts:
        raise ValueError("No rows matched the D-5 to D+5 window.")
    if not risk_parts:
        raise ValueError("No risk rows found in communication data.")

    features = (
        pd.concat(feature_parts, ignore_index=True)
        .groupby(["APAC_CARD_NUMBER", "MONTH", "DAY", "feature"], as_index=False)["count"]
        .sum()
        .rename(
            columns={
                "APAC_CARD_NUMBER": "apac_card_number",
                "MONTH": "month",
                "DAY": "day",
            }
        )
    )

    if strategy_parts:
        strategies = (
            pd.concat(strategy_parts, ignore_index=True)
            .groupby(["APAC_CARD_NUMBER", "MONTH", "DAY", "feature"], as_index=False)["count"]
            .sum()
            .rename(
                columns={
                    "APAC_CARD_NUMBER": "apac_card_number",
                    "MONTH": "month",
                    "DAY": "day",
                }
            )
        )
    else:
        strategies = pd.DataFrame(
            columns=["apac_card_number", "month", "day", "feature", "count"]
        )

    wide = (
        features.pivot_table(
            index=["apac_card_number", "month", "day"],
            columns="feature",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(
            columns={
                "apac_card_number": "APAC_CARD_NUMBER",
                "month": "MONTH",
                "day": "DAY",
            }
        )
    )
    risk_df = (
        pd.concat(risk_parts, ignore_index=True)
        .groupby(["APAC_CARD_NUMBER", "MONTH", "DAY"], as_index=False)[["RISK", "VERTICAL"]]
        .last()
    )
    wide = wide.merge(risk_df, on=["APAC_CARD_NUMBER", "MONTH", "DAY"], how="left")

    if not strategies.empty:
        strategies = strategies.sort_values(
            ["apac_card_number", "month", "day", "count", "feature"],
            ascending=[True, True, True, False, True],
        )
        predicted = strategies.drop_duplicates(
            subset=["apac_card_number", "month", "day"],
            keep="first",
        ).rename(
            columns={
                "apac_card_number": "APAC_CARD_NUMBER",
                "month": "MONTH",
                "day": "DAY",
                "feature": "PREDICTED_STRATEGY",
            }
        )
        wide = wide.merge(
            predicted[["APAC_CARD_NUMBER", "MONTH", "DAY", "PREDICTED_STRATEGY"]],
            on=["APAC_CARD_NUMBER", "MONTH", "DAY"],
            how="left",
        )
    else:
        wide["PREDICTED_STRATEGY"] = pd.NA

    feature_columns = sorted(
        column
        for column in wide.columns
        if column not in {"APAC_CARD_NUMBER", "MONTH", "DAY", "RISK", "VERTICAL", "PREDICTED_STRATEGY"}
    )
    wide = wide[
        ["APAC_CARD_NUMBER", "MONTH", "DAY", "RISK", "VERTICAL", *feature_columns, "PREDICTED_STRATEGY"]
    ]
    wide.to_csv(output_file, index=False)
    print(f"Saved {len(wide):,} rows to {output_file}")


def main() -> None:
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
    )


if __name__ == "__main__":
    main()
