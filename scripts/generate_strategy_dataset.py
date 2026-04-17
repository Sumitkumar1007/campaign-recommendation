from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from project_paths import COMMUNICATION_DATA_DIR, TRAINING_DATA_DIR, ensure_parent_dir


TIME_BUCKETS = {
    8: "8AM",
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


def process_chunk(
    chunk: pd.DataFrame,
    sample_cards: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = chunk.copy()
    df["APAC_CARD_NUMBER"] = df["apac_card_number"].astype(str).str.strip()
    if sample_cards is not None:
        df = df[df["APAC_CARD_NUMBER"].isin(sample_cards)].copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
    df["COMM_TYPE"] = normalize_comm_type(df["communication_type"])
    df["STATUS"] = normalize_status(df["comm_status"])
    df["LANGUAGE"] = normalize_language(df["verbiage_language"])
    df["emi_date"] = pd.to_datetime(df["emi_date"], errors="coerce").dt.normalize()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    created_ts = pd.to_datetime(df["created_date"], errors="coerce")
    df["hr"] = created_ts.dt.hour.astype("Int64")

    df = df[
        df["APAC_CARD_NUMBER"].ne("")
        & df["COMM_TYPE"].notna()
        & df["emi_date"].notna()
        & df["date"].notna()
        & created_ts.notna()
    ].copy()

    df["offset"] = (df["date"] - df["emi_date"]).dt.days
    df = df[df["offset"].between(-5, 5) & df["offset"].ne(0)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["MONTH"] = df["emi_date"].dt.strftime("%b-%Y").str.upper()
    df["DAY"] = df["offset"].map(day_label)
    df["IS_SUCCESS"] = df.apply(
        lambda row: row["STATUS"] in SUCCESS_STATUS_MAP[row["COMM_TYPE"]],
        axis=1,
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
    return feature_counts, strategy_counts


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
) -> None:
    output_file = ensure_parent_dir(output_file)
    exclude_tokens = {token.upper() for token in (exclude_months or [])}
    csv_files = sorted(
        csv_file
        for csv_file in input_dir.glob("*.csv")
        if not any(token in csv_file.name.upper() for token in exclude_tokens)
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    feature_parts: list[pd.DataFrame] = []
    strategy_parts: list[pd.DataFrame] = []
    sample_cards = (
        collect_sample_cards(csv_files, chunksize, sample_size)
        if sample_size > 0
        else None
    )

    for csv_file in csv_files:
        print(f"Processing {csv_file.name} ...")
        for chunk in pd.read_csv(csv_file, usecols=USECOLS, chunksize=chunksize):
            feature_counts, strategy_counts = process_chunk(chunk, sample_cards=sample_cards)
            if not feature_counts.empty:
                feature_parts.append(feature_counts)
            if not strategy_counts.empty:
                strategy_parts.append(strategy_counts)

    if not feature_parts:
        raise ValueError("No rows matched the D-5 to D+5 window.")

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
        if column not in {"APAC_CARD_NUMBER", "MONTH", "DAY", "PREDICTED_STRATEGY"}
    )
    wide = wide[
        ["APAC_CARD_NUMBER", "MONTH", "DAY", *feature_columns, "PREDICTED_STRATEGY"]
    ]
    wide.to_csv(output_file, index=False)
    print(f"Saved {len(wide):,} rows to {output_file}")


def main() -> None:
    args = parse_args()
    build_dataset(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output_file),
        chunksize=args.chunksize,
        sample_size=args.sample_cards,
        exclude_months=args.exclude_months,
    )


if __name__ == "__main__":
    main()
