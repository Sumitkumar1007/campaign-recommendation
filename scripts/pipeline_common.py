from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DAY_COLUMNS = ["D-5", "D-4", "D-3", "D-2", "D-1", "D+1", "D+2", "D+3", "D+4", "D+5"]
NON_FEATURE_COLUMNS = DAY_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK"]
RISK_TOP_K = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def resolve_emi_cycle(config_file: str | Path, emi_cycle_override: str = "") -> list[int]:
    if emi_cycle_override:
        raw_cycle = [value.strip() for value in emi_cycle_override.split(",") if value.strip()]
    else:
        with Path(config_file).open("r", encoding="utf-8") as handle:
            raw_cycle = json.load(handle).get("emi_cycle", [])

    cycles: list[int] = []
    for value in raw_cycle:
        day = int(value)
        if day < 1 or day > 31:
            raise ValueError(f"Invalid EMI cycle day {value!r}. Expected a day from 1 to 31.")
        cycles.append(day)
    return sorted(set(cycles))


def candidate_hours(send_hour_window: dict[str, int]) -> list[int]:
    start_hour = int(send_hour_window["start_hour"])
    end_hour = int(send_hour_window["end_hour"])
    step_hours = int(send_hour_window.get("step_hours", 1))
    if start_hour < 0 or end_hour > 23 or start_hour > end_hour:
        raise ValueError("send_hour_window must use 0-23 hours with start_hour <= end_hour")
    if step_hours <= 0:
        raise ValueError("send_hour_window.step_hours must be greater than 0")
    hours = list(range(start_hour, end_hour + 1, step_hours))
    if hours[-1] != end_hour:
        hours.append(end_hour)
    return hours


def month_to_period(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce").dt.to_period("M")


def build_rolling_feature_windows(
    features: pd.DataFrame,
    history_window_months: int,
) -> pd.DataFrame:
    features = features.copy()
    features["MONTH_PERIOD"] = month_to_period(features["MONTH"])
    features = features.dropna(subset=["MONTH_PERIOD"])
    features = (
        features.sort_values(["APAC_CARD_NUMBER", "MONTH_PERIOD"])
        .drop_duplicates(subset=["APAC_CARD_NUMBER", "MONTH_PERIOD"], keep="last")
        .reset_index(drop=True)
    )

    numeric_columns = [
        col
        for col in features.columns
        if col not in {"APAC_CARD_NUMBER", "MONTH", "RISK", "MONTH_PERIOD"}
    ]
    if features.empty:
        return features[["APAC_CARD_NUMBER", "MONTH", *numeric_columns, "RISK"]].copy()

    features[numeric_columns] = features[numeric_columns].fillna(0)
    rolled_numeric = (
        features.groupby("APAC_CARD_NUMBER", sort=False)[numeric_columns]
        .rolling(window=history_window_months, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rolled = features[["APAC_CARD_NUMBER", "MONTH_PERIOD"]].copy()
    rolled[numeric_columns] = rolled_numeric[numeric_columns]
    rolled["MONTH"] = (
        rolled["MONTH_PERIOD"].dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
    )
    rolled["RISK"] = features["RISK"].fillna("UNKNOWN").astype(str).str.upper().values
    return rolled[["APAC_CARD_NUMBER", "MONTH", *numeric_columns, "RISK"]]


def prepare_next_month_dataset(
    feature_file: Path,
    schedule_file: Path,
    target_offset_months: int,
    history_window_months: int | None = None,
) -> pd.DataFrame:
    features = pd.read_csv(feature_file).copy()
    schedule = pd.read_csv(schedule_file).copy()
    if history_window_months is not None:
        features = build_rolling_feature_windows(features, history_window_months)

    features["SOURCE_MONTH"] = features["MONTH"]
    features["SOURCE_MONTH_PERIOD"] = month_to_period(features["SOURCE_MONTH"])
    features["TARGET_MONTH_PERIOD"] = features["SOURCE_MONTH_PERIOD"] + target_offset_months
    features = features.drop(columns=["MONTH"])

    schedule["TARGET_MONTH"] = schedule["MONTH"]
    schedule["TARGET_MONTH_PERIOD"] = month_to_period(schedule["TARGET_MONTH"])
    schedule = schedule.rename(
        columns={
            "Loan_number": "APAC_CARD_NUMBER",
            "RISK": "TARGET_RISK",
        }
    )
    schedule = schedule.drop(columns=["MONTH"])

    return features.merge(
        schedule[
            ["APAC_CARD_NUMBER", "TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK", *DAY_COLUMNS]
        ],
        on=["APAC_CARD_NUMBER", "TARGET_MONTH_PERIOD"],
        how="left",
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").copy()
    feature_df["RISK"] = feature_df["RISK"].fillna("UNKNOWN")
    feature_df["SOURCE_MONTH"] = feature_df["SOURCE_MONTH"].fillna("UNKNOWN")
    feature_df = pd.get_dummies(
        feature_df,
        columns=["SOURCE_MONTH", "RISK"],
        dummy_na=False,
    )
    feature_df = feature_df.drop(columns=["APAC_CARD_NUMBER", "SOURCE_MONTH_PERIOD"], errors="ignore")
    return feature_df.fillna(0)


def split_by_source_month(
    dataset: pd.DataFrame,
    months: list[str],
    require_target: bool,
) -> pd.DataFrame:
    selected = dataset[dataset["SOURCE_MONTH"].isin(months)].copy()
    if require_target:
        selected = selected.dropna(subset=["TARGET_MONTH"])
    return selected


def predict_top_k_by_risk(
    model,
    encoder,
    X: pd.DataFrame,
    risks: pd.Series,
) -> pd.Series:
    probabilities = model.predict_proba(X)
    output: list[str] = []
    for row_probs, risk in zip(probabilities, risks.fillna("LOW").astype(str), strict=False):
        k = RISK_TOP_K.get(risk.upper(), 1)
        top_indices = row_probs.argsort()[-k:][::-1]
        labels = [str(encoder.classes_[index]) for index in top_indices]
        output.append("|".join(labels))
    return pd.Series(output, index=X.index)
