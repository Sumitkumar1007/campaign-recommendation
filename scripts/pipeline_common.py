from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDUE_DAY_COLUMNS = [f"D-{day}" for day in range(5, 0, -1)]
POSTDUE_DAY_COLUMNS = [f"D+{day}" for day in range(1, 21)]
DAY_COLUMNS = [*PREDUE_DAY_COLUMNS, *POSTDUE_DAY_COLUMNS]
SCHEDULE_DAY_COLUMNS = [*PREDUE_DAY_COLUMNS, "D", *POSTDUE_DAY_COLUMNS]
DAY_WEIGHT_COLUMN_MAP = {day: f"{day}__WEIGHT" for day in DAY_COLUMNS}
DAY_WEIGHT_COLUMNS = [DAY_WEIGHT_COLUMN_MAP[day] for day in DAY_COLUMNS]
NON_FEATURE_COLUMNS = DAY_COLUMNS + DAY_WEIGHT_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK", "VERTICAL"]
RISK_TOP_K = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}



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
    key_column = "ENTITY_KEY" if "ENTITY_KEY" in features.columns else "APAC_CARD_NUMBER"
    features["MONTH_PERIOD"] = month_to_period(features["MONTH"])
    features = features.dropna(subset=["MONTH_PERIOD"])
    features = (
        features.sort_values([key_column, "MONTH_PERIOD"])
        .drop_duplicates(subset=[key_column, "MONTH_PERIOD"], keep="last")
        .reset_index(drop=True)
    )

    numeric_columns = [
        col
        for col in features.columns
        if col not in {key_column, "MONTH", "RISK", "VERTICAL", "MONTH_PERIOD"}
    ]
    if features.empty:
        return features[[key_column, "MONTH", *numeric_columns, "RISK"]].copy()

    features[numeric_columns] = features[numeric_columns].fillna(0)
    rolled_numeric = (
        features.groupby(key_column, sort=False)[numeric_columns]
        .rolling(window=history_window_months, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rolled = features[[key_column, "MONTH_PERIOD"]].copy()
    rolled[numeric_columns] = rolled_numeric[numeric_columns]
    rolled["MONTH"] = (
        rolled["MONTH_PERIOD"].dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
    )
    rolled["RISK"] = features["RISK"].fillna("UNKNOWN").astype(str).str.upper().values
    if "VERTICAL" in features.columns:
        rolled["VERTICAL"] = features["VERTICAL"].fillna("UNKNOWN").astype(str).str.upper().values
        return rolled[[key_column, "MONTH", *numeric_columns, "RISK", "VERTICAL"]]
    return rolled[[key_column, "MONTH", *numeric_columns, "RISK"]]


def prepare_next_month_dataset(
    feature_file: Path,
    schedule_file: Path,
    target_offset_months: int,
    history_window_months: int | None = None,
) -> pd.DataFrame:
    features = pd.read_csv(feature_file).copy()
    schedule = pd.read_csv(schedule_file).copy()
    key_column = "ENTITY_KEY" if "ENTITY_KEY" in features.columns or "ENTITY_KEY" in schedule.columns else "APAC_CARD_NUMBER"
    if history_window_months is not None:
        features = build_rolling_feature_windows(features, history_window_months)

    features["SOURCE_MONTH"] = features["MONTH"]
    features["SOURCE_MONTH_PERIOD"] = month_to_period(features["SOURCE_MONTH"])
    features["TARGET_MONTH_PERIOD"] = features["SOURCE_MONTH_PERIOD"] + target_offset_months
    features = features.drop(columns=["MONTH"])

    schedule["TARGET_MONTH"] = schedule["MONTH"]
    schedule["TARGET_MONTH_PERIOD"] = month_to_period(schedule["TARGET_MONTH"])
    if key_column == "ENTITY_KEY":
        schedule = schedule.rename(columns={"RISK": "TARGET_RISK"})
    else:
        schedule = schedule.rename(columns={"Loan_number": key_column, "RISK": "TARGET_RISK"})
    schedule = schedule.drop(columns=["MONTH"])
    for column in DAY_WEIGHT_COLUMNS:
        if column not in schedule.columns:
            schedule[column] = 1.0

    return features.merge(
        schedule[
            [
                key_column,
                "TARGET_MONTH",
                "TARGET_MONTH_PERIOD",
                "TARGET_RISK",
                *DAY_COLUMNS,
                *DAY_WEIGHT_COLUMNS,
            ]
        ],
        on=[key_column, "TARGET_MONTH_PERIOD"],
        how="left",
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    active_key = "ENTITY_KEY" if "ENTITY_KEY" in df.columns else "APAC_CARD_NUMBER"
    feature_df = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").copy()
    feature_df["RISK"] = feature_df["RISK"].fillna("UNKNOWN")
    feature_df["SOURCE_MONTH"] = feature_df["SOURCE_MONTH"].fillna("UNKNOWN")
    feature_df = pd.get_dummies(
        feature_df,
        columns=["SOURCE_MONTH", "RISK"],
        dummy_na=False,
    )
    feature_df = feature_df.drop(columns=[active_key, "SOURCE_MONTH_PERIOD", "VERTICAL"], errors="ignore")
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
