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
# RISK_TOP_K = {
#     "LOW": 1,
#     "MEDIUM": 2,
#     "HIGH": 3,
# }
RISK_BOUNCE_TOP_K = {
    ("LOW", 0): 1,
    ("LOW", 1): 2,
    ("MEDIUM", 0): 2,
    ("MEDIUM", 1): 3,
    ("HIGH", 0): 3,
    ("HIGH", 1): 4,
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
        # if col not in {key_column, "MONTH", "RISK", "VERTICAL", "MONTH_PERIOD"}
        if col not in {key_column, "MONTH", "RISK", "VERTICAL", "MONTH_PERIOD", "bounce_flag", "paid_flag"}
    ]
    lag_columns = [
        f"{column}_M{lag}"
        for column in numeric_columns
        for lag in range(1, history_window_months + 1)
    ]
    if features.empty:
        # base_columns = [key_column, "MONTH", *lag_columns, "RISK"]
        base_columns = [key_column, "MONTH", *lag_columns]
        if "bounce_flag" in features.columns:
            base_columns.append("bounce_flag")
        if "paid_flag" in features.columns:
            base_columns.append("paid_flag")
        base_columns.append("RISK")
        if "VERTICAL" in features.columns:
            base_columns.append("VERTICAL")
        return pd.DataFrame(columns=base_columns)

    features[numeric_columns] = features[numeric_columns].fillna(0)
    lagged_parts = [features[[key_column, "MONTH_PERIOD"]].copy()]
    for lag in range(1, history_window_months + 1):
        shifted = features.groupby(key_column, sort=False)[numeric_columns].shift(lag - 1)
        lagged_parts.append(shifted.add_suffix(f"_M{lag}"))

    rolled = pd.concat(lagged_parts, axis=1)
    rolled["MONTH"] = (
        rolled["MONTH_PERIOD"].dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
    )
    rolled["RISK"] = features["RISK"].fillna("UNKNOWN").astype(str).str.upper().values
    rolled[lag_columns] = rolled[lag_columns].fillna(0)
    if "bounce_flag" in features.columns:
        rolled["bounce_flag"] = pd.to_numeric(features["bounce_flag"], errors="coerce").fillna(0).astype(int).values
    if "paid_flag" in features.columns:
        rolled["paid_flag"] = pd.to_numeric(features["paid_flag"], errors="coerce").fillna(0).astype(int).values

    returned_cols = [key_column, "MONTH", *lag_columns]
    if "bounce_flag" in features.columns:
        returned_cols.append("bounce_flag")
    if "paid_flag" in features.columns:
        returned_cols.append("paid_flag")
    returned_cols.append("RISK")
    if "VERTICAL" in features.columns:
        returned_cols.append("VERTICAL")
        rolled["VERTICAL"] = features["VERTICAL"].fillna("UNKNOWN").astype(str).str.upper().values
    #     return rolled[[key_column, "MONTH", *lag_columns, "RISK", "VERTICAL"]]
    # return rolled[[key_column, "MONTH", *lag_columns, "RISK"]]
    return rolled[returned_cols]


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
    bounce_flags: pd.Series | None = None,
    day: str | None = None,
    logger = None,
) -> pd.Series:
    probabilities = model.predict_proba(X)
    output: list[str] = []
    # for row_probs, risk in zip(probabilities, risks.fillna("LOW").astype(str), strict=False):
    #     k = RISK_TOP_K.get(risk.upper(), 1)
    #     top_indices = row_probs.argsort()[-k:][::-1]
    #     labels = [str(encoder.classes_[index]) for index in top_indices]
    #     output.append("|".join(labels))
    total_rows = len(X)
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    bounce_counts = {0: 0, 1: 0}
    quota_distribution = {}
    minus_predictions_removed = 0
    fewer_than_requested = 0
    recommendation_counts = {}

    if bounce_flags is None:
        bounce_flags = pd.Series(0, index=X.index)
    else:
        bounce_flags = pd.Series(bounce_flags).reindex(X.index).fillna(0).astype(int)

    for row_probs, risk, bounce in zip(probabilities, risks.fillna("LOW").astype(str), bounce_flags, strict=False):
        risk_upper = risk.upper()
        if risk_upper not in {"LOW", "MEDIUM", "HIGH"}:
            risk_upper = "LOW"
        risk_counts[risk_upper] += 1
        
        b_val = 1 if bounce == 1 else 0
        bounce_counts[b_val] += 1
        
        k = RISK_BOUNCE_TOP_K.get((risk_upper, b_val), 1)
        quota_distribution[k] = quota_distribution.get(k, 0) + 1
        
        sorted_indices = row_probs.argsort()[::-1]
        labels = [str(encoder.classes_[index]) for index in sorted_indices]
        
        is_override_day = day in {"D-5", "D-1"}
        if is_override_day:
            original_len = len(labels)
            labels_no_dash = [lbl for lbl in labels if lbl != "-"]
            removed_count = original_len - len(labels_no_dash)
            minus_predictions_removed += removed_count
            
            selected_labels = labels_no_dash[:k]
            if len(selected_labels) < k:
                fewer_than_requested += 1
        else:
            selected_labels = labels[:k]
            
        joined_label = "|".join(selected_labels)
        output.append(joined_label)
        
        label_len = len(selected_labels)
        recommendation_counts[label_len] = recommendation_counts.get(label_len, 0) + 1
        
    if logger is not None:
        logger.info(
            "Scoring summary | day=%s total_rows=%d risk_counts=%s bounce_counts=%s "
            "quota_distribution=%s minus_predictions_removed=%d fewer_than_requested=%d "
            "recommendation_counts=%s",
            day, total_rows, risk_counts, bounce_counts,
            quota_distribution, minus_predictions_removed, fewer_than_requested,
            recommendation_counts
        )

    return pd.Series(output, index=X.index)
