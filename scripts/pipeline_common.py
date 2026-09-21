from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


PREDUE_DAY_COLUMNS = [f"D-{day}" for day in range(5, 0, -1)]
POSTDUE_DAY_COLUMNS = [f"D+{day}" for day in range(1, 21)]
DAY_COLUMNS = [*PREDUE_DAY_COLUMNS, *POSTDUE_DAY_COLUMNS]
SCHEDULE_DAY_COLUMNS = [*PREDUE_DAY_COLUMNS, "D", *POSTDUE_DAY_COLUMNS]
DAY_WEIGHT_COLUMN_MAP = {day: f"{day}__WEIGHT" for day in DAY_COLUMNS}
DAY_WEIGHT_COLUMNS = [DAY_WEIGHT_COLUMN_MAP[day] for day in DAY_COLUMNS]
NON_FEATURE_COLUMNS = DAY_COLUMNS + DAY_WEIGHT_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK", "VERTICAL", "bounce_flag", "paid_flag"]
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

    use_payment_flags = os.getenv("USE_PAYMENT_FLAGS_IN_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}

    payment_flag_cols = {"bounce_flag", "paid_flag"} if use_payment_flags else set()
    numeric_columns = [
        col
        for col in features.columns
        if col not in {key_column, "MONTH", "RISK", "VERTICAL", "MONTH_PERIOD", "bounce_flag", "paid_flag"}
    ]

    lag_columns = [
        f"{column}_M{lag}"
        for column in numeric_columns
        for lag in range(1, history_window_months + 1)
    ]
    if use_payment_flags:
        for pf in ["bounce_flag", "paid_flag"]:
            if pf in features.columns:
                lag_columns.extend([f"{pf}_M{lag}" for lag in range(1, history_window_months + 1)])
        lag_columns.extend([f"has_history_M{lag}" for lag in range(1, history_window_months + 1)])
        lag_columns.append("history_months_count")

    if features.empty:
        base_columns = [key_column, "MONTH", *lag_columns, "RISK"]
        if "VERTICAL" in features.columns:
            base_columns.append("VERTICAL")
        return pd.DataFrame(columns=base_columns)

    features[numeric_columns] = features[numeric_columns].fillna(0)
    lagged_parts = [features[[key_column, "MONTH_PERIOD"]].copy()]

    for lag in range(1, history_window_months + 1):
        shifted_num = features.groupby(key_column, sort=False)[numeric_columns].shift(lag - 1)
        lagged_parts.append(shifted_num.add_suffix(f"_M{lag}"))

        if use_payment_flags:
            # 3-State Encoding for payment flags: 1=Paid/Bounced, 0=False, -1=No History (Account didn't exist yet)
            for pf in ["bounce_flag", "paid_flag"]:
                if pf in features.columns:
                    shifted_pf = features.groupby(key_column, sort=False)[pf].shift(lag - 1)
                    shifted_pf_encoded = pd.to_numeric(shifted_pf, errors="coerce").fillna(-1).astype(int)
                    shifted_pf_df = pd.DataFrame({f"{pf}_M{lag}": shifted_pf_encoded.values}, index=features.index)
                    lagged_parts.append(shifted_pf_df)

            # Presence Indicator: 1 if month data existed, 0 if account didn't exist yet
            has_hist = features.groupby(key_column, sort=False)["MONTH_PERIOD"].shift(lag - 1).notna().astype(int)
            lagged_parts.append(pd.DataFrame({f"has_history_M{lag}": has_hist.values}, index=features.index))

    rolled = pd.concat(lagged_parts, axis=1)
    rolled["MONTH"] = (
        rolled["MONTH_PERIOD"].dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
    )
    rolled["RISK"] = features["RISK"].fillna("UNKNOWN").astype(str).str.upper().values

    # Clean numeric lags
    num_lags = [f"{column}_M{lag}" for column in numeric_columns for lag in range(1, history_window_months + 1)]
    rolled[num_lags] = rolled[num_lags].fillna(0)

    if use_payment_flags:
        hist_cols = [f"has_history_M{lag}" for lag in range(1, history_window_months + 1)]
        rolled["history_months_count"] = rolled[hist_cols].sum(axis=1).astype(int)

    returned_cols = [key_column, "MONTH", *lag_columns, "RISK"]
    if "bounce_flag" in features.columns and "bounce_flag" not in returned_cols:
        returned_cols.append("bounce_flag")
        rolled["bounce_flag"] = features["bounce_flag"].values
    if "paid_flag" in features.columns and "paid_flag" not in returned_cols:
        returned_cols.append("paid_flag")
        rolled["paid_flag"] = features["paid_flag"].values
    if "VERTICAL" in features.columns:
        returned_cols.append("VERTICAL")
        rolled["VERTICAL"] = features["VERTICAL"].fillna("UNKNOWN").astype(str).str.upper().values
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
        
        # For D-5 and D-1, contact full base by bypassing '-' and selecting 2nd best active strategy
        is_compulsory_day = day is not None and str(day).strip().upper() in {"D-5", "D-1", "D5", "D1"}
        if is_compulsory_day:
            active_labels = [label for label in labels if label != "-"]
            selected_labels = active_labels[:k] if active_labels else labels[:k]
            if labels and labels[0] == "-":
                minus_predictions_removed += 1
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
