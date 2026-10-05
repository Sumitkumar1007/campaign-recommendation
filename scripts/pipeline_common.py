from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDUE_DAY_COLUMNS = [f"D-{day}" for day in range(5, 0, -1)]
POSTDUE_DAY_COLUMNS = [f"D+{day}" for day in range(1, 21)]
DAY_COLUMNS = [*PREDUE_DAY_COLUMNS, *POSTDUE_DAY_COLUMNS]
SCHEDULE_DAY_COLUMNS = [*PREDUE_DAY_COLUMNS, "D", *POSTDUE_DAY_COLUMNS]
DAY_WEIGHT_COLUMN_MAP = {day: f"{day}__WEIGHT" for day in DAY_COLUMNS}
DAY_WEIGHT_COLUMNS = [DAY_WEIGHT_COLUMN_MAP[day] for day in DAY_COLUMNS]
AUDIT_PAYMENT_COLUMNS = ["HAS_LINK_PAYMENT", "HAS_EXTERNAL_PAYMENT", "LINK_PAYMENT_FLAG", "EXTERNAL_PAYMENT_FLAG"]
NON_FEATURE_COLUMNS = DAY_COLUMNS + DAY_WEIGHT_COLUMNS + AUDIT_PAYMENT_COLUMNS + ["TARGET_MONTH", "TARGET_MONTH_PERIOD", "TARGET_RISK", "VERTICAL"]
RISK_TOP_K = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}

PREDUE_LIMITS_BY_RISK = {
    "LOW": {"target_exact": 2},
    "MEDIUM": {"target_exact": 3},
    "HIGH": {"target_exact": 4},
}
DEFAULT_PREDUE_STRATEGY = "SMS-11AM-ENGLISH"


def apply_predue_risk_rules(df: pd.DataFrame, default_strategy: str = DEFAULT_PREDUE_STRATEGY) -> pd.DataFrame:
    """
    Enforces D-5 full base communication rule and risk-based pre-due touchpoint limits:
    - D-5 is mandatory for 100% of base population (defaults to SMS-11AM-ENGLISH if blank).
    - Ensures single communication per day (selects 1st option if multiple given).
    - LOW Risk: Exactly 2 communications across D-5..D-1 (1 per day).
    - MEDIUM Risk: Exactly 3 communications across D-5..D-1 (1 per day).
    - HIGH Risk: Exactly 4 communications across D-5..D-1 (1 per day).
    """
    df = df.copy()
    risk_col = "SOURCE_RISK" if "SOURCE_RISK" in df.columns else ("RISK" if "RISK" in df.columns else None)
    other_predue_days = ["D-4", "D-3", "D-2", "D-1"]

    def get_first_valid_communication(val: str) -> str:
        if "|" not in val:
            return val.strip()
        for p in val.split("|"):
            p = p.strip()
            if p not in ("", "-", "None", "nan"):
                return p
        return "-"

    # 1. Enforce single communication per day across all pre-due days
    for day in PREDUE_DAY_COLUMNS:
        if day in df.columns:
            df[day] = df[day].fillna("-").astype(str).apply(get_first_valid_communication)

    for idx, row in df.iterrows():
        risk_raw = str(row[risk_col]).upper().strip() if risk_col and pd.notna(row[risk_col]) else "LOW"
        limits = PREDUE_LIMITS_BY_RISK.get(risk_raw, PREDUE_LIMITS_BY_RISK["LOW"])
        target_exact = limits["target_exact"]

        # 2. Mandatory D-5 Full Base Communication
        d5_val = str(df.at[idx, "D-5"]).strip() if "D-5" in df.columns else "-"
        if d5_val in ("", "-", "None", "nan"):
            df.at[idx, "D-5"] = default_strategy

        # 3. Identify active days among D-4, D-3, D-2, D-1
        active_other = [
            day for day in other_predue_days
            if str(df.at[idx, day]).strip() not in ("", "-", "None", "nan")
        ]

        current_active = 1 + len(active_other)

        # 4. Trim extra pre-due days if count exceeds target_exact
        if current_active > target_exact:
            allowed_other_count = target_exact - 1
            allowed_other = set(active_other[:allowed_other_count])
            for day in other_predue_days:
                if day not in allowed_other:
                    df.at[idx, day] = "-"

        # 5. Fill in missing pre-due days if count is below target_exact
        elif current_active < target_exact:
            needed = target_exact - current_active
            priority_fill = ["D-1", "D-2", "D-3", "D-4"]
            filled = 0
            for day in priority_fill:
                if filled >= needed:
                    break
                val = str(df.at[idx, day]).strip() if day in df.columns else "-"
                if val in ("", "-", "None", "nan"):
                    df.at[idx, day] = default_strategy
                    filled += 1

    return df



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
    features[key_column] = features[key_column].astype(str).str.strip()
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
    lag_columns = [
        f"{column}_M{lag}"
        for column in numeric_columns
        for lag in range(1, history_window_months + 1)
    ]
    if features.empty:
        base_columns = [key_column, "MONTH", *lag_columns, "RISK"]
        if "VERTICAL" in features.columns:
            base_columns.append("VERTICAL")
        return pd.DataFrame(columns=base_columns)

    # Ensure numeric columns are numeric dtypes even when loaded as dtype=str
    for col in numeric_columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    lagged_parts = [features[[key_column, "MONTH_PERIOD"]].copy()]
    has_data_flags: dict[str, pd.Series] = {}

    for lag in range(1, history_window_months + 1):
        shifted = features.groupby(key_column, sort=False)[numeric_columns].shift(lag - 1)
        lagged_parts.append(shifted.add_suffix(f"_M{lag}"))
        
        # Indicator flag: 1 if historical month was present in extracted data for this entity, else 0
        has_data = shifted.iloc[:, 0].notna() if not shifted.empty else pd.Series(False, index=features.index)
        has_data_flags[f"HAS_M{lag}_DATA"] = has_data.astype(int)

    rolled = pd.concat(lagged_parts, axis=1)

    # Attach history indicator flags
    for flag_name, flag_series in has_data_flags.items():
        rolled[flag_name] = flag_series.values

    # Calculate count of available historical months per row
    available_months = sum(has_data_flags.values())
    available_months_clipped = available_months.clip(lower=1)

    # Compute Normalized Average (_MEAN) across available historical months for every numeric feature
    for column in numeric_columns:
        lag_cols = [f"{column}_M{lag}" for lag in range(1, history_window_months + 1)]
        sum_lags = rolled[lag_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1, skipna=True)
        rolled[f"{column}_MEAN"] = sum_lags / available_months_clipped

    rolled["MONTH"] = (
        rolled["MONTH_PERIOD"].dt.to_timestamp().dt.strftime("%b-%Y").str.upper()
    )
    rolled["RISK"] = features["RISK"].fillna("UNKNOWN").astype(str).str.upper().values
    rolled[lag_columns] = rolled[lag_columns].fillna(0)

    # Complete output column list preserving expected ordering
    all_lag_cols = [
        col for lag in range(1, history_window_months + 1)
        for col in [f"{c}_M{lag}" for c in numeric_columns]
    ]
    mean_cols = [f"{c}_MEAN" for c in numeric_columns]
    flag_cols = [f"HAS_M{lag}_DATA" for lag in range(1, history_window_months + 1)]

    extra_cols = ["RISK"]
    if "VERTICAL" in features.columns:
        rolled["VERTICAL"] = features["VERTICAL"].fillna("UNKNOWN").astype(str).str.upper().values
        extra_cols.append("VERTICAL")

    out_cols = [key_column, "MONTH", *all_lag_cols, *mean_cols, *flag_cols, *extra_cols]
    return rolled[out_cols]


def prepare_next_month_dataset(
    feature_file: Path,
    schedule_file: Path,
    target_offset_months: int,
    history_window_months: int | None = None,
) -> pd.DataFrame:
    features = pd.read_csv(feature_file, dtype=str).copy()
    schedule = pd.read_csv(schedule_file, dtype=str).copy()
    key_column = "ENTITY_KEY" if "ENTITY_KEY" in features.columns or "ENTITY_KEY" in schedule.columns else "APAC_CARD_NUMBER"
    features[key_column] = features[key_column].astype(str).str.strip()
    schedule[key_column] = schedule[key_column].astype(str).str.strip()
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

    schedule_cols = [
        key_column,
        "TARGET_MONTH",
        "TARGET_MONTH_PERIOD",
        "TARGET_RISK",
        *DAY_COLUMNS,
        *DAY_WEIGHT_COLUMNS,
    ]
    for col in AUDIT_PAYMENT_COLUMNS:
        if col in schedule.columns and col not in schedule_cols:
            schedule_cols.append(col)

    return features.merge(
        schedule[schedule_cols],
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
