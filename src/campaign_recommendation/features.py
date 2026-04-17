from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DAY_AVAILABILITY_COLUMNS, HISTORY_FEATURE_COLUMNS


FEATURE_COLUMNS = [
    "risk",
    "communication_type",
    "campaign_identifier",
    "day_offset",
    "send_hour",
    "collectable_amount",
    "emi_day",
    "emi_month_number",
    "is_predue",
    "is_postdue",
    "prev_month_same_day_available",
    *HISTORY_FEATURE_COLUMNS,
    *DAY_AVAILABILITY_COLUMNS,
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["risk"] = features["risk"].fillna("unknown").astype(str).str.lower().str.strip()
    features["communication_type"] = (
        features["communication_type"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    )
    features["campaign_identifier"] = (
        features["campaign_identifier"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    )
    features["collectable_amount"] = pd.to_numeric(
        features["collectable_amount"], errors="coerce"
    ).fillna(0.0)
    features["day_offset"] = pd.to_numeric(features["day_offset"], errors="coerce").fillna(0).astype(int)
    features["send_hour"] = pd.to_numeric(features["send_hour"], errors="coerce").fillna(0).astype(int)
    emi_dates = pd.to_datetime(features["emi_date"], errors="coerce")
    features["emi_day"] = emi_dates.dt.day.fillna(0).astype(int)
    features["emi_month_number"] = emi_dates.dt.month.fillna(0).astype(int)
    features["is_predue"] = (features["day_offset"] < 0).astype(int)
    features["is_postdue"] = (features["day_offset"] > 0).astype(int)
    if "prev_month_same_day_available" not in features.columns:
        features["prev_month_same_day_available"] = 0
    features["prev_month_same_day_available"] = pd.to_numeric(
        features["prev_month_same_day_available"], errors="coerce"
    ).fillna(0).astype(int)

    for column in HISTORY_FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    for column in DAY_AVAILABILITY_COLUMNS:
        if column not in features.columns:
            features[column] = 0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0).astype(int)

    return features


def get_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_features(df)
    return engineered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
