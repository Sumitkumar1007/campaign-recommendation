from __future__ import annotations

import pandas as pd


def normalize_text(series: pd.Series, uppercase: bool = True) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.str.upper() if uppercase else values.str.lower()


def apply_success_label(df: pd.DataFrame, success_statuses: dict[str, list[str]]) -> pd.DataFrame:
    labeled = df.copy()
    labeled["communication_type"] = normalize_text(labeled["communication_type"])
    labeled["comm_status"] = normalize_text(labeled["comm_status"])

    success_lookup: dict[str, set[str]] = {
        channel.upper(): {status.upper() for status in statuses}
        for channel, statuses in success_statuses.items()
    }

    labeled["target_success"] = labeled.apply(
        lambda row: int(row["comm_status"] in success_lookup.get(row["communication_type"], set())),
        axis=1,
    )
    return labeled
