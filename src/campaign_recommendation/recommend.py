from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import get_day_availability_column
from .features import get_model_matrix


CHANNEL_LABELS = {
    "SMS": "SMS",
    "WHATSAPP": "WH",
    "VOICE": "IVR",
    "VOICE_BOT": "VOICE_BOT",
}

@dataclass(frozen=True)
class RecommendationPolicy:
    allowed_day_offsets: list[int]
    risk_quota: dict[str, int]
    send_hour_window: dict[str, int]
    channel_priority: list[str]


def _format_hour(hour: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    normalized = hour % 12
    if normalized == 0:
        normalized = 12
    return f"{normalized}{suffix}"


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


def build_strategy_label(communication_type: str, send_hour: int | float | None) -> str:
    if communication_type in {None, "", "-"} or pd.isna(send_hour):
        return "-"
    channel = CHANNEL_LABELS.get(str(communication_type).upper(), str(communication_type).upper())
    return f"{channel}-{_format_hour(int(send_hour))}"


def generate_candidates(base_population: pd.DataFrame, policy: RecommendationPolicy) -> pd.DataFrame:
    rows: list[dict] = []
    for record in base_population.to_dict(orient="records"):
        risk = str(record.get("risk", "unknown")).lower().strip()
        for day_offset in policy.allowed_day_offsets:
            availability_column = get_day_availability_column(day_offset)
            same_day_available = int(record.get(availability_column, 0))
            for channel in policy.channel_priority:
                for hour in candidate_hours(policy.send_hour_window):
                    candidate = dict(record)
                    candidate.update(
                        {
                            "apac_card_number": record["apac_card_number"],
                            "risk": risk,
                            "collectable_amount": record.get("collectable_amount", 0.0),
                            "campaign_identifier": record.get("campaign_identifier", "UNKNOWN"),
                            "emi_date": record["emi_date"],
                            "day_offset": day_offset,
                            "communication_type": channel,
                            "send_hour": hour,
                            "prev_month_same_day_available": same_day_available,
                        }
                    )
                    rows.append(candidate)
    return pd.DataFrame(rows)


def score_candidates(model, candidates: pd.DataFrame) -> pd.DataFrame:
    features = get_model_matrix(candidates)
    scored = candidates.copy()
    scored["predicted_success_probability"] = model.predict_proba(features)[:, 1]
    scored["recommendation_day"] = scored["day_offset"].apply(
        lambda value: f"D{value}" if value < 0 else f"D+{value}"
    )
    scored["strategy_label"] = scored.apply(
        lambda row: build_strategy_label(row["communication_type"], row["send_hour"]),
        axis=1,
    )
    return scored


def select_best_candidate_per_day(scored_candidates: pd.DataFrame, policy: RecommendationPolicy) -> pd.DataFrame:
    scored = scored_candidates[scored_candidates["prev_month_same_day_available"] > 0].copy()
    scored["channel_rank"] = scored["communication_type"].map(
        {channel: rank for rank, channel in enumerate(policy.channel_priority, start=1)}
    )
    scored = scored.sort_values(
        ["apac_card_number", "day_offset", "predicted_success_probability", "channel_rank"],
        ascending=[True, True, False, True],
    )
    return scored.groupby(["apac_card_number", "day_offset"], group_keys=False).head(1).reset_index(drop=True)


def select_recommendations(scored_candidates: pd.DataFrame, policy: RecommendationPolicy) -> pd.DataFrame:
    scored = scored_candidates[scored_candidates["prev_month_same_day_available"] > 0].copy()
    scored["channel_rank"] = scored["communication_type"].map(
        {channel: rank for rank, channel in enumerate(policy.channel_priority, start=1)}
    )
    scored = scored.sort_values(
        ["apac_card_number", "day_offset", "predicted_success_probability", "channel_rank"],
        ascending=[True, True, False, True],
    )

    selected_frames: list[pd.DataFrame] = []
    for risk, quota in policy.risk_quota.items():
        group = scored[scored["risk"] == risk].copy()
        if group.empty:
            continue
        chosen = group.groupby(["apac_card_number", "day_offset"], group_keys=False).head(quota).copy()
        chosen["slot_rank"] = chosen.groupby(["apac_card_number", "day_offset"]).cumcount() + 1
        selected_frames.append(chosen)

    if not selected_frames:
        return scored.iloc[0:0].copy()

    result = pd.concat(selected_frames, ignore_index=True)
    return result.sort_values(
        ["apac_card_number", "day_offset", "slot_rank"], ascending=[True, True, True]
    ).reset_index(drop=True)


def build_strategy_calendar(
    base_population: pd.DataFrame,
    selected_recommendations: pd.DataFrame,
    policy: RecommendationPolicy,
) -> pd.DataFrame:
    rows: list[dict] = []
    available_lookup = {}
    for record in base_population.to_dict(orient="records"):
        account_id = record["apac_card_number"]
        for day_offset in policy.allowed_day_offsets:
            available_lookup[(account_id, day_offset)] = int(record.get(get_day_availability_column(day_offset), 0))

    selected_lookup = {}
    if not selected_recommendations.empty:
        for row in selected_recommendations.to_dict(orient="records"):
            key = (row["apac_card_number"], row["day_offset"])
            selected_lookup.setdefault(key, []).append(row)

    for record in base_population.to_dict(orient="records"):
        account_id = record["apac_card_number"]
        risk = str(record.get("risk", "unknown")).lower().strip()
        for day_offset in policy.allowed_day_offsets:
            key = (account_id, day_offset)
            day_rows = selected_lookup.get(key, [])
            if day_rows:
                for row in day_rows:
                    enriched = dict(row)
                    enriched["day_available"] = available_lookup.get(key, 0)
                    rows.append(enriched)
                continue

            placeholder = dict(record)
            placeholder.update(
                {
                    "risk": risk,
                    "day_offset": day_offset,
                    "recommendation_day": f"D{day_offset}" if day_offset < 0 else f"D+{day_offset}",
                    "slot_rank": 0,
                    "communication_type": "-",
                    "send_hour": pd.NA,
                    "predicted_success_probability": pd.NA,
                    "channel_rank": pd.NA,
                    "strategy_label": "-",
                    "day_available": available_lookup.get(key, 0),
                }
            )
            rows.append(placeholder)

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["apac_card_number", "day_offset", "slot_rank"], ascending=[True, True, True]
    ).reset_index(drop=True)


def build_bucket_counts(strategy_calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    populated = strategy_calendar[strategy_calendar["strategy_label"] != "-"].copy()

    overall_counts = (
        populated.groupby(["risk", "strategy_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["risk", "count", "strategy_label"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    daywise_counts = (
        populated.groupby(["risk", "recommendation_day", "strategy_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            ["risk", "recommendation_day", "count", "strategy_label"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )
    return overall_counts, daywise_counts
