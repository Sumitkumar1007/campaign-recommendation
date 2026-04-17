from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import DAY_AVAILABILITY_COLUMNS, HISTORY_FEATURE_COLUMNS, build_base_population, build_modeling_dataset
from ..explain import explain_prediction
from ..modeling import load_model
from ..recommend import (
    RecommendationPolicy,
    build_bucket_counts,
    build_strategy_calendar,
    generate_candidates,
    score_candidates,
    select_best_candidate_per_day,
    select_recommendations,
)
from ..settings import AppConfig


def _load_base_population(config: AppConfig, input_csv: str | None) -> pd.DataFrame:
    if input_csv:
        frame = pd.read_csv(input_csv)
        if "collectable_amount" not in frame.columns and "outstanding_balance" in frame.columns:
            frame["collectable_amount"] = frame["outstanding_balance"]
        if "campaign_identifier" not in frame.columns:
            frame["campaign_identifier"] = "UNKNOWN"

        required = {"apac_card_number", "risk", "collectable_amount", "campaign_identifier", "emi_date"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Base population input is missing required columns: {sorted(missing)}")
        frame = frame.copy()
        for column in HISTORY_FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        for column in DAY_AVAILABILITY_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0
        keep_columns = list(required) + HISTORY_FEATURE_COLUMNS + DAY_AVAILABILITY_COLUMNS
        return frame[keep_columns].copy()

    training_cfg = config.raw["training"]
    historical, _ = build_modeling_dataset(
        csv_path=config.training_data_path,
        usecols=training_cfg["usecols"],
        chunk_size=training_cfg["chunk_size"],
        max_rows=training_cfg["max_rows"],
        allowed_day_offsets=config.raw["allowed_day_offsets"],
        success_statuses=config.raw["success_statuses"],
        success_scores=config.raw["success_scores"],
        positive_boost=training_cfg["sample_weight_positive_boost"],
        emi_day_filter=config.raw["emi_day_filter"],
    )
    return build_base_population(historical)


def run_recommendations(config: AppConfig, model_dir: str | None = None, input_csv: str | None = None) -> Path:
    chosen_model_dir = Path(model_dir) if model_dir else config.model_output_dir
    model = load_model(chosen_model_dir)
    base_population = _load_base_population(config=config, input_csv=input_csv)

    policy = RecommendationPolicy(
        allowed_day_offsets=config.raw["allowed_day_offsets"],
        risk_quota=config.raw["risk_quota"],
        candidate_hours=config.raw["candidate_hours"],
        channel_priority=config.raw["channel_priority"],
    )

    candidates = generate_candidates(base_population=base_population, policy=policy)
    scored = score_candidates(model=model, candidates=candidates)
    selected = select_recommendations(scored_candidates=scored, policy=policy)
    strategy_calendar = build_strategy_calendar(
        base_population=base_population,
        selected_recommendations=selected,
        policy=policy,
    )

    output_path = config.recommendation_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_calendar.to_csv(output_path, index=False)
    overall_counts, daywise_counts = build_bucket_counts(strategy_calendar)
    overall_counts.to_csv(output_path.parent / "bucket_counts.csv", index=False)
    daywise_counts.to_csv(output_path.parent / "bucket_counts_by_day.csv", index=False)
    return output_path


def run_account_explanation(
    config: AppConfig,
    account_id: str,
    model_dir: str | None = None,
    input_csv: str | None = None,
) -> dict:
    chosen_model_dir = Path(model_dir) if model_dir else config.model_output_dir
    model = load_model(chosen_model_dir)
    base_population = None
    if input_csv is None and config.recommendation_output_path.exists():
        recommendation_frame = pd.read_csv(config.recommendation_output_path)
        account_rows = recommendation_frame[recommendation_frame["apac_card_number"] == account_id].copy()
        if not account_rows.empty:
            keep_columns = [
                "apac_card_number",
                "risk",
                "collectable_amount",
                "campaign_identifier",
                "emi_date",
                *HISTORY_FEATURE_COLUMNS,
                *DAY_AVAILABILITY_COLUMNS,
            ]
            available_columns = [column for column in keep_columns if column in account_rows.columns]
            base_population = account_rows[available_columns].drop_duplicates().reset_index(drop=True)

    if base_population is None:
        base_population = _load_base_population(config=config, input_csv=input_csv)
    account_frame = base_population[base_population["apac_card_number"] == account_id].copy()
    if account_frame.empty:
        raise ValueError(f"Account {account_id} was not found in the available base population.")

    policy = RecommendationPolicy(
        allowed_day_offsets=config.raw["allowed_day_offsets"],
        risk_quota=config.raw["risk_quota"],
        candidate_hours=config.raw["candidate_hours"],
        channel_priority=config.raw["channel_priority"],
    )

    candidates = generate_candidates(base_population=account_frame, policy=policy)
    scored = score_candidates(model=model, candidates=candidates)
    best_per_day = select_best_candidate_per_day(scored_candidates=scored, policy=policy)
    final_selection = select_recommendations(scored_candidates=scored, policy=policy)
    strategy_calendar = build_strategy_calendar(
        base_population=account_frame,
        selected_recommendations=final_selection,
        policy=policy,
    )

    top_choice = final_selection.iloc[[0]].copy()
    explanation = explain_prediction(model=model, candidate_row=top_choice, top_k=10)

    output_dir = config.root_dir / "outputs" / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    day_scores_path = output_dir / f"{account_id}_day_scores.csv"
    final_path = output_dir / f"{account_id}_final_recommendation.csv"
    calendar_path = output_dir / f"{account_id}_strategy_calendar.csv"

    best_per_day.to_csv(day_scores_path, index=False)
    final_selection.to_csv(final_path, index=False)
    strategy_calendar.to_csv(calendar_path, index=False)
    final_payload = final_selection.copy()
    if "emi_date" in final_payload.columns:
        final_payload["emi_date"] = pd.to_datetime(final_payload["emi_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return {
        "account_id": account_id,
        "day_scores_path": str(day_scores_path),
        "final_recommendation_path": str(final_path),
        "strategy_calendar_path": str(calendar_path),
        "final_recommendation": final_payload.to_dict(orient="records"),
        "top_choice_explanation": explanation,
    }
