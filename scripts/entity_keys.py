from __future__ import annotations

import os

import pandas as pd


ENTITY_KEY_COLUMN = "ENTITY_KEY"
DEFAULT_PARTY_ID_COLUMN = "party_id"
DEFAULT_LOAN_ID_COLUMN = "apac_card_number"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def strategy_use_party_id() -> bool:
    return env_flag("STRATEGY_USE_PARTY_ID", default=False)


def normalize_identifier(series: pd.Series, *, default: str = "") -> pd.Series:
    return series.fillna(default).astype(str).str.strip().replace({"": default})


def annotate_entity_key(
    df: pd.DataFrame,
    *,
    loan_column: str = DEFAULT_LOAN_ID_COLUMN,
    party_column: str = DEFAULT_PARTY_ID_COLUMN,
    output_column: str = ENTITY_KEY_COLUMN,
) -> pd.DataFrame:
    result = df.copy()
    loan_values = normalize_identifier(result.get(loan_column, pd.Series("", index=result.index)), default="")
    result[loan_column] = loan_values
    use_party = strategy_use_party_id()
    if use_party:
        if party_column not in result.columns:
            raise ValueError(
                f"STRATEGY_USE_PARTY_ID is enabled but required column {party_column!r} is missing."
            )
        party_values = normalize_identifier(result[party_column], default="")
        if party_values.eq("").all():
            raise ValueError(
                f"STRATEGY_USE_PARTY_ID is enabled but column {party_column!r} is blank for all rows."
            )
        result[party_column] = party_values
        result[output_column] = party_values
    else:
        if party_column in result.columns:
            result[party_column] = normalize_identifier(result[party_column], default="")
        result[output_column] = loan_values
    return result


def entity_key_column_name() -> str:
    return ENTITY_KEY_COLUMN


def resolve_active_entity_column(df: pd.DataFrame) -> str:
    return ENTITY_KEY_COLUMN if ENTITY_KEY_COLUMN in df.columns else "APAC_CARD_NUMBER"
