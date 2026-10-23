from __future__ import annotations

import json
import os
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


def load_default_config(config_path: Path | str | None = None) -> dict[str, object]:
    """Loads configuration JSON from the specified path, env var, or repo config/default_config.json."""
    if config_path is None:
        env_path = os.getenv("CONFIG_FILE") or os.getenv("DEFAULT_CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
        else:
            repo_root = Path(__file__).resolve().parent.parent
            config_path = repo_root / "config" / "default_config.json"
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            cleaned = "\n".join(
                line for line in f
                if not line.strip().startswith("//") and not line.strip().startswith("#")
            )
            return json.loads(cleaned)


def offset_to_day_label(offset: int | str) -> str:
    """Converts an integer or string offset to a day label, e.g. -5 -> 'D-5', 0 -> 'D', 2 -> 'D+2'."""
    if isinstance(offset, str):
        val = offset.strip()
        if val.startswith("D"):
            return val
        try:
            offset_int = int(val)
        except ValueError:
            return val
    else:
        offset_int = int(offset)
    if offset_int < 0:
        return f"D{offset_int}"
    elif offset_int == 0:
        return "D"
    else:
        return f"D+{offset_int}"


def day_label_to_offset(day: str) -> int:
    """Converts a day label back to an integer offset for sorting, e.g. 'D-5' -> -5, 'D' -> 0, 'D+2' -> 2."""
    day_str = str(day).strip()
    if day_str == "D":
        return 0
    if day_str.startswith("D-"):
        return -int(day_str[2:])
    if day_str.startswith("D+"):
        return int(day_str[2:])
    try:
        return int(day_str)
    except ValueError:
        return 0


def parse_day_offset(value: object) -> int | None:
    """Parses a day offset from int, float, or string (e.g. -5, 5, 'D-5', 'D+5', '+5', '5', '-5').
    Returns None if value is empty, None, or cannot be parsed as a valid integer offset.
    """
    if value is None:
        return None
    val_str = str(value).strip()
    if not val_str:
        return None
    if val_str.upper() == "D":
        return 0
    if val_str.upper().startswith("D-"):
        try:
            return -int(val_str[2:].strip())
        except ValueError:
            return None
    if val_str.upper().startswith("D+"):
        try:
            return int(val_str[2:].strip())
        except ValueError:
            return None
    try:
        return int(float(val_str))
    except (ValueError, TypeError):
        return None


def offsets_from_window_range(start: object, end: object) -> list[int] | None:
    """Computes a sorted list of non-zero integer day offsets between start and end (inclusive).
    Returns None if either start or end is invalid.
    """
    start_offset = parse_day_offset(start)
    end_offset = parse_day_offset(end)
    if start_offset is None or end_offset is None:
        return None
    if start_offset > end_offset:
        start_offset, end_offset = end_offset, start_offset
    return [d for d in range(start_offset, end_offset + 1) if d != 0]


def offsets_to_day_columns(offsets: list[int | str]) -> list[str]:
    """Converts a list of day offsets into chronologically sorted day labels (D-x..D-1, D, D+1..D+x)."""
    seen = set()
    labels: list[str] = []
    for off in offsets:
        lbl = offset_to_day_label(off)
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)
    return sorted(labels, key=day_label_to_offset)


def parse_strategy_parts(strategy: str) -> tuple[str, str, str] | None:
    """Parses a strategy token like 'SMS-11AM-ENGLISH' into (channel, hour, language)."""
    parts = str(strategy or "").strip().split("-")
    if len(parts) != 3:
        return None
    return parts[0].strip().upper(), parts[1].strip().upper(), parts[2].strip().upper()


def readable_strategy(strategy: str) -> str:
    """Formats a strategy token or multi-mode bundle into a business-readable string.
    Example: 'SMS-11AM-ENGLISH' -> 'SMS at 11AM in English'
    Example: 'SMS-11AM-ENGLISH|WH-12PM-ENGLISH' -> 'SMS at 11AM in English, WhatsApp at 12PM in English'
    """
    if not strategy or str(strategy).strip() in ("", "-"):
        return "no communication"
    tokens = [s.strip() for s in str(strategy).split("|") if s.strip() and s.strip() != "-"]
    if not tokens:
        return "no communication"
    formatted = []
    for token in tokens:
        parts = parse_strategy_parts(token)
        if not parts:
            formatted.append(token)
            continue
        channel, hour, language = parts
        channel_name = {"SMS": "SMS", "WH": "WhatsApp", "VOICE": "voice call", "VOICE_BOT": "voice bot"}.get(channel, channel)
        language_name = "Regional language" if language == "REGIONAL" else language.title()
        formatted.append(f"{channel_name} at {hour} in {language_name}")
    return ", ".join(formatted)


def extract_template_from_entry(entry: object, language: str = "") -> str | None:
    """Extracts a template name string from a config entry (which can be a string, dict, etc.)."""
    if entry is None or pd.isna(entry):
        return None
    if isinstance(entry, str):
        val = entry.strip()
        return val or None
    if isinstance(entry, dict):
        lang_u = str(language).strip().upper() if language else ""
        # 1. Match specific language key if present
        if lang_u and lang_u in entry:
            res = extract_template_from_entry(entry[lang_u], language=language)
            if res:
                return res
        # 2. Match standard language/default fallbacks
        for fallback_k in ("ENGLISH", "DEFAULT", "default", "standard"):
            if fallback_k in entry:
                res = extract_template_from_entry(entry[fallback_k], language=language)
                if res:
                    return res
        # 3. Match template name attribute keys
        for key in ("template_name", "name", "template", "iteration"):
            if key in entry and entry[key]:
                res = extract_template_from_entry(entry[key], language=language)
                if res:
                    return res
        # 4. If ID is provided as fallback
        for id_key in ("id", "rule_id", "digital_rule_id", "template_id"):
            if id_key in entry and entry[id_key]:
                val = str(entry[id_key]).strip()
                if val:
                    return val
    return None


def get_configured_default_template(
    default_templates: dict | list | None,
    *,
    due_type: str,
    mode: str,
    language: str = "",
) -> str | None:
    """
    Looks up a default template from the configuration based on due_type, mode, and language.
    Supports nested dicts, flat dicts, or lists of template definitions.
    """
    if not default_templates:
        return None

    # Canonical due_type aliases
    due_type_u = str(due_type).strip().upper()
    due_type_keys = [due_type_u]
    if due_type_u in ("PRE", "PREDUE"):
        due_type_keys = ["PREDUE", "PRE"]
    elif due_type_u in ("POST", "POSTDUE"):
        due_type_keys = ["POSTDUE", "POST"]
    elif due_type_u in ("DUE", "DUEDATE"):
        due_type_keys = ["DUEDATE", "DUE"]

    # Canonical mode aliases
    mode_u = str(mode).strip().upper()
    mode_keys = [mode_u]
    if mode_u in ("WHATSAPP", "WA", "WH"):
        mode_keys = ["WHATSAPP", "WA", "WH"]
    elif mode_u in ("SMS", "S"):
        mode_keys = ["SMS", "S"]
    elif mode_u in ("VOICE", "IVR", "V"):
        mode_keys = ["VOICE", "IVR", "VOICE_BOT", "V"]
    elif mode_u in ("VOICE_BOT", "BOT", "B"):
        mode_keys = ["VOICE_BOT", "VOICE", "IVR"]

    if isinstance(default_templates, dict):
        # Case A: nested by due_type -> mode -> ...
        for dt_key in due_type_keys:
            if dt_key in default_templates and isinstance(default_templates[dt_key], dict):
                due_dict = default_templates[dt_key]
                for m_key in mode_keys:
                    if m_key in due_dict:
                        res = extract_template_from_entry(due_dict[m_key], language=language)
                        if res:
                            return res
                # Check for DEFAULT in due_dict
                for fallback_k in ("DEFAULT", "default"):
                    if fallback_k in due_dict:
                        res = extract_template_from_entry(due_dict[fallback_k], language=language)
                        if res:
                            return res

        # Case B: dict directly keyed by mode without due_type level
        for m_key in mode_keys:
            if m_key in default_templates:
                res = extract_template_from_entry(default_templates[m_key], language=language)
                if res:
                    return res

    elif isinstance(default_templates, list):
        lang_u = str(language).strip().upper() if language else ""
        # 1. Exact match on due_type, mode, and language
        for item in default_templates:
            if not isinstance(item, dict):
                continue
            item_dt = str(item.get("due_type", "")).strip().upper()
            item_mode = str(item.get("mode", "")).strip().upper()
            item_lang = str(item.get("language", "")).strip().upper()
            if item_dt in due_type_keys and item_mode in mode_keys and item_lang == lang_u:
                res = extract_template_from_entry(item, language=language)
                if res:
                    return res
        # 2. Match on due_type and mode without specific language requirement
        for item in default_templates:
            if not isinstance(item, dict):
                continue
            item_dt = str(item.get("due_type", "")).strip().upper()
            item_mode = str(item.get("mode", "")).strip().upper()
            item_lang = str(item.get("language", "")).strip().upper()
            if item_dt in due_type_keys and item_mode in mode_keys and item_lang in ("", "DEFAULT", "ENGLISH"):
                res = extract_template_from_entry(item, language=language)
                if res:
                    return res

    return None



def get_first_valid_communication(val: object) -> str:
    """Returns the first non-empty, valid strategy if multiple are separated by '|'."""
    val_str = str(val).strip() if val is not None and not pd.isna(val) else "-"
    if "|" not in val_str:
        return val_str if val_str not in ("", "-", "None", "nan") else "-"
    for p in val_str.split("|"):
        p = p.strip()
        if p not in ("", "-", "None", "nan"):
            return p
    return "-"


def enforce_single_communication_per_day(
    df: pd.DataFrame,
    day_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Enforces at most one communication per day across all specified day columns
    by selecting the top-ranked (first valid) recommendation and discarding pipe-separated alternatives.
    """
    df = df.copy()
    cols_to_check = [c for c in (day_columns or DAY_COLUMNS) if c in df.columns]
    for day in cols_to_check:
        df[day] = df[day].fillna("-").astype(str).apply(get_first_valid_communication)
    return df


TARGET_POSTDUE_MODES = ["SMS", "WHATSAPP", "VOICE"]
DEFAULT_MODE_FALLBACKS = {
    "SMS": "SMS-11AM-ENGLISH",
    "WHATSAPP": "WH-12PM-ENGLISH",
    "VOICE": "VOICE-4PM-ENGLISH",
}


def extract_canonical_mode_from_label(label: str) -> str | None:
    """Extracts canonical mode (SMS, WHATSAPP, VOICE) from a strategy label."""
    if not label or label in ("-", "None", "nan") or pd.isna(label):
        return None
    channel = str(label).split("-")[0].strip().upper()
    if channel == "SMS":
        return "SMS"
    if channel in ("WH", "WHATSAPP"):
        return "WHATSAPP"
    if channel in ("VOICE", "VOICE_BOT", "IVR"):
        return "VOICE"
    return None


def extract_hour_from_label(label: str) -> str | None:
    """Extracts hour part from a strategy label like 'SMS-11AM-ENGLISH' -> '11AM'."""
    if not label or label in ("-", "None", "nan") or pd.isna(label):
        return None
    parts = str(label).split("-")
    if len(parts) >= 3:
        return "-".join(parts[1:-1]).strip()
    return None


def enforce_single_communication_per_mode_per_day(
    df: pd.DataFrame,
    day_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Enforces at most one communication per distinct mode (SMS, WHATSAPP, VOICE) per day,
    preserving distinct mode recommendations and discarding duplicate modes.
    """
    df = df.copy()
    cols_to_check = [c for c in (day_columns or POSTDUE_DAY_COLUMNS) if c in df.columns]
    for day in cols_to_check:
        new_vals = []
        for val in df[day]:
            if pd.isna(val) or val in ("", "-", "None", "nan"):
                new_vals.append("-")
                continue
            val_str = str(val).strip()
            parts = [p.strip() for p in val_str.split("|") if p.strip() not in ("", "-", "None", "nan")]
            if not parts:
                new_vals.append("-")
                continue
            seen_modes: set[str] = set()
            deduped: list[str] = []
            for p in parts:
                mode = extract_canonical_mode_from_label(p)
                if mode and mode not in seen_modes:
                    seen_modes.add(mode)
                    deduped.append(p)
                elif not mode and not deduped:
                    deduped.append(p)
            new_vals.append("|".join(deduped) if deduped else "-")
        df[day] = new_vals
    return df


def apply_postdue_risk_rules(
    df: pd.DataFrame,
    postdue_days: list[str] | None = None,
    allow_multimode: bool = False,
) -> pd.DataFrame:
    """
    Enforces post-due communication rules across post-due days (D+1..D+20).
    When allow_multimode is True, allows up to 1 communication per distinct mode (SMS, WHATSAPP, VOICE) per day.
    When allow_multimode is False, caps each day to the single first valid communication.
    """
    cols = [d for d in (postdue_days or POSTDUE_DAY_COLUMNS) if d in df.columns]
    if allow_multimode:
        return enforce_single_communication_per_mode_per_day(df, cols)
    return enforce_single_communication_per_day(df, cols)


def apply_predue_risk_rules(
    df: pd.DataFrame,
    default_strategy: str | None = DEFAULT_PREDUE_STRATEGY,
    predue_days: list[str] | None = None,
) -> pd.DataFrame:
    """
    Enforces mandatory baseline communication rule and risk-based pre-due touchpoint limits:
    - If D-5 is configured, it is mandatory for 100% of base population (defaults to SMS-11AM-ENGLISH if blank and default_strategy is set).
    - Ensures single communication per day (selects 1st option if multiple given).
    - LOW Risk: Exactly 2 communications across configured pre-due days (1 per day).
    - MEDIUM Risk: Exactly 3 communications across configured pre-due days (1 per day).
    - HIGH Risk: Exactly 4 communications across configured pre-due days (1 per day).
    """
    df = df.copy()
    risk_col = "SOURCE_RISK" if "SOURCE_RISK" in df.columns else ("RISK" if "RISK" in df.columns else None)
    active_predue_cols = [
        d for d in (predue_days or PREDUE_DAY_COLUMNS)
        if d in df.columns
    ]
    if not active_predue_cols:
        return df

    has_d5 = "D-5" in active_predue_cols
    other_predue_days = [d for d in active_predue_cols if d != "D-5"]

    # 1. Enforce single communication per day across active pre-due days
    df = enforce_single_communication_per_day(df, active_predue_cols)

    for idx, row in df.iterrows():
        risk_raw = str(row[risk_col]).upper().strip() if risk_col and pd.notna(row[risk_col]) else "LOW"
        limits = PREDUE_LIMITS_BY_RISK.get(risk_raw, PREDUE_LIMITS_BY_RISK["LOW"])
        # If fewer pre-due days are configured than target_exact, cap target at total available
        target_exact = min(limits["target_exact"], len(active_predue_cols))

        # 2. Mandatory D-5 Full Base Communication (if D-5 is configured)
        if has_d5:
            d5_val = str(df.at[idx, "D-5"]).strip()
            if d5_val in ("", "-", "None", "nan") and default_strategy:
                df.at[idx, "D-5"] = default_strategy
            base_count = 1 if str(df.at[idx, "D-5"]).strip() not in ("", "-", "None", "nan") else 0
        else:
            base_count = 0

        # 3. Identify active days among other pre-due days
        active_other = [
            day for day in other_predue_days
            if str(df.at[idx, day]).strip() not in ("", "-", "None", "nan")
        ]

        current_active = base_count + len(active_other)

        # 4. Trim extra pre-due days if count exceeds target_exact
        if current_active > target_exact:
            allowed_other_count = max(0, target_exact - base_count)
            allowed_other = set(active_other[:allowed_other_count])
            for day in other_predue_days:
                if day not in allowed_other:
                    df.at[idx, day] = "-"

        # 5. Fill in missing pre-due days if count is below target_exact (only when default_strategy is provided)
        elif current_active < target_exact and default_strategy:
            needed = target_exact - current_active
            priority_fill = sorted(other_predue_days, key=day_label_to_offset, reverse=True)
            filled = 0
            for day in priority_fill:
                if filled >= needed:
                    break
                val = str(df.at[idx, day]).strip()
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


def snap_hour_to_window(hour: int, send_hour_window: dict[str, int]) -> int:
    """Clamps hour to [start_hour, end_hour] and snaps to the nearest candidate hour by step_hours."""
    start_hour = int(send_hour_window["start_hour"])
    end_hour = int(send_hour_window["end_hour"])
    if hour <= start_hour:
        return start_hour
    if hour >= end_hour:
        return end_hour
    hours = candidate_hours(send_hour_window)
    return min(hours, key=lambda h: (abs(h - hour), h))


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

    available_day_cols = [col for col in DAY_COLUMNS if col in schedule.columns]
    available_weight_cols = [col for col in DAY_WEIGHT_COLUMNS if col in schedule.columns]
    schedule_cols = [
        key_column,
        "TARGET_MONTH",
        "TARGET_MONTH_PERIOD",
        "TARGET_RISK",
        *available_day_cols,
        *available_weight_cols,
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
    mandatory_communication: bool = False,
    k_override: int | None = None,
) -> pd.Series:
    probabilities = model.predict_proba(X)
    classes = [str(c) for c in encoder.classes_]
    output: list[str] = []
    for row_probs, risk in zip(probabilities, risks.fillna("LOW").astype(str), strict=False):
        effective_k = k_override if k_override is not None else RISK_TOP_K.get(risk.upper(), 1)
        sorted_indices = row_probs.argsort()[::-1]
        if mandatory_communication:
            non_dash_indices = [idx for idx in sorted_indices if classes[idx] not in ("", "-", "None", "nan")]
            dash_indices = [idx for idx in sorted_indices if classes[idx] in ("", "-", "None", "nan")]
            chosen_indices = (non_dash_indices + dash_indices)[:effective_k]
        else:
            chosen_indices = list(sorted_indices[:effective_k])
        labels = [classes[index] for index in chosen_indices]
        output.append("|".join(labels))
    return pd.Series(output, index=X.index)


def predict_distinct_modes_by_risk(
    model,
    encoder,
    X: pd.DataFrame,
    risks: pd.Series | None = None,
    target_modes: list[str] | None = None,
    default_fallbacks: dict[str, str] | None = None,
) -> pd.Series:
    """
    Selects the highest probability recommendation for each distinct mode (SMS, WHATSAPP, VOICE)
    for each account from model probabilities, avoiding same-hour collision across the selected modes.
    Language can be repeated across modes.
    """
    modes_to_use = target_modes or TARGET_POSTDUE_MODES
    fallbacks = default_fallbacks or DEFAULT_MODE_FALLBACKS
    probabilities = model.predict_proba(X)
    classes = [str(c) for c in encoder.classes_]

    # Pre-index class indices by canonical mode
    mode_class_indices: dict[str, list[int]] = {m: [] for m in modes_to_use}
    for idx, label in enumerate(classes):
        m = extract_canonical_mode_from_label(label)
        if m in mode_class_indices:
            mode_class_indices[m].append(idx)

    # If the day has no candidates across any target mode (e.g. only '-' in model vocabulary), return '-'
    if not any(mode_class_indices.values()):
        return pd.Series(["-"] * len(X), index=X.index)

    output: list[str] = []
    for row_probs in probabilities:
        # Collect candidate classes per mode sorted by probability descending
        mode_candidates: dict[str, list[tuple[str, float, str]]] = {}
        for mode in modes_to_use:
            cand_list: list[tuple[str, float, str]] = []
            for c_idx in mode_class_indices.get(mode, []):
                lbl = classes[c_idx]
                prob = float(row_probs[c_idx])
                hr = extract_hour_from_label(lbl) or ""
                cand_list.append((lbl, prob, hr))
            cand_list.sort(key=lambda item: item[1], reverse=True)
            mode_candidates[mode] = cand_list

        # Order modes so that the mode with highest peak probability chooses its optimal hour first
        mode_order = sorted(
            modes_to_use,
            key=lambda m: mode_candidates[m][0][1] if mode_candidates[m] else -1.0,
            reverse=True,
        )

        chosen_by_mode: dict[str, str] = {}
        used_hours: set[str] = set()

        for mode in mode_order:
            candidates = mode_candidates[mode]
            if not candidates:
                fb = fallbacks.get(mode, f"{mode}-12PM-ENGLISH")
                chosen_by_mode[mode] = fb
                fb_hr = extract_hour_from_label(fb)
                if fb_hr:
                    used_hours.add(fb_hr)
                continue

            # Pick highest probability candidate whose hour has not been taken yet
            selected = None
            for lbl, prob, hr in candidates:
                if hr and hr not in used_hours:
                    selected = (lbl, hr)
                    break

            # If all candidate hours were used, take top candidate
            if selected is None:
                selected = (candidates[0][0], candidates[0][2])

            chosen_by_mode[mode] = selected[0]
            if selected[1]:
                used_hours.add(selected[1])

        day_recommendations = [chosen_by_mode[m] for m in modes_to_use if m in chosen_by_mode]
        output.append("|".join(day_recommendations) if day_recommendations else "-")

    return pd.Series(output, index=X.index)
