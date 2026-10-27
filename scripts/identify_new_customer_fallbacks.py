from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from pipeline_common import (
    DAY_COLUMNS,
    POSTDUE_DAY_COLUMNS,
    PREDUE_DAY_COLUMNS,
    TARGET_POSTDUE_MODES,
    apply_predue_risk_rules,
    extract_canonical_mode_from_label,
    extract_hour_from_label,
    load_default_config,
    offsets_to_day_columns,
    parse_day_offset,
    readable_strategy,
)
from env_utils import env_flag, load_dotenv
from project_paths import CASE_DATA_DIR, COMMUNICATION_DATA_DIR, REPO_ROOT, SCHEDULE_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description=(
            "Identify APACs present in current cases but absent from communication "
            "history, then prepare fallback strategies."
        )
    )
    parser.add_argument(
        "--cases-file",
        default=str(CASE_DATA_DIR / "digital_cases_SEP2026.csv"),
        help="Current prediction cases CSV.",
    )
    parser.add_argument(
        "--communication-files",
        nargs="*",
        default=[],
        help="Communication history CSV files. Defaults to all data/communication/*.csv.",
    )
    parser.add_argument(
        "--schedule-file",
        default=str(SCHEDULE_DATA_DIR / "strategy_schedule_dataset_train.csv"),
        help="Historical schedule target CSV used to derive average LOW-risk strategy.",
    )
    parser.add_argument(
        "--output-file",
        default=str(CASE_DATA_DIR / "new_customer_low_risk_fallbacks.csv"),
        help="Output CSV for new-customer fallback recommendations.",
    )
    parser.add_argument(
        "--prediction-file",
        default="",
        help="Optional prediction CSV to update in-place for new APACs.",
    )
    parser.add_argument(
        "--config-file",
        default=os.getenv("CONFIG_FILE", str(REPO_ROOT / "config" / "default_config.json")),
        help="Path to JSON configuration file containing allowed_day_offsets.",
    )
    parser.add_argument(
        "--allowed-day-offsets",
        nargs="*",
        default=None,
        help="Allowed day offsets for prediction window (e.g. -5 -4 -3 -2 -1 1 2 3 4 5). Overrides config-file.",
    )
    parser.add_argument(
        "--multimode-postdue",
        action=argparse.BooleanOptionalAction,
        default=env_flag("MULTIMODE_POSTDUE", env_flag("STRATEGY_MULTIMODE_POSTDUE", default=True)),
        help="Enable multi-mode post-due fallback (SMS, WhatsApp, Voice) for post-due days [default: True, or MULTIMODE_POSTDUE env].",
    )
    return parser.parse_args()


def find_apac_column(columns: list[str] | pd.Index) -> str | None:
    cols_map = {str(c).replace("\ufeff", "").strip().lower(): c for c in columns}
    for candidate in ["apac_card_number", "apaccardnumber", "loan_number", "account_number", "contract_number"]:
        if candidate in cols_map:
            return str(cols_map[candidate])
    return None


def normalize_apac(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def load_case_apacs(cases_file: Path) -> pd.DataFrame:
    cases = pd.read_csv(cases_file, dtype=str, encoding="utf-8-sig")
    col = find_apac_column(cases.columns)
    if not col:
        raise ValueError("Cases file must contain apac_card_number, APAC_CARD_NUMBER, or Loan_number.")

    output = pd.DataFrame({"apacCardNumber": normalize_apac(cases[col])})
    if "risk" in cases.columns:
        output["originalRisk"] = cases["risk"].fillna("").astype(str).str.upper().str.strip()
    elif "RISK" in cases.columns:
        output["originalRisk"] = cases["RISK"].fillna("").astype(str).str.upper().str.strip()
    else:
        output["originalRisk"] = ""
    if "vertical" in cases.columns:
        output["vertical"] = cases["vertical"].fillna("").astype(str).str.upper().str.strip()
    elif "VERTICAL" in cases.columns:
        output["vertical"] = cases["VERTICAL"].fillna("").astype(str).str.upper().str.strip()
    else:
        output["vertical"] = ""
    return output[output["apacCardNumber"].ne("")].drop_duplicates("apacCardNumber")


def load_history_apacs(communication_files: list[Path]) -> set[str]:
    history: set[str] = set()
    for csv_file in communication_files:
        if not csv_file.exists():
            continue
        try:
            header = pd.read_csv(csv_file, nrows=0, encoding="utf-8-sig").columns
        except Exception:
            continue
        col = find_apac_column(header)
        if not col:
            continue
        try:
            for chunk in pd.read_csv(csv_file, usecols=[col], dtype=str, chunksize=200_000, encoding="utf-8-sig"):
                normed = normalize_apac(chunk[col])
                history.update(normed)
                history.update(normed.str.lstrip("0"))
        except Exception:
            continue
    history.discard("")
    return history


def normalize_fallback_strategy(strategy: str) -> str:
    if not strategy or strategy == "-" or pd.isna(strategy):
        return "-"
    parts = [p.strip() for p in str(strategy).split("-") if p.strip()]
    if len(parts) < 3:
        return strategy
    channel = parts[0]
    language = parts[-1]
    raw_hour = "-".join(parts[1:-1]).upper()

    slot_map = {
        "8-11": "9AM",
        "12-3": "12PM",
        "4-7": "4PM",
        "MORNING": "9AM",
        "AFTERNOON": "12PM",
        "EVENING": "4PM",
    }

    if raw_hour in slot_map:
        hour_str = slot_map[raw_hour]
    elif raw_hour.isdigit():
        h = int(raw_hour)
        if h < 9:
            h = 9
        elif h > 18:
            h = 18
        hour_str = f"{h}AM" if h < 12 else ("12PM" if h == 12 else f"{h-12}PM")
    else:
        hour_str = raw_hour

    return f"{channel}-{hour_str}-{language}"


def most_common_strategy(values: pd.Series) -> str:
    normalized = values.fillna("").astype(str).str.strip()
    normalized = normalized.map(normalize_fallback_strategy)
    normalized = normalized[normalized.ne("") & normalized.ne("-")]
    if normalized.empty:
        return "-"
    return str(normalized.value_counts().sort_values(ascending=False).index[0])


DEFAULT_POSTDUE_FALLBACK_MODES: dict[str, str] = {
    "SMS": "SMS-10AM-ENGLISH",
    "WHATSAPP": "WH-1PM-ENGLISH",
    "VOICE": "VOICE-4PM-ENGLISH",
}


def build_postdue_multimode_strategy(
    values: pd.Series | None,
    target_modes: list[str] | None = None,
    default_fallbacks: dict[str, str] | None = None,
) -> str:
    """
    Constructs a multi-mode fallback strategy for a post-due day
    (e.g. SMS-10AM-ENGLISH|WH-1PM-ENGLISH|VOICE-4PM-ENGLISH).
    Extracts most common strategies per mode from historical schedule values if present,
    harmonizes language across modes, resolves hour collisions, and fills missing modes with
    staggered default fallbacks.
    """
    modes_to_use = target_modes or TARGET_POSTDUE_MODES
    fallbacks = default_fallbacks or DEFAULT_POSTDUE_FALLBACK_MODES

    mode_token_counts: dict[str, dict[str, int]] = {m: {} for m in modes_to_use}
    detected_languages: list[str] = []

    if values is not None and not values.empty:
        for raw_val in values.dropna():
            val_str = str(raw_val).strip()
            if not val_str or val_str in ("-", "None", "nan"):
                continue
            for token in val_str.split("|"):
                token_clean = normalize_fallback_strategy(token.strip())
                if not token_clean or token_clean in ("-", "None", "nan"):
                    continue
                mode = extract_canonical_mode_from_label(token_clean)
                if mode in mode_token_counts:
                    mode_token_counts[mode][token_clean] = mode_token_counts[mode].get(token_clean, 0) + 1
                parts = token_clean.split("-")
                if len(parts) >= 3:
                    lang = parts[-1].strip().upper()
                    if lang and lang not in ("-", "NONE", "NAN"):
                        detected_languages.append(lang)

    dominant_language = "ENGLISH"
    if detected_languages:
        dominant_language = str(pd.Series(detected_languages).value_counts().index[0])

    default_hour_slots = {"SMS": "10AM", "WHATSAPP": "1PM", "VOICE": "4PM"}
    allowed_fallback_hours = ["10AM", "1PM", "4PM", "11AM", "12PM", "2PM", "3PM", "5PM"]

    used_hours: set[str] = set()
    chosen_strategies: list[str] = []

    for mode in modes_to_use:
        counts = mode_token_counts.get(mode, {})
        chosen_token = None
        if counts:
            sorted_candidates = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            for cand, _ in sorted_candidates:
                cand_hr = extract_hour_from_label(cand)
                if cand_hr and cand_hr not in used_hours:
                    chosen_token = cand
                    used_hours.add(cand_hr)
                    break

        if chosen_token is None:
            fallback_template = fallbacks.get(mode, f"{mode}-{default_hour_slots.get(mode, '12PM')}-ENGLISH")
            parts = fallback_template.split("-")
            ch = parts[0]
            pref_hr = parts[1] if len(parts) >= 2 else default_hour_slots.get(mode, "12PM")
            lang = dominant_language if dominant_language else (parts[-1] if len(parts) >= 3 else "ENGLISH")

            if pref_hr not in used_hours:
                chosen_hr = pref_hr
            else:
                chosen_hr = next((h for h in allowed_fallback_hours if h not in used_hours), pref_hr)

            used_hours.add(chosen_hr)
            chosen_token = f"{ch}-{chosen_hr}-{lang}"

        chosen_strategies.append(chosen_token)

    return "|".join(chosen_strategies) if chosen_strategies else "-"


def build_average_strategy_by_risk(
    schedule_file: Path,
    multimode_postdue: bool = True,
    apply_predue_rules: bool = True,
    allowed_predue_days: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    schedule = pd.read_csv(schedule_file, dtype=str)
    if "RISK" not in schedule.columns:
        average_by_risk = {
            "LOW": {
                day: (
                    build_postdue_multimode_strategy(schedule[day])
                    if multimode_postdue and day in POSTDUE_DAY_COLUMNS
                    else most_common_strategy(schedule[day])
                )
                if day in schedule.columns
                else (
                    build_postdue_multimode_strategy(None)
                    if multimode_postdue and day in POSTDUE_DAY_COLUMNS
                    else "-"
                )
                for day in DAY_COLUMNS
            }
        }
    else:
        schedule["_RISK"] = schedule["RISK"].fillna("").astype(str).str.upper().str.strip()
        average_by_risk = {}
        for risk, risk_df in schedule.groupby("_RISK", sort=False):
            if not risk:
                continue
            average_by_risk[risk] = {
                day: (
                    build_postdue_multimode_strategy(risk_df[day])
                    if multimode_postdue and day in POSTDUE_DAY_COLUMNS
                    else most_common_strategy(risk_df[day])
                )
                if day in risk_df.columns
                else (
                    build_postdue_multimode_strategy(None)
                    if multimode_postdue and day in POSTDUE_DAY_COLUMNS
                    else "-"
                )
                for day in DAY_COLUMNS
            }

    for r in ("LOW", "MEDIUM", "HIGH"):
        if r not in average_by_risk:
            average_by_risk[r] = {
                day: (
                    build_postdue_multimode_strategy(schedule[day] if day in schedule.columns else None)
                    if multimode_postdue and day in POSTDUE_DAY_COLUMNS
                    else (most_common_strategy(schedule[day]) if day in schedule.columns else "-")
                )
                for day in DAY_COLUMNS
            }

    if apply_predue_rules and average_by_risk:
        risk_rows = []
        for risk, day_dict in average_by_risk.items():
            risk_rows.append({"RISK": risk, **day_dict})
        risk_df = pd.DataFrame(risk_rows)
        risk_df = apply_predue_risk_rules(risk_df, predue_days=allowed_predue_days)
        for _, row in risk_df.iterrows():
            risk_val = str(row["RISK"]).upper().strip()
            if risk_val in average_by_risk:
                for col in risk_df.columns:
                    if col in DAY_COLUMNS:
                        average_by_risk[risk_val][col] = str(row[col])

    return average_by_risk


def build_fallback_rows(
    new_cases: pd.DataFrame,
    fallback_by_risk: dict[str, dict[str, str]],
    allowed_day_labels: set[str] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for case in new_cases.to_dict("records"):
        risk = str(case.get("originalRisk", "")).strip().upper() or "LOW"
        lookup_risk = "LOW" if risk in ("UNKNOWN", "NULL", "NONE", "NAN", "") else risk
        fallback_by_day = fallback_by_risk.get(lookup_risk) or fallback_by_risk.get("LOW", {})
        display_risk = "LOW" if risk in ("UNKNOWN", "NULL", "NONE", "NAN", "") else risk
        for day in DAY_COLUMNS:
            strategy = fallback_by_day.get(day, "-") if (allowed_day_labels is None or day in allowed_day_labels) else "-"
            if strategy and strategy != "-":
                reason_text = (
                    f"Cold-start fallback: {readable_strategy(strategy)} is recommended based on "
                    f"{display_risk}-risk historical average, as no communication history was found for this customer."
                )
            else:
                reason_text = (
                    f"No communication is recommended on {day} based on {display_risk}-risk cohort guidelines."
                )
            records.append(
                {
                    "apacCardNumber": case["apacCardNumber"],
                    "originalRisk": case.get("originalRisk", ""),
                    "fallbackRiskUsed": lookup_risk,
                    "vertical": case.get("vertical", ""),
                    "day": day,
                    "recommendedStrategy": strategy,
                    "isActionable": "false" if strategy == "-" else "true",
                    "reason": reason_text,
                }
            )
    return pd.DataFrame(records)


def apply_fallbacks_to_prediction_file(
    prediction_file: Path,
    fallback_rows: pd.DataFrame,
    allowed_day_labels: set[str] | None = None,
) -> int:
    if not prediction_file.exists() or fallback_rows.empty:
        return 0
    predictions = pd.read_csv(prediction_file, dtype=str).fillna("")
    if "Loan_number" not in predictions.columns:
        raise ValueError("Prediction file must contain Loan_number column.")

    fallback_map: dict[tuple[str, str], str] = {}
    day_reason_map: dict[tuple[str, str], str] = {}
    risk_map: dict[str, str] = {}
    vertical_map: dict[str, str] = {}

    for row in fallback_rows.itertuples(index=False):
        raw_card = str(row.apacCardNumber).strip()
        clean_card = raw_card.replace(".0", "").lstrip("0")
        strat = str(row.recommendedStrategy).strip() or "-"
        risk_val = str(row.fallbackRiskUsed).strip().upper() or "LOW"
        reas_val = str(row.reason).strip()
        vert_val = str(row.vertical).strip().upper()
        day_val = str(row.day).strip()

        for key_c in {raw_card, clean_card, raw_card.zfill(18)}:
            if not key_c:
                continue
            fallback_map[(key_c, day_val)] = strat
            day_reason_map[(key_c, day_val)] = reas_val
            risk_map[key_c] = risk_val
            vertical_map[key_c] = vert_val

    if "IS_NEW_CUSTOMER" not in predictions.columns:
        predictions["IS_NEW_CUSTOMER"] = "False"

    updated = 0
    existing_loans = set(predictions["Loan_number"].astype(str).str.strip())
    existing_clean_loans = {l.replace(".0", "").lstrip("0") for l in existing_loans}

    for idx, row in predictions.iterrows():
        loan = str(row.get("Loan_number", "")).strip()
        clean_loan = loan.replace(".0", "").lstrip("0")
        matched_key = loan if loan in risk_map else (clean_loan if clean_loan in risk_map else None)

        if matched_key is None:
            continue

        if "SOURCE_RISK" in predictions.columns and (pd.isna(predictions.at[idx, "SOURCE_RISK"]) or str(predictions.at[idx, "SOURCE_RISK"]).strip() in ("", "-", "None", "nan")):
            predictions.at[idx, "SOURCE_RISK"] = risk_map[matched_key]

        # If the row was already evaluated by the model and has active predictions, preserve them!
        has_existing_predictions = any(
            str(predictions.at[idx, d]).strip() not in ("", "-", "None", "nan")
            for d in DAY_COLUMNS if d in predictions.columns
        )
        if has_existing_predictions:
            predictions.at[idx, "IS_NEW_CUSTOMER"] = "True"
            updated += 1
            continue

        existing_reasons = {}
        if "PREDICTION_REASON" in predictions.columns and predictions.at[idx, "PREDICTION_REASON"]:
            try:
                existing_reasons = json.loads(predictions.at[idx, "PREDICTION_REASON"])
            except Exception:
                existing_reasons = {}

        day_reasons: dict[str, str] = dict(existing_reasons)
        for day in DAY_COLUMNS:
            if day in predictions.columns:
                if allowed_day_labels is not None and day not in allowed_day_labels:
                    predictions.at[idx, day] = "-"
                    day_reasons[day] = f"No communication scheduled on {day} (outside active prediction window)."
                else:
                    strat = fallback_map.get((matched_key, day), "-")
                    if strat not in ("", "-", "None", "nan"):
                        predictions.at[idx, day] = strat
                        day_reasons[day] = day_reason_map.get(
                            (matched_key, day),
                            f"Cold-start fallback strategy applied based on {risk_map[matched_key]}-risk historical average."
                        )
                    else:
                        curr_val = str(predictions.at[idx, day]).strip()
                        if curr_val not in ("", "-", "None", "nan") and day in existing_reasons:
                            day_reasons[day] = existing_reasons[day]
                        else:
                            predictions.at[idx, day] = curr_val if curr_val not in ("", "None", "nan") else "-"
                            day_reasons[day] = day_reason_map.get(
                                (matched_key, day),
                                f"No communication scheduled on {day} based on {risk_map[matched_key]}-risk cohort guidelines."
                            )
        if "D" in predictions.columns:
            predictions.at[idx, "D"] = "-"
            day_reasons["D"] = "No communication is scheduled on the EMI due date."
        if "PREDICTION_REASON" in predictions.columns:
            predictions.at[idx, "PREDICTION_REASON"] = json.dumps(day_reasons, sort_keys=True)
        predictions.at[idx, "IS_NEW_CUSTOMER"] = "True"
        updated += 1

    raw_fallback_cards = set(fallback_rows["apacCardNumber"].astype(str).str.strip())
    missing_loans = [
        card for card in raw_fallback_cards
        if card not in existing_loans and card.replace(".0", "").lstrip("0") not in existing_clean_loans
    ]
    if missing_loans:
        sample_month_used = predictions["SOURCE_MONTH_USED"].iloc[0] if "SOURCE_MONTH_USED" in predictions.columns and not predictions.empty else ""
        sample_month = predictions["MONTH"].iloc[0] if "MONTH" in predictions.columns and not predictions.empty else ""

        new_rows = []
        for loan in missing_loans:
            clean_loan = loan.replace(".0", "").lstrip("0")
            matched_key = loan if loan in risk_map else clean_loan
            risk = risk_map.get(matched_key, "LOW")
            vertical = vertical_map.get(matched_key, "LAP")
            row_day_reasons: dict[str, str] = {}
            row_dict = {
                "SOURCE_RISK": risk,
                "SOURCE_VERTICAL": vertical,
                "EMI_DATE": "",
                "Loan_number": loan,
                "SOURCE_MONTH_USED": sample_month_used,
                "MONTH": sample_month,
                "IS_NEW_CUSTOMER": "True",
            }
            for day in DAY_COLUMNS:
                if allowed_day_labels is not None and day not in allowed_day_labels:
                    row_dict[day] = "-"
                    row_day_reasons[day] = f"No communication scheduled on {day} (outside active prediction window)."
                else:
                    strat = fallback_map.get((loan, day), "-")
                    row_dict[day] = strat
                    row_day_reasons[day] = day_reason_map.get(
                        (matched_key, day),
                        f"Cold-start fallback strategy applied based on {risk}-risk historical average."
                    )
            if "D" in row_dict or "D" in predictions.columns:
                row_dict["D"] = "-"
                row_day_reasons["D"] = "No communication is scheduled on the EMI due date."
            row_dict["PREDICTION_REASON"] = json.dumps(row_day_reasons, sort_keys=True)
            new_rows.append(row_dict)

        new_df = pd.DataFrame(new_rows)
        predictions = pd.concat([predictions, new_df], ignore_index=True)
        updated += len(missing_loans)

    if updated or "IS_NEW_CUSTOMER" in predictions.columns:
        predictions.to_csv(prediction_file, index=False)
    return updated


def main() -> None:
    args = parse_args()
    cases_file = Path(args.cases_file)
    output_file = ensure_parent_dir(args.output_file)
    communication_files = [Path(path) for path in args.communication_files] or sorted(COMMUNICATION_DATA_DIR.glob("*.csv"))

    if args.allowed_day_offsets is not None and len(args.allowed_day_offsets) > 0:
        parsed_offsets = [parse_day_offset(o) for o in args.allowed_day_offsets]
        allowed_offsets = [o for o in parsed_offsets if o is not None and o != 0]
    else:
        app_config = load_default_config(args.config_file) if getattr(args, "config_file", None) else {}
        allowed_offsets = app_config.get("allowed_day_offsets")
    allowed_day_labels = set(offsets_to_day_columns(allowed_offsets)) if allowed_offsets is not None else None

    case_apacs = load_case_apacs(cases_file)
    history_apacs = load_history_apacs(communication_files)
    is_in_history = (
        case_apacs["apacCardNumber"].isin(history_apacs)
        | case_apacs["apacCardNumber"].str.lstrip("0").isin(history_apacs)
    )
    new_cases = case_apacs[~is_in_history].copy()
    active_predue_days = [d for d in PREDUE_DAY_COLUMNS if allowed_day_labels is None or d in allowed_day_labels]
    fallback_by_risk = build_average_strategy_by_risk(
        Path(args.schedule_file),
        multimode_postdue=getattr(args, "multimode_postdue", True),
        apply_predue_rules=True,
        allowed_predue_days=active_predue_days,
    )
    output = build_fallback_rows(new_cases, fallback_by_risk, allowed_day_labels=allowed_day_labels)
    output.to_csv(output_file, index=False)
    updated_predictions = apply_fallbacks_to_prediction_file(Path(args.prediction_file), output, allowed_day_labels=allowed_day_labels) if args.prediction_file else 0

    print(
        "New customer fallback summary | "
        f"cases={len(case_apacs)} history_apacs={len(history_apacs)} "
        f"new_cases={len(new_cases)} output_rows={len(output)} output_file={output_file} "
        f"updated_predictions={updated_predictions}"
    )


if __name__ == "__main__":
    main()
