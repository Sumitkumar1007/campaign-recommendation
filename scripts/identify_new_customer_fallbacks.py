from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import DAY_COLUMNS
from project_paths import CASE_DATA_DIR, COMMUNICATION_DATA_DIR, SCHEDULE_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify APACs present in current cases but absent from communication "
            "history, then prepare LOW-risk average fallback strategies."
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
    return parser.parse_args()


def normalize_apac(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_case_apacs(cases_file: Path) -> pd.DataFrame:
    cases = pd.read_csv(cases_file, dtype=str)
    if "apac_card_number" in cases.columns:
        apac_column = "apac_card_number"
    elif "APAC_CARD_NUMBER" in cases.columns:
        apac_column = "APAC_CARD_NUMBER"
    elif "Loan_number" in cases.columns:
        apac_column = "Loan_number"
    else:
        raise ValueError("Cases file must contain apac_card_number, APAC_CARD_NUMBER, or Loan_number.")

    output = pd.DataFrame({"apacCardNumber": normalize_apac(cases[apac_column])})
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
        header = pd.read_csv(csv_file, nrows=0).columns
        if "apac_card_number" not in header:
            continue
        for chunk in pd.read_csv(csv_file, usecols=["apac_card_number"], dtype=str, chunksize=200_000):
            history.update(normalize_apac(chunk["apac_card_number"]))
    history.discard("")
    return history


def most_common_strategy(values: pd.Series) -> str:
    normalized = values.fillna("").astype(str).str.strip()
    normalized = normalized[normalized.ne("") & normalized.ne("-")]
    if normalized.empty:
        return "-"
    return str(normalized.value_counts().sort_values(ascending=False).index[0])


def build_average_strategy_by_risk(schedule_file: Path) -> dict[str, dict[str, str]]:
    schedule = pd.read_csv(schedule_file, dtype=str)
    if "RISK" not in schedule.columns:
        return {
            "LOW": {
                day: most_common_strategy(schedule[day]) if day in schedule.columns else "-"
                for day in DAY_COLUMNS
            }
        }

    schedule["_RISK"] = schedule["RISK"].fillna("").astype(str).str.upper().str.strip()
    average_by_risk: dict[str, dict[str, str]] = {}
    for risk, risk_df in schedule.groupby("_RISK", sort=False):
        if not risk:
            continue
        average_by_risk[risk] = {
            day: most_common_strategy(risk_df[day]) if day in risk_df.columns else "-"
            for day in DAY_COLUMNS
        }

    if "LOW" not in average_by_risk:
        average_by_risk["LOW"] = {
            day: most_common_strategy(schedule[day]) if day in schedule.columns else "-"
            for day in DAY_COLUMNS
        }
    return average_by_risk


def build_fallback_rows(new_cases: pd.DataFrame, fallback_by_risk: dict[str, dict[str, str]]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for case in new_cases.to_dict("records"):
        risk = str(case.get("originalRisk", "")).strip().upper() or "LOW"
        fallback_by_day = fallback_by_risk.get(risk) or fallback_by_risk.get("LOW", {})
        for day in DAY_COLUMNS:
            strategy = fallback_by_day.get(day, "-")
            records.append(
                {
                    "apacCardNumber": case["apacCardNumber"],
                    "originalRisk": case.get("originalRisk", ""),
                    "fallbackRiskUsed": risk,
                    "vertical": case.get("vertical", ""),
                    "day": day,
                    "recommendedStrategy": strategy,
                    "isActionable": "false" if strategy == "-" else "true",
                    "reason": (
                        "No communication history found for this APAC. "
                        f"{risk}-risk average historical strategy is used as cold-start fallback."
                    ),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    cases_file = Path(args.cases_file)
    output_file = ensure_parent_dir(args.output_file)
    communication_files = [Path(path) for path in args.communication_files] or sorted(COMMUNICATION_DATA_DIR.glob("*.csv"))

    case_apacs = load_case_apacs(cases_file)
    history_apacs = load_history_apacs(communication_files)
    new_cases = case_apacs[~case_apacs["apacCardNumber"].isin(history_apacs)].copy()
    fallback_by_risk = build_average_strategy_by_risk(Path(args.schedule_file))
    output = build_fallback_rows(new_cases, fallback_by_risk)
    output.to_csv(output_file, index=False)

    print(
        "New customer fallback summary | "
        f"cases={len(case_apacs)} history_apacs={len(history_apacs)} "
        f"new_cases={len(new_cases)} output_rows={len(output)} output_file={output_file}"
    )


if __name__ == "__main__":
    main()
