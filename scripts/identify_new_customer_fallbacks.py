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
    parser.add_argument(
        "--prediction-file",
        default="",
        help="Optional prediction CSV to update in-place for new APACs.",
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


def apply_fallbacks_to_prediction_file(prediction_file: Path, fallback_rows: pd.DataFrame) -> int:
    if not prediction_file.exists() or fallback_rows.empty:
        return 0
    predictions = pd.read_csv(prediction_file, dtype=str).fillna("")
    if "Loan_number" not in predictions.columns:
        raise ValueError("Prediction file must contain Loan_number column.")

    fallback_map: dict[tuple[str, str], str] = {
        (str(row.apacCardNumber).strip(), str(row.day).strip()): str(row.recommendedStrategy).strip() or "-"
        for row in fallback_rows.itertuples(index=False)
    }
    risk_map: dict[str, str] = {
        str(row.apacCardNumber).strip(): str(row.fallbackRiskUsed).strip().upper() or "LOW"
        for row in fallback_rows.itertuples(index=False)
    }
    reason_map: dict[str, str] = {
        str(row.apacCardNumber).strip(): str(row.reason).strip()
        for row in fallback_rows.itertuples(index=False)
    }

    if "IS_NEW_CUSTOMER" not in predictions.columns:
        predictions["IS_NEW_CUSTOMER"] = "False"

    updated = 0
    existing_loans = set(predictions["Loan_number"].astype(str).str.strip())
    vertical_map: dict[str, str] = {
        str(row.apacCardNumber).strip(): str(row.vertical).strip().upper()
        for row in fallback_rows.itertuples(index=False)
    }

    for idx, row in predictions.iterrows():
        loan = str(row.get("Loan_number", "")).strip()
        if loan not in risk_map:
            continue
        if "SOURCE_RISK" in predictions.columns:
            predictions.at[idx, "SOURCE_RISK"] = risk_map[loan]
        for day in DAY_COLUMNS:
            if day in predictions.columns:
                predictions.at[idx, day] = fallback_map.get((loan, day), "-")
        if "D" in predictions.columns:
            predictions.at[idx, "D"] = "-"
        if "PREDICTION_REASON" in predictions.columns:
            predictions.at[idx, "PREDICTION_REASON"] = reason_map[loan]
        predictions.at[idx, "IS_NEW_CUSTOMER"] = "True"
        updated += 1

    missing_loans = set(risk_map.keys()) - existing_loans
    if missing_loans:
        sample_month_used = predictions["SOURCE_MONTH_USED"].iloc[0] if "SOURCE_MONTH_USED" in predictions.columns and not predictions.empty else ""
        sample_month = predictions["MONTH"].iloc[0] if "MONTH" in predictions.columns and not predictions.empty else ""

        new_rows = []
        for loan in missing_loans:
            risk = risk_map[loan]
            vertical = vertical_map.get(loan, "LAP")
            reason = reason_map.get(loan, "")
            row_dict = {
                "SOURCE_RISK": risk,
                "SOURCE_VERTICAL": vertical,
                "EMI_DATE": "",
                "Loan_number": loan,
                "SOURCE_MONTH_USED": sample_month_used,
                "MONTH": sample_month,
                "PREDICTION_REASON": reason,
                "IS_NEW_CUSTOMER": "True",
            }
            for day in DAY_COLUMNS:
                row_dict[day] = fallback_map.get((loan, day), "-")
            if "D" in row_dict:
                row_dict["D"] = "-"
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

    case_apacs = load_case_apacs(cases_file)
    history_apacs = load_history_apacs(communication_files)
    is_in_history = (
        case_apacs["apacCardNumber"].isin(history_apacs)
        | case_apacs["apacCardNumber"].str.lstrip("0").isin(history_apacs)
    )
    new_cases = case_apacs[~is_in_history].copy()
    fallback_by_risk = build_average_strategy_by_risk(Path(args.schedule_file))
    output = build_fallback_rows(new_cases, fallback_by_risk)
    output.to_csv(output_file, index=False)
    updated_predictions = apply_fallbacks_to_prediction_file(Path(args.prediction_file), output) if args.prediction_file else 0

    print(
        "New customer fallback summary | "
        f"cases={len(case_apacs)} history_apacs={len(history_apacs)} "
        f"new_cases={len(new_cases)} output_rows={len(output)} output_file={output_file} "
        f"updated_predictions={updated_predictions}"
    )


if __name__ == "__main__":
    main()
