from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from project_paths import FEATURE_DATA_DIR, PAYMENT_DATA_DIR


FLAG_COLUMNS = ["paid_flag", "bounce_flag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append source-month payment flags to monthly feature CSV.")
    parser.add_argument(
        "--feature-file",
        default=str(FEATURE_DATA_DIR / "strategy_monthly_features_train.csv"),
        help="Monthly feature CSV to update in-place unless --output-file is supplied.",
    )
    parser.add_argument(
        "--payment-files",
        nargs="*",
        default=[],
        help="Payment CSV files. Defaults to data/payments/payment_data_*.csv.",
    )
    parser.add_argument("--output-file", default="", help="Optional output CSV path. Defaults to overwriting --feature-file.")
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


def month_label_from_payment_file(path: Path) -> str | None:
    stem = path.stem
    token = stem.replace("payment_data_", "", 1).strip()
    if re.fullmatch(r"[A-Za-z]{3}\d{4}", token, re.IGNORECASE):
        try:
            return datetime.strptime(token, "%b%Y").strftime("%b-%Y").upper()
        except Exception:
            pass
    parsed = pd.to_datetime(token, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%b-%Y").upper()
    return None


def load_paid_keys(
    payment_files: list[Path],
) -> tuple[
    set[tuple[str, str]],
    set[str],
    set[str],
    set[tuple[str, str]],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
]:
    paid: set[tuple[str, str]] = set()
    fetched_months: set[str] = set()
    link_ref_numbers: set[str] = set()
    external_payment_dates: set[tuple[str, str]] = set()
    link_pay_counts: dict[tuple[str, str], int] = {}
    ext_pay_counts: dict[tuple[str, str], int] = {}
    tot_pay_counts: dict[tuple[str, str], int] = {}

    for payment_file in payment_files:
        if not payment_file.exists() or payment_file.stat().st_size <= 60:
            continue
        try:
            header = pd.read_csv(payment_file, nrows=0, encoding="utf-8-sig").columns
        except Exception:
            continue
        apac_col = find_apac_column(header)
        if not apac_col:
            continue
        month_label = month_label_from_payment_file(payment_file)
        if month_label:
            fetched_months.add(month_label)
        usecols = [apac_col]
        if "payment_datetime" in header:
            usecols.append("payment_datetime")
        elif "PAYMENT_DATETIME" in header:
            usecols.append("PAYMENT_DATETIME")
        if "reference_number" in header:
            usecols.append("reference_number")
        if "month" in header:
            usecols.append("month")
        if "payment_flag" in header:
            usecols.append("payment_flag")

        for chunk in pd.read_csv(payment_file, usecols=usecols, dtype=str, chunksize=200_000):
            apacs = normalize_apac(chunk["apac_card_number"])
            m_col = chunk["month"].fillna("").astype(str).str.upper().str.strip() if "month" in chunk.columns else pd.Series("", index=chunk.index)
            dt_months = pd.Series("", index=chunk.index)
            if "payment_datetime" in chunk.columns:
                parsed_dates = pd.to_datetime(chunk["payment_datetime"], errors="coerce")
                dt_months = parsed_dates.dt.strftime("%b-%Y").str.upper().fillna("")
                date_strs = parsed_dates.dt.strftime("%Y-%m-%d")
            else:
                date_strs = pd.Series(None, index=chunk.index)

            months = m_col.replace({"": None}).fillna(dt_months.replace({"": None})).fillna(month_label if month_label else "")
            ref_nums = chunk["reference_number"].fillna("").astype(str).str.strip() if "reference_number" in chunk.columns else pd.Series("", index=chunk.index)
            flags = chunk["payment_flag"].fillna("").astype(str).str.upper().str.strip() if "payment_flag" in chunk.columns else pd.Series("", index=chunk.index)

            for apac, month, date_str, ref_num, flag in zip(apacs, months, date_strs, ref_nums, flags, strict=False):
                if not apac or not month or month == "UNKNOWN":
                    continue
                normalized_month = str(month).upper().strip()
                key = (apac, normalized_month)

                if flag == "NO_COMMUNICATION_PAYMENT":
                    continue

                paid.add(key)
                fetched_months.add(normalized_month)
                tot_pay_counts[key] = tot_pay_counts.get(key, 0) + 1

                if flag == "LINK_PAYMENT":
                    if ref_num:
                        link_ref_numbers.add(ref_num)
                    link_pay_counts[key] = link_pay_counts.get(key, 0) + 1
                elif flag == "EXTERNAL_PAYMENT":
                    ext_pay_counts[key] = ext_pay_counts.get(key, 0) + 1
                    if date_str and pd.notna(date_str):
                        external_payment_dates.add((apac, str(date_str)))
                else:
                    if ref_num:
                        link_ref_numbers.add(ref_num)
                        link_pay_counts[key] = link_pay_counts.get(key, 0) + 1
                    else:
                        ext_pay_counts[key] = ext_pay_counts.get(key, 0) + 1

                if apac and date_str and pd.notna(date_str):
                    external_payment_dates.add((apac, str(date_str)))

    return paid, fetched_months, link_ref_numbers, external_payment_dates, link_pay_counts, ext_pay_counts, tot_pay_counts


def append_payment_flags(feature_file: Path, payment_files: list[Path], output_file: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_file, dtype=str)
    key_column = "APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in features.columns else "ENTITY_KEY"
    if key_column not in features.columns:
        raise ValueError("Feature file must contain APAC_CARD_NUMBER or ENTITY_KEY.")
    if "MONTH" not in features.columns:
        raise ValueError("Feature file must contain MONTH.")

    paid_keys, fetched_months, _, _, _, _, _ = load_paid_keys(payment_files)
    features = features.drop(columns=FLAG_COLUMNS, errors="ignore")
    keys = list(zip(normalize_apac(features[key_column]), features["MONTH"].fillna("").astype(str).str.upper().str.strip(), strict=False))
    features["paid_flag"] = [1 if key in paid_keys else 0 for key in keys]
    features["bounce_flag"] = [1 if month in fetched_months and (apac, month) not in paid_keys else 0 for apac, month in keys]
    features.to_csv(output_file, index=False)
    return features


def main() -> None:
    args = parse_args()
    feature_file = Path(args.feature_file)
    output_file = Path(args.output_file) if args.output_file else feature_file
    payment_files = [Path(path) for path in args.payment_files] or sorted(PAYMENT_DATA_DIR.glob("payment_data_*.csv"))
    result = append_payment_flags(feature_file, payment_files, output_file)
    print(
        "Payment flags appended | "
        f"feature_file={feature_file} output_file={output_file} payment_files={len(payment_files)} "
        f"rows={len(result)} paid={int(pd.to_numeric(result['paid_flag']).sum())} "
        f"bounced={int(pd.to_numeric(result['bounce_flag']).sum())}"
    )


if __name__ == "__main__":
    main()
