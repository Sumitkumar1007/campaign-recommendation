from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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


def normalize_apac(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def month_label_from_payment_file(path: Path) -> str | None:
    stem = path.stem
    token = stem.replace("payment_data_", "", 1)
    try:
        return datetime.strptime(token, "%b%Y").strftime("%b-%Y").upper()
    except Exception:
        return None


def load_paid_keys(payment_files: list[Path]) -> tuple[set[tuple[str, str]], set[str]]:
    paid: set[tuple[str, str]] = set()
    fetched_months: set[str] = set()
    for payment_file in payment_files:
        if not payment_file.exists():
            continue
        header = pd.read_csv(payment_file, nrows=0).columns
        if "apac_card_number" not in header:
            continue
        month_label = month_label_from_payment_file(payment_file)
        if month_label:
            fetched_months.add(month_label)
        usecols = ["apac_card_number"]
        if "payment_datetime" in header:
            usecols.append("payment_datetime")
        for chunk in pd.read_csv(payment_file, usecols=usecols, dtype=str, chunksize=200_000):
            apacs = normalize_apac(chunk["apac_card_number"])
            if "payment_datetime" in chunk.columns:
                months = pd.to_datetime(chunk["payment_datetime"], errors="coerce").dt.strftime("%b-%Y").str.upper()
            else:
                months = pd.Series(month_label, index=chunk.index)
            for apac, month in zip(apacs, months, strict=False):
                if apac and pd.notna(month):
                    normalized_month = str(month).upper().strip()
                    paid.add((apac, normalized_month))
                    fetched_months.add(normalized_month)
    return paid, fetched_months


def append_payment_flags(feature_file: Path, payment_files: list[Path], output_file: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_file, dtype=str)
    key_column = "APAC_CARD_NUMBER" if "APAC_CARD_NUMBER" in features.columns else "ENTITY_KEY"
    if key_column not in features.columns:
        raise ValueError("Feature file must contain APAC_CARD_NUMBER or ENTITY_KEY.")
    if "MONTH" not in features.columns:
        raise ValueError("Feature file must contain MONTH.")

    paid_keys, fetched_months = load_paid_keys(payment_files)
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
