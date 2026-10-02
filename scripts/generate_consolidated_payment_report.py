from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from app_logging import setup_logging
from env_utils import load_dotenv
from project_paths import COMMUNICATION_DATA_DIR, PAYMENT_DATA_DIR, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate consolidated payment report matching payment logs with communication logs."
    )
    parser.add_argument(
        "--payment-dir",
        default=str(PAYMENT_DATA_DIR),
        help="Directory containing payment CSV files (data/payments).",
    )
    parser.add_argument(
        "--payment-files",
        nargs="*",
        default=[],
        help="Optional explicit list of payment CSV paths.",
    )
    parser.add_argument(
        "--communication-dir",
        default=str(COMMUNICATION_DATA_DIR),
        help="Directory containing communication CSV files (data/communication).",
    )
    parser.add_argument(
        "--communication-files",
        nargs="*",
        default=[],
        help="Optional explicit list of communication CSV paths.",
    )
    parser.add_argument(
        "--output-file",
        default=str(PAYMENT_DATA_DIR / "consolidated_payment_report.csv"),
        help="Output path for the consolidated payment report CSV.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path. Defaults to artifacts/logs/generate_consolidated_payment_report.log.",
    )
    return parser.parse_args()


def find_apac_column(columns: list[str] | pd.Index) -> str | None:
    cols_map = {str(c).replace("\ufeff", "").strip().lower(): c for c in columns}
    for candidate in ["apac_card_number", "apaccardnumber", "loan_number", "account_number", "contract_number"]:
        if candidate in cols_map:
            return str(cols_map[candidate])
    return None


def month_label_from_filename(path: Path) -> str | None:
    stem = path.stem
    for prefix in ["payment_data_", "communication_data_", "digital_cases_"]:
        if stem.startswith(prefix):
            token = stem.replace(prefix, "", 1).strip()
            if re.fullmatch(r"[A-Za-z]{3}\d{4}", token, re.IGNORECASE):
                try:
                    return datetime.strptime(token, "%b%Y").strftime("%b-%Y").upper()
                except Exception:
                    pass
            parsed = pd.to_datetime(token, errors="coerce")
            if not pd.isna(parsed):
                return parsed.strftime("%b-%Y").upper()
    return None


def generate_consolidated_payment_report(
    payment_files: list[Path],
    communication_files: list[Path],
    output_file: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    # 1. Load Communication Logs & Index reference IDs / Dates / APACs
    logger.info("Loading communication dataset files...")
    comm_records: list[dict[str, str]] = []
    comm_unique_ids: dict[str, dict[str, str]] = {}            # pay_unique_id -> comm_info
    comm_account_date_ids: dict[tuple[str, str], dict[str, str]] = {}  # (apac, date_str) -> comm_info
    comm_account_month_ids: dict[tuple[str, str], dict[str, str]] = {} # (apac, month) -> comm_info
    comm_account_latest_id: dict[str, dict[str, str]] = {}            # apac -> comm_info
    comm_apac_dates: set[tuple[str, str]] = set()              # (apac, comm_date_str)
    all_comm_apacs: set[str] = set()

    for c_file in communication_files:
        if not c_file.exists() or c_file.stat().st_size <= 60:
            continue
        try:
            file_month = month_label_from_filename(c_file)
            for chunk in pd.read_csv(c_file, dtype=str, chunksize=100_000, encoding="utf-8-sig"):
                apac_col = find_apac_column(chunk.columns)
                if not apac_col:
                    continue
                
                chunk["APAC_CARD_NUMBER"] = chunk[apac_col].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                pay_id_col = "payment_unique_id" if "payment_unique_id" in chunk.columns else "PAYMENT_UNIQUE_ID"
                comm_type_col = "communication_type" if "communication_type" in chunk.columns else "COMMUNICATION_TYPE"
                date_col = "date" if "date" in chunk.columns else ("created_date" if "created_date" in chunk.columns else "")

                for _, row in chunk.iterrows():
                    apac = row["APAC_CARD_NUMBER"]
                    if not apac:
                        continue
                    clean_apac = apac.lstrip("0")
                    all_comm_apacs.add(apac)
                    if clean_apac:
                        all_comm_apacs.add(clean_apac)
                    pay_id = str(row.get(pay_id_col, "") or "").strip() if pay_id_col else ""
                    comm_type = str(row.get(comm_type_col, "") or "").strip() if comm_type_col else ""
                    comm_date_str = ""
                    month_label = file_month if file_month else ""
                    if date_col and pd.notna(row.get(date_col)):
                        dt = pd.to_datetime(row[date_col], errors="coerce")
                        if pd.notna(dt):
                            comm_date_str = dt.strftime("%Y-%m-%d")
                            if not month_label:
                                month_label = dt.strftime("%b-%Y").upper()

                    comm_info = {
                        "apac_card_number": apac,
                        "communication_type": comm_type,
                        "comm_date": comm_date_str,
                        "payment_unique_id": pay_id,
                    }
                    if pay_id:
                        comm_unique_ids[pay_id] = comm_info
                    if comm_date_str:
                        comm_apac_dates.add((apac, comm_date_str))
                        if clean_apac:
                            comm_apac_dates.add((clean_apac, comm_date_str))
                        for key_apac in {apac, clean_apac}:
                            if not key_apac:
                                continue
                            if (key_apac, comm_date_str) not in comm_account_date_ids or (pay_id and not comm_account_date_ids[(key_apac, comm_date_str)].get("payment_unique_id")):
                                comm_account_date_ids[(key_apac, comm_date_str)] = comm_info
                    if month_label:
                        for key_apac in {apac, clean_apac}:
                            if not key_apac:
                                continue
                            if (key_apac, month_label) not in comm_account_month_ids or (pay_id and not comm_account_month_ids[(key_apac, month_label)].get("payment_unique_id")):
                                comm_account_month_ids[(key_apac, month_label)] = comm_info
                    for key_apac in {apac, clean_apac}:
                        if not key_apac:
                            continue
                        if key_apac not in comm_account_latest_id or (pay_id and not comm_account_latest_id[key_apac].get("payment_unique_id")):
                            comm_account_latest_id[key_apac] = comm_info
        except Exception as exc:
            logger.warning("Error reading communication file %s: %s", c_file, exc)

    logger.info(
        "Communication index complete | unique_apacs=%s link_ref_ids=%s date_pairs=%s",
        len(all_comm_apacs),
        len(comm_unique_ids),
        len(comm_apac_dates),
    )

    # 2. Load Payment Logs & Classify Payment Matches
    logger.info("Loading payment dataset files...")
    report_rows: list[dict[str, object]] = []

    for p_file in payment_files:
        if not p_file.exists() or p_file.stat().st_size <= 60:
            continue
        try:
            file_month_label = month_label_from_filename(p_file)
            for chunk in pd.read_csv(p_file, dtype=str, chunksize=100_000, encoding="utf-8-sig"):
                apac_col = find_apac_column(chunk.columns)
                if not apac_col:
                    continue

                chunk["APAC_CARD_NUMBER"] = chunk[apac_col].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                ref_col = "reference_number" if "reference_number" in chunk.columns else "REFERENCE_NUMBER"
                dt_col = "payment_datetime" if "payment_datetime" in chunk.columns else "PAYMENT_DATETIME"

                for _, row in chunk.iterrows():
                    apac = row["APAC_CARD_NUMBER"]
                    if not apac:
                        continue
                    clean_apac = apac.lstrip("0")
                    ref_num = str(row.get(ref_col, "") or "").strip() if ref_col else ""
                    pay_dt_raw = row.get(dt_col, "") if dt_col else ""
                    pay_dt = pd.to_datetime(pay_dt_raw, errors="coerce") if pd.notna(pay_dt_raw) else pd.NaT

                    # Month label extraction
                    if pd.notna(pay_dt):
                        month_label = pay_dt.strftime("%b-%Y").upper()
                        pay_date_str = pay_dt.strftime("%Y-%m-%d")
                    else:
                        month_label = file_month_label if file_month_label else "UNKNOWN"
                        pay_date_str = ""

                    # Classification logic
                    payment_flag = "UNKNOWN"
                    matched_comm_type = ""
                    matched_comm_date = ""
                    matched_pay_id = ""
                    payment_multiplier = 1.0

                    # Check 1: Link Payment Match (p.reference_number == c.payment_unique_id)
                    if ref_num and ref_num in comm_unique_ids:
                        payment_flag = "LINK_PAYMENT"
                        comm_info = comm_unique_ids[ref_num]
                        matched_comm_type = comm_info["communication_type"]
                        matched_comm_date = comm_info["comm_date"]
                        matched_pay_id = comm_info["payment_unique_id"]
                        payment_multiplier = 5.0
                        match_verdict = "VALID_LINK_PAYMENT"

                    # Check 2: External Payment Match (p.apac_card_number in communication dataset & date/month match)
                    elif apac in all_comm_apacs or (clean_apac and clean_apac in all_comm_apacs):
                        payment_flag = "EXTERNAL_PAYMENT"
                        matched_comm_date = pay_date_str
                        payment_multiplier = 2.5
                        match_verdict = "VALID_EXTERNAL_PAYMENT"
                        ext_comm_info = (
                            comm_account_date_ids.get((apac, pay_date_str))
                            or comm_account_date_ids.get((clean_apac, pay_date_str))
                            or comm_account_month_ids.get((apac, month_label))
                            or comm_account_month_ids.get((clean_apac, month_label))
                            or comm_account_latest_id.get(apac)
                            or comm_account_latest_id.get(clean_apac)
                        )
                        if ext_comm_info:
                            matched_pay_id = ext_comm_info.get("payment_unique_id", "")
                            matched_comm_type = ext_comm_info.get("communication_type", "")
                            if not matched_comm_date:
                                matched_comm_date = ext_comm_info.get("comm_date", "")
                        if not matched_pay_id:
                            matched_pay_id = ref_num if ref_num else f"EXT-{apac}-{pay_date_str or month_label}"

                    # Check 3: APAC not found in communication dataset
                    else:
                        payment_flag = "NO_COMMUNICATION_PAYMENT"
                        payment_multiplier = 1.0
                        match_verdict = "UNCONTACTED_ACCOUNT_PAYMENT"
                        matched_pay_id = ref_num if ref_num else f"NOC-{apac}-{pay_date_str or month_label}"

                    report_rows.append(
                        {
                            "apac_card_number": apac,
                            "payment_datetime": pay_dt_raw if pay_dt_raw else "",
                            "reference_number": ref_num,
                            "payment_unique_id": matched_pay_id,  # Blank for external/uncontacted payments
                            "month": month_label,
                            "payment_flag": payment_flag,
                            "match_verdict": match_verdict,
                            "is_link_payment": 1 if payment_flag == "LINK_PAYMENT" else 0,
                            "is_external_payment": 1 if payment_flag == "EXTERNAL_PAYMENT" else 0,
                            "is_uncontacted_payment": 1 if payment_flag == "NO_COMMUNICATION_PAYMENT" else 0,
                            "payment_multiplier": payment_multiplier,
                            "matched_communication_type": matched_comm_type,
                            "matched_communication_date": matched_comm_date,
                            "found_in_communication_db": 1 if apac in all_comm_apacs else 0,
                        }
                    )
        except Exception as exc:
            logger.warning("Error reading payment file %s: %s", p_file, exc)

    report_df = pd.DataFrame(report_rows)
    output_path = ensure_parent_dir(output_file)
    if not report_df.empty:
        report_df.to_csv(output_path, index=False)
        logger.info("Saved consolidated payment report | path=%s rows=%s", output_path, len(report_df))
    else:
        # Save empty schema placeholder
        empty_df = pd.DataFrame(
            columns=[
                "apac_card_number",
                "payment_datetime",
                "reference_number",
                "payment_unique_id",
                "month",
                "payment_flag",
                "match_verdict",
                "is_link_payment",
                "is_external_payment",
                "is_uncontacted_payment",
                "payment_multiplier",
                "matched_communication_type",
                "matched_communication_date",
                "found_in_communication_db",
            ]
        )
        empty_df.to_csv(output_path, index=False)
        logger.info("Saved empty consolidated payment report | path=%s", output_path)

    return report_df


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    logger = setup_logging(args.log_file, "consolidated_payment_report")

    payment_files = [Path(p) for p in args.payment_files] if args.payment_files else sorted(Path(args.payment_dir).glob("*.csv"))
    comm_files = [Path(c) for c in args.communication_files] if args.communication_files else sorted(Path(args.communication_dir).glob("*.csv"))

    output_file = Path(args.output_file)
    logger.info(
        "Consolidated payment report generation | payment_files=%s communication_files=%s output_file=%s",
        len(payment_files),
        len(comm_files),
        output_file,
    )

    generate_consolidated_payment_report(payment_files, comm_files, output_file, logger)


if __name__ == "__main__":
    main()
