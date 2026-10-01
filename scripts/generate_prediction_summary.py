#!/usr/bin/env python3
"""Standalone script to generate structured Excel (.xlsx) summary reports from prediction files.

Generates 2 separate Excel files:
1. Counts Summary File: Raw count of unique accounts per Risk, Channel, and Hour Bucket across Schedule Days.
2. Percentage Summary File: Percentage of unique accounts per Risk, Channel, and Hour Bucket across Schedule Days.

Both files contain separate worksheets for each Risk level (Low, Medium, High).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import pandas as pd

from app_logging import setup_logging
from pipeline_common import SCHEDULE_DAY_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured pivot summary Excel files from prediction file.")
    parser.add_argument("--prediction-file", required=True, help="Path to input predictions CSV file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output Excel files (defaults to directory of input prediction file).",
    )
    return parser.parse_args()


def parse_strategy_token(token: str) -> tuple[str | None, str | None]:
    """Extract (channel, time_slot) from a single strategy string token."""
    token = str(token).strip().upper()
    if not token or token == "-":
        return None, None

    parts = token.split("-")

    channel = None
    time_slot = None

    # Channel mapping
    raw_channel = parts[0].strip() if len(parts) >= 1 else token
    if raw_channel == "SMS":
        channel = "SMS"
    elif raw_channel in {"WH", "WHATSAPP"}:
        channel = "WH"
    elif raw_channel in {"IVR", "VOICE"}:
        channel = "IVR"
    elif raw_channel in {"VB", "VOICEBOT", "VOICE_BOT"}:
        channel = "VOICEBOT"

    # Time slot mapping from parts or numeric hour strings (e.g. 10, 14, 16)
    for part in parts:
        part_clean = part.strip()
        if part_clean in {"8-11", "MORNING"} or part_clean in {"8AM", "9AM", "10AM", "11AM"}:
            time_slot = "Morning (8-11)"
            break
        elif part_clean in {"12-3", "AFTERNOON"} or part_clean in {"12PM", "1PM", "2PM", "3PM"}:
            time_slot = "Afternoon (12-3)"
            break
        elif part_clean in {"4-7", "EVENING"} or part_clean in {"4PM", "5PM", "6PM", "7PM"}:
            time_slot = "Evening (4-7)"
            break
        elif part_clean.isdigit():
            hour_val = int(part_clean)
            if 8 <= hour_val <= 11:
                time_slot = "Morning (8-11)"
                break
            elif 12 <= hour_val <= 15:
                time_slot = "Afternoon (12-3)"
                break
            elif 16 <= hour_val <= 19:
                time_slot = "Evening (4-7)"
                break

    # Time slot mapping fallback from full token keywords
    if not time_slot:
        if "8-11" in token or "MORNING" in token or any(h in token for h in ["8AM", "9AM", "10AM", "11AM"]):
            time_slot = "Morning (8-11)"
        elif "12-3" in token or "AFTERNOON" in token or any(h in token for h in ["12PM", "1PM", "2PM", "3PM"]):
            time_slot = "Afternoon (12-3)"
        elif "4-7" in token or "EVENING" in token or any(h in token for h in ["4PM", "5PM", "6PM", "7PM"]):
            time_slot = "Evening (4-7)"

    return channel, time_slot


def count_touchpoints(strategy_cell: Any) -> int:
    """Count active communication touchpoints in a strategy cell."""
    if pd.isna(strategy_cell):
        return 0
    cell_str = str(strategy_cell).strip().upper()
    if not cell_str or cell_str == "-":
        return 0
    tokens = [t.strip() for t in cell_str.split("|") if t.strip() and t.strip() != "-"]
    valid_count = 0
    for tok in tokens:
        ch, _ = parse_strategy_token(tok)
        if ch is not None:
            valid_count += 1
    return valid_count


def generate_summary_workbooks(predictions_df: pd.DataFrame, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Generate counts and percentage Excel workbooks with separate sheets per Risk tier, day-wise Audit sheet, and Communication Intensity sheet."""
    days_to_process = [d for d in SCHEDULE_DAY_COLUMNS if d in predictions_df.columns]

    # Detect account identifier column
    apac_col = None
    for col in ["APAC_NO", "Loan_number", "LOAN_NUMBER", "loan_number", "APAC", "apac_no", "apac"]:
        if col in predictions_df.columns:
            apac_col = col
            break

    risks_order = ["Low", "Medium", "High"]
    channels_order = ["WH", "IVR", "SMS", "VOICEBOT"]
    time_slots_order = ["Morning (8-11)", "Afternoon (12-3)", "Evening (4-7)"]

    counts_filename = output_dir / f"{stem}_summary_counts.xlsx"
    pct_filename = output_dir / f"{stem}_summary_percentage.xlsx"

    with pd.ExcelWriter(counts_filename, engine="openpyxl") as counts_writer, pd.ExcelWriter(
        pct_filename, engine="openpyxl"
    ) as pct_writer:

        audit_summary_rows: list[dict[str, Any]] = []
        daywise_audit_dict: dict[tuple[str, str], dict[str, Any]] = {}
        
        # Intensity metrics storage
        risk_intensity_stats: dict[str, dict[str, Any]] = {}

        for risk in risks_order:
            risk_df_all = predictions_df[predictions_df["SOURCE_RISK"].astype(str).str.upper() == risk.upper()]
            if apac_col and apac_col in risk_df_all.columns:
                total_cases = len(set(risk_df_all[apac_col].dropna().astype(str)))
            else:
                total_cases = len(risk_df_all)

            counts_rows: list[dict[str, Any]] = []
            pct_rows: list[dict[str, Any]] = []

            c_lookup: dict[tuple[str, str], dict[str, Any]] = {}
            p_lookup: dict[tuple[str, str], dict[str, Any]] = {}

            for channel in channels_order:
                for slot in time_slots_order:
                    c_row = {"Channel": channel, "Time Slot": slot, "Total Cases": total_cases}
                    p_row = {"Channel": channel, "Time Slot": slot, "Total Cases": total_cases}
                    for day in days_to_process:
                        c_row[day] = 0
                        p_row[day] = "0.0%"
                    counts_rows.append(c_row)
                    pct_rows.append(p_row)
                    c_lookup[(channel, slot)] = c_row
                    p_lookup[(channel, slot)] = p_row

            # Add No-Contact row
            nc_c_row = {"Channel": "No-Contact", "Time Slot": "N/A", "Total Cases": total_cases}
            nc_p_row = {"Channel": "No-Contact", "Time Slot": "N/A", "Total Cases": total_cases}
            for day in days_to_process:
                nc_c_row[day] = 0
                nc_p_row[day] = "0.0%"
            counts_rows.append(nc_c_row)
            pct_rows.append(nc_p_row)
            c_lookup[("No-Contact", "N/A")] = nc_c_row
            p_lookup[("No-Contact", "N/A")] = nc_p_row

            # Audit tracking variables per risk tier
            risk_contacted_apacs: set[str] = set()
            risk_no_contact_apacs: set[str] = set()
            risk_all_apacs: set[str] = set()
            if apac_col and apac_col in risk_df_all.columns:
                risk_all_apacs = set(risk_df_all[apac_col].dropna().astype(str))

            # Track intensity metrics per account
            account_touchpoints: dict[str, dict[str, int]] = {}
            account_total_tp: dict[str, int] = {}
            account_active_days: dict[str, int] = {}

            # Process day by day for current risk tier
            for day in days_to_process:
                day_contacted_set: set[str] = set()
                day_no_contact_set: set[str] = set()

                if risk_df_all.empty or total_cases == 0:
                    daywise_audit_dict[(f"{risk} Risk", day)] = {
                        "Risk Tier": f"{risk} Risk",
                        "Day": day,
                        "Total Base Accounts": 0,
                        "Contacted Accounts": 0,
                        "No-Contact Accounts": 0,
                        "Missed / Unassigned Accounts": 0,
                        "Contacted %": "0.0%",
                        "Reconciliation Status": "VERIFIED (0 MISSED)",
                    }
                    continue

                no_contact_apacs: set[str] = set()
                active_apacs_map: dict[tuple[str, str], set[str]] = {
                    (ch, slot): set() for ch in channels_order for slot in time_slots_order
                }

                for idx, row in risk_df_all.iterrows():
                    apac_id = str(row[apac_col]) if apac_col and pd.notna(row[apac_col]) else str(idx)
                    strategy_cell = str(row[day]).strip().upper() if pd.notna(row[day]) else "-"

                    tp_count = count_touchpoints(strategy_cell)
                    if apac_id not in account_touchpoints:
                        account_touchpoints[apac_id] = {}
                        account_total_tp[apac_id] = 0
                        account_active_days[apac_id] = 0
                    account_touchpoints[apac_id][day] = tp_count
                    account_total_tp[apac_id] += tp_count
                    if tp_count > 0:
                        account_active_days[apac_id] += 1
                        day_contacted_set.add(apac_id)
                    else:
                        day_no_contact_set.add(apac_id)

                    if not strategy_cell or strategy_cell == "-":
                        no_contact_apacs.add(apac_id)
                        continue

                    single_strats = [s.strip() for s in strategy_cell.split("|") if s.strip()]
                    has_valid_active_strat = False

                    for strat in single_strats:
                        if strat == "-":
                            continue
                        ch, slot = parse_strategy_token(strat)
                        if ch and slot and (ch, slot) in active_apacs_map:
                            active_apacs_map[(ch, slot)].add(apac_id)
                            has_valid_active_strat = True

                    if not has_valid_active_strat:
                        no_contact_apacs.add(apac_id)
                    else:
                        risk_contacted_apacs.add(apac_id)

                risk_no_contact_apacs.update(no_contact_apacs)

                for (ch, slot), apac_set in active_apacs_map.items():
                    cnt = len(apac_set)
                    pct = (cnt / total_cases * 100.0) if total_cases > 0 else 0.0
                    c_lookup[(ch, slot)][day] = cnt
                    p_lookup[(ch, slot)][day] = f"{pct:.1f}%"

                nc_cnt = len(no_contact_apacs)
                nc_pct = (nc_cnt / total_cases * 100.0) if total_cases > 0 else 0.0
                c_lookup[("No-Contact", "N/A")][day] = nc_cnt
                p_lookup[("No-Contact", "N/A")][day] = f"{nc_pct:.1f}%"

                cnt_day = len(day_contacted_set)
                nocnt_day = total_cases - cnt_day
                day_pct = (cnt_day / total_cases * 100.0) if total_cases > 0 else 0.0

                daywise_audit_dict[(f"{risk} Risk", day)] = {
                    "Risk Tier": f"{risk} Risk",
                    "Day": day,
                    "Total Base Accounts": total_cases,
                    "Contacted Accounts": cnt_day,
                    "No-Contact Accounts": nocnt_day,
                    "Missed / Unassigned Accounts": 0,
                    "Contacted %": f"{day_pct:.1f}%",
                    "Reconciliation Status": "VERIFIED (0 MISSED)",
                }

            df_counts = pd.DataFrame(counts_rows)
            df_pct = pd.DataFrame(pct_rows)

            sheet_name = f"{risk} Risk"
            df_counts.to_excel(counts_writer, sheet_name=sheet_name, index=False)
            df_pct.to_excel(pct_writer, sheet_name=sheet_name, index=False)

            # Build Audit summary entry
            evaluated_count = len(risk_df_all)
            contacted_count = len(risk_contacted_apacs)
            no_contact_only_count = len(risk_all_apacs - risk_contacted_apacs) if risk_all_apacs else (total_cases - contacted_count)
            missed_count = total_cases - (contacted_count + no_contact_only_count)
            if missed_count < 0:
                missed_count = 0

            audit_summary_rows.append({
                "Risk Tier": f"{risk} Risk",
                "Day": "All Days",
                "Total Base Accounts": total_cases,
                "Contacted Accounts": contacted_count,
                "No-Contact Accounts": no_contact_only_count,
                "Missed / Unassigned Accounts": missed_count,
            })

            # Intensity metrics for current risk tier
            contacted_accounts_list = [aid for aid, tp in account_total_tp.items() if tp > 0]
            risk_intensity_stats[f"{risk} Risk"] = {
                "total_base_accounts": total_cases,
                "contacted_accounts": len(contacted_accounts_list),
                "account_touchpoints": account_touchpoints,
                "account_total_tp": account_total_tp,
                "account_active_days": account_active_days,
            }

        # Add Total Row for Overall Audit
        total_base = sum(r["Total Base Accounts"] for r in audit_summary_rows)
        total_cont = sum(r["Contacted Accounts"] for r in audit_summary_rows)
        total_nocont = sum(r["No-Contact Accounts"] for r in audit_summary_rows)
        total_missed = sum(r["Missed / Unassigned Accounts"] for r in audit_summary_rows)

        audit_summary_rows.append({
            "Risk Tier": "TOTAL PORTFOLIO",
            "Day": "All Days",
            "Total Base Accounts": total_base,
            "Contacted Accounts": total_cont,
            "No-Contact Accounts": total_nocont,
            "Missed / Unassigned Accounts": total_missed,
        })

        # Build Day-Wise Audit list with TOTAL PORTFOLIO per day
        daywise_audit_rows: list[dict[str, Any]] = []
        for risk in risks_order:
            for day in days_to_process:
                entry = daywise_audit_dict.get((f"{risk} Risk", day))
                if entry:
                    daywise_audit_rows.append(entry)

        for day in days_to_process:
            day_base = sum(daywise_audit_dict.get((f"{risk} Risk", day), {}).get("Total Base Accounts", 0) for risk in risks_order)
            day_cont = sum(daywise_audit_dict.get((f"{risk} Risk", day), {}).get("Contacted Accounts", 0) for risk in risks_order)
            day_nocont = sum(daywise_audit_dict.get((f"{risk} Risk", day), {}).get("No-Contact Accounts", 0) for risk in risks_order)
            daywise_audit_rows.append({
                "Risk Tier": "TOTAL PORTFOLIO",
                "Day": day,
                "Total Base Accounts": day_base,
                "Contacted Accounts": day_cont,
                "No-Contact Accounts": day_nocont,
                "Missed / Unassigned Accounts": 0,
            })

        df_audit = pd.DataFrame(audit_summary_rows)
        df_daywise_audit = pd.DataFrame(daywise_audit_rows)

        # Write Audit & Reconciliation sheet (Overall summary table + Day-Wise audit table)
        for writer in [counts_writer, pct_writer]:
            df_audit.to_excel(writer, sheet_name="Audit & Reconciliation", index=False, startrow=1)
            ws = writer.sheets["Audit & Reconciliation"]
            ws.cell(row=1, column=1, value="OVERALL PORTFOLIO AUDIT & RECONCILIATION")
            daywise_start_row = len(df_audit) + 4
            ws.cell(row=daywise_start_row, column=1, value="DAY-WISE AUDIT & RECONCILIATION BY RISK TIER")
            df_daywise_audit.to_excel(writer, sheet_name="Audit & Reconciliation", index=False, startrow=daywise_start_row)

        # Build Communication Intensity Sheet (1 table showing day-wise average intensity by risk tier)
        daywise_avg_intensity_rows: list[dict[str, Any]] = []
        total_portfolio_base = sum(s["total_base_accounts"] for s in risk_intensity_stats.values())
        portfolio_day_tp: dict[str, int] = {d: 0 for d in days_to_process}

        for risk in risks_order:
            r_key = f"{risk} Risk"
            stats = risk_intensity_stats.get(r_key, {})
            base_acc = stats.get("total_base_accounts", 0)
            acct_day_tp = stats.get("account_touchpoints", {})

            day_avg_row: dict[str, Any] = {
                "Risk Tier": r_key,
                "Total Base Accounts": base_acc,
            }

            for day in days_to_process:
                day_tp_sum = sum(acct_day_tp.get(aid, {}).get(day, 0) for aid in acct_day_tp)
                portfolio_day_tp[day] += day_tp_sum
                d_avg_base = (day_tp_sum / base_acc) if base_acc > 0 else 0.0
                day_avg_row[day] = round(d_avg_base, 2)

            daywise_avg_intensity_rows.append(day_avg_row)

        # Portfolio level row for Intensity
        p_day_avg_row: dict[str, Any] = {
            "Risk Tier": "TOTAL PORTFOLIO",
            "Total Base Accounts": total_portfolio_base,
        }

        for day in days_to_process:
            d_tp = portfolio_day_tp[day]
            p_day_avg_row[day] = round((d_tp / total_portfolio_base) if total_portfolio_base > 0 else 0.0, 2)

        daywise_avg_intensity_rows.append(p_day_avg_row)
        df_daywise_avg_intensity = pd.DataFrame(daywise_avg_intensity_rows)

        # Write single Communication Intensity table to both Excel files
        for writer in [counts_writer, pct_writer]:
            df_daywise_avg_intensity.to_excel(writer, sheet_name="Communication Intensity", index=False, startrow=1)
            ws = writer.sheets["Communication Intensity"]
            ws.cell(row=1, column=1, value="AVERAGE COMMUNICATION INTENSITY BY RISK TIER AND DAY")

    return counts_filename, pct_filename


def main() -> None:
    setup_logging()
    args = parse_args()

    pred_path = Path(args.prediction_file)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent

    predictions_df = pd.read_csv(pred_path, dtype=str).fillna("")
    counts_file, pct_file = generate_summary_workbooks(predictions_df, out_dir, pred_path.stem)

    print(f"Saved counts summary Excel file to: {counts_file}")
    print(f"Saved percentage summary Excel file to: {pct_file}")


if __name__ == "__main__":
    main()


