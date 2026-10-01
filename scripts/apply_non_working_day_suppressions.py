from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
import re

import pandas as pd

from app_logging import setup_logging
from pipeline_common import DAY_COLUMNS
from project_paths import PREDICTIONS_DIR, ensure_parent_dir

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify non-working days (Sundays and 2nd/4th or all Saturdays) for each "
            "account's EMI date, and suppress any recommended communication on those days."
        )
    )
    parser.add_argument(
        "--prediction-file",
        required=True,
        help="Path to prediction CSV file to update.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output CSV path. Defaults to updating --prediction-file in-place.",
    )
    parser.add_argument(
        "--saturday-mode",
        choices=["bank_saturdays", "all_saturdays", "none"],
        default="bank_saturdays",
        help="Saturday suppression mode: 'bank_saturdays' (2nd & 4th Saturday) [default], 'all_saturdays', or 'none'.",
    )
    parser.add_argument(
        "--suppress-sundays",
        action="store_true",
        default=True,
        help="Suppress recommendations on Sundays [default: True].",
    )
    parser.add_argument(
        "--allow-sundays",
        action="store_false",
        dest="suppress_sundays",
        help="Do NOT suppress recommendations on Sundays.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path.",
    )
    return parser.parse_args()


def is_non_working_saturday(dt: datetime.date) -> bool:
    """Returns True if dt is the 2nd or 4th Saturday of the month."""
    if dt.weekday() != 5:  # 5 is Saturday (Monday=0 ... Sunday=6)
        return False
    saturday_index = (dt.day - 1) // 7 + 1
    return saturday_index in (2, 4)


def is_saturday(dt: datetime.date) -> bool:
    """Returns True if dt is any Saturday."""
    return dt.weekday() == 5


def is_sunday(dt: datetime.date) -> bool:
    """Returns True if dt is Sunday."""
    return dt.weekday() == 6


def parse_day_offset(day_label: str) -> int:
    """Parses relative day label like 'D-5' -> -5, 'D+2' -> 2, 'D' -> 0."""
    label = str(day_label).strip().upper()
    if label == "D":
        return 0
    match = re.fullmatch(r"D([+-]\d+)", label)
    if not match:
        raise ValueError(f"Invalid day label format: {day_label!r}")
    return int(match.group(1))


def suppress_non_working_days_in_dataframe(
    df: pd.DataFrame,
    *,
    saturday_mode: str = "bank_saturdays",
    suppress_sundays: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Identifies non-working Saturdays and Sundays relative to each account's EMI_DATE
    and replaces any active campaign recommendation on those target dates with '-'.
    """
    df = df.copy()
    emi_col = "EMI_DATE" if "EMI_DATE" in df.columns else ("emi_date" if "emi_date" in df.columns else None)
    if not emi_col:
        raise ValueError("Prediction dataframe must contain 'EMI_DATE' or 'emi_date' column.")

    day_columns = [col for col in DAY_COLUMNS if col in df.columns]

    stats = {
        "total_rows": len(df),
        "saturday_suppressions": 0,
        "sunday_suppressions": 0,
        "total_suppressions": 0,
    }

    for idx, row in df.iterrows():
        raw_emi = row[emi_col]
        if pd.isna(raw_emi) or not str(raw_emi).strip():
            continue

        parsed_date = pd.to_datetime(raw_emi, dayfirst=True, errors="coerce")
        if pd.isna(parsed_date):
            continue
        base_emi_date = parsed_date.date()

        for day_col in day_columns:
            try:
                offset = parse_day_offset(day_col)
            except ValueError:
                continue

            target_date = base_emi_date + datetime.timedelta(days=offset)
            suppress_reason = None

            if suppress_sundays and is_sunday(target_date):
                suppress_reason = "Sunday"
                stats["sunday_suppressions"] += 1
            elif saturday_mode == "bank_saturdays" and is_non_working_saturday(target_date):
                saturday_occ = (target_date.day - 1) // 7 + 1
                suppress_reason = "2nd Saturday" if saturday_occ == 2 else "4th Saturday"
                stats["saturday_suppressions"] += 1
            elif saturday_mode == "all_saturdays" and is_saturday(target_date):
                suppress_reason = "Saturday"
                stats["saturday_suppressions"] += 1

            if suppress_reason:
                df.at[idx, day_col] = suppress_reason
                stats["total_suppressions"] += 1

    return df, stats


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file, "non_working_day_suppression")
    prediction_file = Path(args.prediction_file)
    output_file = Path(args.output_file) if args.output_file else prediction_file

    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_file}")

    logger.info(
        "Loading prediction file for non-working day suppression | path=%s saturday_mode=%s suppress_sundays=%s",
        prediction_file,
        args.saturday_mode,
        args.suppress_sundays,
    )

    df = pd.read_csv(prediction_file, dtype=str).fillna("")
    updated_df, stats = suppress_non_working_days_in_dataframe(
        df,
        saturday_mode=args.saturday_mode,
        suppress_sundays=args.suppress_sundays,
    )

    ensure_parent_dir(output_file)
    updated_df.to_csv(output_file, index=False)

    logger.info(
        "Non-working day suppression summary | rows=%d sunday_suppressions=%d saturday_suppressions=%d total_suppressions=%d output_file=%s",
        stats["total_rows"],
        stats["sunday_suppressions"],
        stats["saturday_suppressions"],
        stats["total_suppressions"],
        output_file,
    )

    try:
        from generate_prediction_summary import generate_summary_workbooks
        generate_summary_workbooks(updated_df, output_file.parent, output_file.stem)
        logger.info("Saved updated summary Excel files post non-working day suppression.")
    except Exception as e:
        logger.warning("Could not update summary Excel workbooks: %s", e)

    print(
        f"Non-working day suppression complete: {stats['total_suppressions']} recommendations suppressed "
        f"({stats['sunday_suppressions']} Sundays, {stats['saturday_suppressions']} Saturdays) "
        f"across {stats['total_rows']} rows. Output: {output_file}"
    )


if __name__ == "__main__":
    main()
