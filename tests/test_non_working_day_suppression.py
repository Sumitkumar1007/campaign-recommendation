from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from apply_non_working_day_suppressions import (
    is_non_working_saturday,
    is_saturday,
    is_sunday,
    parse_day_offset,
    suppress_non_working_days_in_dataframe,
)


def test_saturday_and_sunday_helpers():
    # September 2026 dates:
    # 2026-09-05: 1st Saturday (day 5)
    # 2026-09-06: Sunday (day 6)
    # 2026-09-12: 2nd Saturday (day 12)
    # 2026-09-19: 3rd Saturday (day 19)
    # 2026-09-26: 4th Saturday (day 26)
    d_1st_sat = datetime.date(2026, 9, 5)
    d_sun = datetime.date(2026, 9, 6)
    d_2nd_sat = datetime.date(2026, 9, 12)
    d_3rd_sat = datetime.date(2026, 9, 19)
    d_4th_sat = datetime.date(2026, 9, 26)

    assert is_saturday(d_1st_sat) is True
    assert is_non_working_saturday(d_1st_sat) is False

    assert is_sunday(d_sun) is True

    assert is_non_working_saturday(d_2nd_sat) is True

    assert is_saturday(d_3rd_sat) is True
    assert is_non_working_saturday(d_3rd_sat) is False

    assert is_non_working_saturday(d_4th_sat) is True


def test_parse_day_offset():
    assert parse_day_offset("D-5") == -5
    assert parse_day_offset("D-1") == -1
    assert parse_day_offset("D") == 0
    assert parse_day_offset("D+1") == 1
    assert parse_day_offset("D+20") == 20

    with pytest.raises(ValueError):
        parse_day_offset("INVALID")


def test_suppress_non_working_days_in_dataframe():
    # Account with EMI_DATE 05/09/2026 (Saturday)
    # D-1 -> 04/09/2026 (Friday, working)
    # D+1 -> 06/09/2026 (Sunday, non-working)
    # D+7 -> 12/09/2026 (2nd Saturday, non-working bank day)
    # D+14 -> 19/09/2026 (3rd Saturday, working bank day)
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC001",
                "EMI_DATE": "05/09/2026",
                "D-1": "SMS-11AM-ENGLISH",
                "D+1": "WH-4PM-HINDI",
                "D+7": "VOICE-2PM-ENGLISH",
                "D+14": "SMS-9AM-ENGLISH",
            }
        ]
    )

    updated_df, stats = suppress_non_working_days_in_dataframe(
        df,
        saturday_mode="bank_saturdays",
        suppress_sundays=True,
    )

    assert updated_df.iloc[0]["D-1"] == "SMS-11AM-ENGLISH"  # Friday preserved
    assert updated_df.iloc[0]["D+1"] == "-"                # Sunday suppressed
    assert updated_df.iloc[0]["D+7"] == "-"                # 2nd Saturday suppressed
    assert updated_df.iloc[0]["D+14"] == "SMS-9AM-ENGLISH" # 3rd Saturday preserved

    assert stats["sunday_suppressions"] == 1
    assert stats["saturday_suppressions"] == 1
    assert stats["total_suppressions"] == 2


def test_suppress_all_saturdays_mode():
    df = pd.DataFrame(
        [
            {
                "Loan_number": "ACC002",
                "EMI_DATE": "05/09/2026",
                "D-1": "SMS-11AM-ENGLISH", # Friday
                "D+14": "SMS-9AM-ENGLISH", # 3rd Saturday
            }
        ]
    )

    updated_df, stats = suppress_non_working_days_in_dataframe(
        df,
        saturday_mode="all_saturdays",
        suppress_sundays=True,
    )

    assert updated_df.iloc[0]["D-1"] == "SMS-11AM-ENGLISH"
    assert updated_df.iloc[0]["D+14"] == "-"  # 3rd Saturday suppressed in all_saturdays mode
    assert stats["saturday_suppressions"] == 1
