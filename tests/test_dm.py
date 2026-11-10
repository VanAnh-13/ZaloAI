import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from datamate import dm

CAL = Path("C:/Users/Admin/Downloads/train/train/calendar.csv")


def P(a, b):
    return (date.fromisoformat(a), date.fromisoformat(b))


def span(p):
    return (p.start, p.end)


def test_weeks_and_months():
    as_of = date(2026, 3, 4)  # Wednesday
    assert span(dm.this_week(as_of)) == P("2026-03-02", "2026-03-04")
    assert span(dm.last_week(as_of)) == P("2026-02-23", "2026-03-01")
    assert span(dm.this_month(as_of)) == P("2026-03-01", "2026-03-04")
    assert span(dm.last_month(as_of)) == P("2026-02-01", "2026-02-28")
    assert span(dm.same_days_prev_month(date(2026, 3, 31))) == P("2026-02-01", "2026-02-28")
    assert span(dm.last_n_days(7, as_of)) == P("2026-02-26", "2026-03-04")
    assert span(dm.ytd(as_of)) == P("2026-01-01", "2026-03-04")


def test_month_and_quarter_without_year():
    as_of = date(2026, 3, 4)
    assert span(dm.month(3, as_of=as_of)) == P("2026-03-01", "2026-03-31")  # not after today: this year
    assert span(dm.month(5, as_of=as_of)) == P("2025-05-01", "2025-05-31")  # after today: last year
    assert span(dm.quarter(2, as_of=as_of)) == P("2025-04-01", "2025-06-30")
    assert span(dm.month(7, 25)) == P("2025-07-01", "2025-07-31")


def test_prev_period_is_calendar_aware():
    assert span(dm.prev_period(dm.month(3, 2026))) == P("2026-02-01", "2026-02-28")
    assert span(dm.prev_period(dm.month(6, 2025))) == P("2025-05-01", "2025-05-31")
    assert span(dm.prev_period(dm.quarter(1, 2026))) == P("2025-10-01", "2025-12-31")
    assert span(dm.prev_period(dm.year(2026))) == P("2025-01-01", "2025-12-31")
    assert span(dm.prev_period(dm.this_month(date(2026, 3, 31)))) == P("2026-02-01", "2026-02-28")
    assert span(dm.prev_period(dm.this_week(date(2026, 3, 4)))) == P("2026-02-23", "2026-02-25")
    assert span(dm.prev_period(dm.last_n_days(10, date(2026, 3, 10)))) == P("2026-02-19", "2026-02-28")


def test_mask_and_shift():
    s = pd.Series(pd.to_datetime(["2026-02-28 23:59", "2026-03-01 00:00", "2026-03-04 13:00", "2026-03-05 00:00"]))
    assert dm.this_month(date(2026, 3, 4)).mask(s).tolist() == [False, True, True, False]
    assert dm.shift_of(pd.Timestamp("2026-03-01 13:59")) == "sáng"
    assert dm.shift_of(pd.Timestamp("2026-03-01 14:00")) == "chiều"
    assert dm.shift_of(pd.Timestamp("2026-03-01 22:00")) is None


@pytest.mark.skipif(not CAL.exists(), reason="train calendar not available")
def test_tet_2025():
    cal = pd.read_csv(CAL)
    assert span(dm.tet(2025, cal)) == P("2025-01-22", "2025-02-07")  # 23 tháng Chạp → mùng 10 tháng Giêng
    assert span(dm.before_tet(2025, cal)) == P("2025-01-05", "2025-01-21")


def test_rounding_is_half_up():
    assert dm.round_money(2.5) == 3 and dm.round_money(3.5) == 4  # round() would give 2 and 4
    assert dm.round2(0.125) == 0.13
    assert dm.pct(1, 8) == 12.5
    assert dm.pct(1, 48) == 2.08
    assert dm.pct_change(90, 100) == -10.0
    assert dm.pct(1, 0) is None


def test_top_k_tie_break_and_boundary():
    df = pd.DataFrame({"ma": ["SP3", "SP1", "SP2", "SP4"], "sl": [5, 9, 5, 1]})
    rows, tie = dm.top_k(df, "sl", 2, code="ma")
    assert rows["ma"].tolist() == ["SP1", "SP2"] and tie
    rows, tie = dm.top_k(df, "sl", 1, code="ma")
    assert rows["ma"].tolist() == ["SP1"] and not tie


def test_find_ignores_diacritics_and_typos():
    names = ["Sting dâu 330ml", "Bia Tiger 330ml", "Cà phê sữa (M)"]
    assert dm.find(names, "sting dau 330ml")[0][0] == "Sting dâu 330ml"
    assert dm.find(names, "ca phe sua M")[0][0] == "Cà phê sữa (M)"


def test_doc_mentions_only_existing_helpers():
    for name in set(re.findall(r"dm\.(\w+)", dm.DOC)):
        assert hasattr(dm, name), f"dm.DOC mentions dm.{name}, which does not exist"
