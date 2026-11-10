"""Time periods exactly as rulebook §2 defines them. All ends are inclusive.

p = month(7, 2025)                 # whole July 2025
p = last_week(as_of)               # Monday..Sunday of the previous week
df[p.mask(df["Ngày"])]             # rows inside the period
"""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

DATA_START = date(2000, 1, 1)  # start used for "đến hết ngày X" snapshots


def as_date(x) -> date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str = ""

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def mask(self, s: pd.Series) -> pd.Series:
        """Boolean mask of datetimes in `s` that fall on a day inside the period."""
        d = pd.to_datetime(s, errors="coerce").dt.normalize()
        return (d >= pd.Timestamp(self.start)) & (d <= pd.Timestamp(self.end))

    def contains(self, x) -> bool:
        return self.start <= as_date(x) <= self.end

    def __str__(self) -> str:
        span = str(self.start) if self.start == self.end else f"{self.start} → {self.end}"
        return f"{self.label}: {span}" if self.label else span


def _month_end(y: int, m: int) -> date:
    return date(y, m, _calendar.monthrange(y, m)[1])


def day(d, label: str = "") -> Period:
    d = as_date(d)
    return Period(d, d, label or f"ngày {d:%d/%m/%Y}")


def today(as_of) -> Period:
    return day(as_of, "hôm nay")


def yesterday(as_of) -> Period:
    return day(as_date(as_of) - timedelta(days=1), "hôm qua")


def until(d, label: str = "") -> Period:
    """'Đến hết ngày X' / 'tính đến hôm nay': state at the end of day X,
    counting everything from the start of the data."""
    d = as_date(d)
    return Period(DATA_START, d, label or f"đến hết {d:%d/%m/%Y}")


def this_week(as_of) -> Period:
    t = as_date(as_of)
    return Period(t - timedelta(days=t.weekday()), t, "tuần này")


def last_week(as_of) -> Period:
    mon = this_week(as_of).start - timedelta(days=7)
    return Period(mon, mon + timedelta(days=6), "tuần trước")


def this_month(as_of) -> Period:
    t = as_date(as_of)
    return Period(t.replace(day=1), t, "tháng này")


def month(m: int, y: int | None = None, as_of=None) -> Period:
    """Whole month. Without a year: this year unless that is after today,
    otherwise last year (rulebook §2)."""
    if y is None:
        t = as_date(as_of)
        y = t.year if date(t.year, m, 1) <= t else t.year - 1
    elif y < 100:
        y += 2000
    return Period(date(y, m, 1), _month_end(y, m), f"tháng {m}/{y}")


def last_month(as_of) -> Period:
    t = as_date(as_of).replace(day=1) - timedelta(days=1)
    return month(t.month, t.year)


def same_days_prev_month(as_of) -> Period:
    """Comparison window for 'tháng này' vs 'tháng trước': day 1 .. same day
    number of last month (or its last day if that day does not exist)."""
    t = as_date(as_of)
    pm = t.replace(day=1) - timedelta(days=1)
    return Period(pm.replace(day=1), pm.replace(day=min(t.day, pm.day)), "cùng kỳ tháng trước")


def quarter(q: int, y: int | None = None, as_of=None) -> Period:
    if y is None:
        t = as_date(as_of)
        y = t.year if date(t.year, 3 * q - 2, 1) <= t else t.year - 1
    elif y < 100:
        y += 2000
    return Period(date(y, 3 * q - 2, 1), _month_end(y, 3 * q), f"quý {q}/{y}")


def year(y: int) -> Period:
    return Period(date(y, 1, 1), date(y, 12, 31), f"năm {y}")


def ytd(as_of) -> Period:
    t = as_date(as_of)
    return Period(date(t.year, 1, 1), t, "từ đầu năm")


def last_n_days(n: int, as_of) -> Period:
    t = as_date(as_of)
    return Period(t - timedelta(days=n - 1), t, f"{n} ngày gần nhất")


def _add_months(y: int, m: int, k: int) -> tuple[int, int]:
    i = y * 12 + (m - 1) + k
    return i // 12, i % 12 + 1


def prev_period(p: Period) -> Period:
    """The comparable period right before p ('so với kỳ trước'):
    whole month(s)/quarter/year -> the same number of whole months before;
    a month started on day 1 but not finished ('tháng này') -> day 1 .. same day of
    the previous month (rulebook §2); a week started on Monday -> same weekdays of the
    previous week; anything else -> the window of the same length right before."""
    label = f"kỳ trước của {p.label}".strip()
    s, e = p.start, p.end
    if p.label.endswith("ngày gần nhất"):  # "10 ngày gần nhất" ending on the 10th is not "tháng này"
        end = s - timedelta(days=1)
        return Period(end - timedelta(days=p.days - 1), end, label)
    if s.day == 1 and e == _month_end(e.year, e.month):
        n = (e.year - s.year) * 12 + e.month - s.month + 1
        y0, m0 = _add_months(s.year, s.month, -n)
        y1, m1 = _add_months(e.year, e.month, -n)
        return Period(date(y0, m0, 1), _month_end(y1, m1), label)
    if s.day == 1 and (s.year, s.month) == (e.year, e.month):
        y, m = _add_months(s.year, s.month, -1)
        return Period(date(y, m, 1), date(y, m, min(e.day, _month_end(y, m).day)), label)
    if s.weekday() == 0 and p.days <= 7:
        return Period(s - timedelta(days=7), e - timedelta(days=7), label)
    end = s - timedelta(days=1)
    return Period(end - timedelta(days=p.days - 1), end, label)


def same_period_last_year(p: Period) -> Period:
    def back(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:  # 29/02
            return d.replace(year=d.year - 1, day=28)

    return Period(back(p.start), back(p.end), f"cùng kỳ năm trước của {p.label}".strip())


def months_ending(m: int, y: int, n: int) -> list[Period]:
    """n consecutive months ending at m/y, oldest first."""
    out = []
    for i in range(n - 1, -1, -1):
        mm, yy = m - i, y
        while mm <= 0:
            mm += 12
            yy -= 1
        out.append(month(mm, yy))
    return out


def tet(y: int, cal: pd.DataFrame) -> Period:
    """Tết năm Y: 23 tháng Chạp (lunar year Y-1) .. mùng 10 tháng Giêng (lunar year Y)."""
    c = cal.copy()
    leap = c["lunar_leap"].astype(str).str.lower().isin(["true", "1"])
    c = c[~leap]
    start = c[(c.lunar_year == y - 1) & (c.lunar_month == 12) & (c.lunar_day == 23)]["date"]
    end = c[(c.lunar_year == y) & (c.lunar_month == 1) & (c.lunar_day == 10)]["date"]
    if start.empty or end.empty:
        raise ValueError(f"calendar.csv không có đủ ngày cho Tết {y}")
    return Period(as_date(start.iloc[0]), as_date(end.iloc[0]), f"Tết {y}")


def before_tet(y: int, cal: pd.DataFrame) -> Period:
    """'So với trước Tết': the same number of days right before 23 tháng Chạp."""
    p = tet(y, cal)
    return Period(p.start - timedelta(days=p.days), p.start - timedelta(days=1), f"trước Tết {y}")


VN_WEEKDAYS = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật")


def vn_weekday(d) -> str:
    return VN_WEEKDAYS[as_date(d).weekday()]


def shift_of(ts) -> str | None:
    """Ca theo giờ tạo đơn (rulebook §2): 'sáng' 06:00–13:59, 'chiều' 14:00–21:59, else None.
    Works on a scalar; for a column use s.map(dm.shift_of)."""
    if ts is None or (not isinstance(ts, str) and pd.isna(ts)):
        return None
    h = pd.Timestamp(ts).hour
    if 6 <= h < 14:
        return "sáng"
    if 14 <= h < 22:
        return "chiều"
    return None
