"""Rounding exactly as the rulebook says (§9): money to whole dong, half up;
percentages and other averages to 2 decimals. Never use built-in round(),
which does banker's rounding (round(2.5) == 2)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def _dec(x) -> Decimal:
    # repr() of a float is the shortest string that round-trips, so 0.125 stays 0.125
    return Decimal(repr(float(x)))


def round_half_up(x, ndigits: int = 0):
    q = Decimal(1).scaleb(-ndigits)
    r = _dec(x).quantize(q, rounding=ROUND_HALF_UP)
    return int(r) if ndigits == 0 else float(r)


def round_money(x) -> int:
    """Money amounts and averages of money: whole dong, half up."""
    return round_half_up(x, 0)


def round2(x) -> float:
    """Percentages and non-money averages: 2 decimals, half up."""
    return round_half_up(x, 2)


def pct(part, whole) -> float | None:
    """part / whole as a percentage with 2 decimals (12.4 means 12.4 %)."""
    if not whole:
        return None
    return round2(float(part) * 100.0 / float(whole))


def pct_change(new, old) -> float | None:
    """Percentage change from old to new; negative when it decreases."""
    if not old:
        return None
    return round2((float(new) - float(old)) * 100.0 / float(old))
