"""Ranking with the rulebook tie-break: descending by the criterion, equal
values ordered by code ascending (by name when there is no code)."""

from __future__ import annotations

import pandas as pd


def top_k(
    df: pd.DataFrame,
    by: str,
    k: int | None = None,
    code: str | None = None,
    name: str | None = None,
    ascending: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Sort `df` by `by` (descending unless ascending=True), tie-break by `code`
    then `name` ascending, keep the first k rows.

    Returns (rows, tie_at_boundary). The organizers removed questions whose
    reference ranking ties at the k boundary, so tie_at_boundary=True usually
    means the computation is wrong."""
    keys = [by] + [c for c in (code, name) if c]
    order = [ascending] + [True] * (len(keys) - 1)
    s = df.sort_values(keys, ascending=order, kind="mergesort").reset_index(drop=True)
    if k is None or len(s) <= k:
        return s, False
    tie = bool(s[by].iloc[k - 1] == s[by].iloc[k])
    return s.head(k), tie
