"""Text helpers: diacritic folding, the official answer-string normalization,
and fuzzy entity lookup."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def fold(s: str) -> str:
    """Lowercase, strip Vietnamese diacritics (đ -> d), collapse whitespace."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D").lower()
    return _WS.sub(" ", s).strip()


def norm(s: str) -> str:
    """Official string normalization used by the scorer: NFC, lowercase, every
    char that is not a letter/digit/whitespace -> space, collapse whitespace.
    Diacritics are kept."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return _WS.sub(" ", s).strip()


def key(s: str) -> str:
    """Loose comparison key: folded + punctuation removed."""
    return norm(fold(s))


def find(candidates, query: str, k: int = 5, min_score: float = 60.0):
    """Fuzzy-match `query` against an iterable of names (diacritics-insensitive).
    Returns [(name, score)] best first. Exact key matches score 100."""
    from rapidfuzz import fuzz, process

    names = [c for c in dict.fromkeys(candidates) if isinstance(c, str) and c.strip()]
    if not names:
        return []
    q = key(query)
    keys = [key(n) for n in names]
    exact = [(n, 100.0) for n, kk in zip(names, keys, strict=False) if kk == q]
    if exact:
        return exact[:k]
    hits = process.extract(q, keys, scorer=fuzz.WRatio, limit=k * 3)
    out, seen = [], set()
    for _, score, idx in hits:
        if score >= min_score and names[idx] not in seen:
            seen.add(names[idx])
            out.append((names[idx], round(float(score), 1)))
    return out[:k]
