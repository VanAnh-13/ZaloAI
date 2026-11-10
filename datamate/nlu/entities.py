"""Names from the shop's data that a question seems to mention ("did you mean" hints)."""

from __future__ import annotations

import re

from datamate.core.text import key, norm


class EntityIndex:
    def __init__(self, entities: list[tuple[str, str, str]]):
        """entities: (value, table, column) for the text columns of one shop."""
        seen: dict[str, tuple[str, str, str]] = {}
        for value, table, column in entities:
            k = key(value)
            if len(k) >= 4 and not k.replace(" ", "").isdigit() and k not in seen:
                seen[k] = (value, table, column)
        self.keys = list(seen)  # accent-free keys, for fuzzy "did you mean" hints
        self.items = [seen[k] for k in self.keys]
        self.spellings = {norm(v) for v, _, _ in entities}  # forms the scorer accepts (diacritics kept)

    def __contains__(self, name: str) -> bool:
        """True when `name` is a spelling that occurs in the data (scorer normalization)."""
        return norm(name) in self.spellings

    def mentioned(self, question: str, limit: int = 8, cutoff: float = 90) -> list[tuple[str, str, str]]:
        """Values whose (accent-free) text appears in the question, longest match first."""
        if not self.keys:
            return []
        from rapidfuzz import fuzz, process

        q = key(question)
        hits = process.extract(q, self.keys, scorer=fuzz.partial_ratio, score_cutoff=cutoff, limit=80)
        hits.sort(key=lambda h: (-h[1], -len(h[0])))
        picked: list[str] = []
        out = []
        for k, _score, i in hits:
            if len(k) < 6 and not re.search(rf"(?<!\w){re.escape(k)}(?!\w)", q):
                continue  # short names must appear as whole words
            if any(k in p for p in picked):
                continue  # already covered by a longer name
            picked.append(k)
            out.append(self.items[i])
            if len(out) >= limit:
                break
        return out
