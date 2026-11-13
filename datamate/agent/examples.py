"""Worked examples for the code agent: train questions whose program gave the reference answer.

`python -m eval.harvest` builds datamate/agent/examples.jsonl from a run on the TRAIN split
(every program is re-executed and must reproduce the reference answer). At run time the most
similar examples are shown before the question, so the model sees how questions of the same
shape were solved — with this shop's own tables it still has to write the code itself.

Retrieval is lexical (TF-IDF over words and word pairs, 0 parameters): the model budget of the
challenge leaves no room for an embedding model. Dates, numbers and period words are masked
first, so "doanh thu tháng 7/2025" and "doanh thu th3/2026" are the same question.

Without an examples file nothing is shown and the agent behaves as before.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datamate.core.text import fold
from datamate.nlu.normalize import rewrite

DEFAULT_PATH = Path(__file__).with_name("examples.jsonl")
MIN_SCORE = 0.22  # below this the "similar" question is not similar enough to help

_MASKS = (
    (re.compile(r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b"), " <ngay> "),
    (re.compile(r"\b(thang|th|t)\s*\d{1,2}(\s*(/|-|nam)\s*\d{2,4})?\b"), " <thang> "),
    (re.compile(r"\b(quy|q)\s*[1-4](\s*(/|-|nam)\s*\d{2,4})?\b"), " <quy> "),
    (re.compile(r"\bnam\s*20\d{2}\b|\b20\d{2}\b"), " <nam> "),
    (re.compile(r"\b\d{1,2}\s*[/-]\s*\d{2,4}\b"), " <thang> "),
    (re.compile(r"\d+([.,]\d+)*"), " <so> "),
)


def signature(question: str) -> str:
    """Question with teencode rewritten, accents folded and every date/number masked."""
    s = fold(rewrite(question)).replace("<", " ").replace(">", " ")
    for rx, rep in _MASKS:  # before punctuation is dropped: "7/2025" must still have its slash
        s = rx.sub(rep, s)
    s = re.sub(r"[^a-z0-9<>\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _terms(sig: str) -> list[str]:
    words = sig.split()
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]


@dataclass
class Example:
    id: str
    business_id: str
    question: str
    as_of: str
    type: str
    code: str

    def render(self, n: int) -> str:
        return f'### Ví dụ {n}\nCâu hỏi (AS_OF {self.as_of}): "{self.question}"\n```python\n{self.code.strip()}\n```'


class ExampleBank:
    def __init__(self, examples: list[Example]):
        self.examples = examples
        self._sigs = [signature(e.question) for e in examples]
        df: Counter = Counter()
        tfs = []
        for sig in self._sigs:
            tf = Counter(_terms(sig))
            tfs.append(tf)
            df.update(tf.keys())
        n = max(1, len(examples))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._default_idf = math.log(n + 1) + 1.0
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, tf in enumerate(tfs):
            vec = {t: (1 + math.log(c)) * self._idf[t] for t, c in tf.items()}
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            for t, w in vec.items():
                self._postings[t].append((i, w / norm))

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "ExampleBank":
        if not Path(path).exists():
            return cls([])
        out = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(Example(d["id"], d.get("business_id", ""), d["question"], d["as_of"], d["type"], d["code"]))
        return cls(out)

    def retrieve(self, question: str, k: int = 3, exclude_business: str | None = None) -> list[tuple[Example, float]]:
        """The k most similar examples (cosine of TF-IDF vectors), best first; one per distinct
        wording. exclude_business is for development runs on train only (examples_holdout)."""
        if not self.examples or k <= 0:
            return []
        tf = Counter(_terms(signature(question)))
        vec = {t: (1 + math.log(c)) * self._idf.get(t, self._default_idf) for t, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for t, w in vec.items():
            for i, wi in self._postings.get(t, ()):
                scores[i] += (w / norm) * wi
        picked: list[tuple[Example, float]] = []
        seen: set[str] = set()
        for i, sc in sorted(scores.items(), key=lambda x: (-x[1], x[0])):
            if sc < MIN_SCORE or len(picked) >= k:
                break
            e = self.examples[i]
            if exclude_business is not None and e.business_id == exclude_business:
                continue
            if self._sigs[i] in seen:
                continue
            seen.add(self._sigs[i])
            picked.append((e, round(sc, 3)))
        return picked


def render(examples: list[tuple[Example, float]]) -> str:
    if not examples:
        return ""
    head = (
        "## Ví dụ đã giải đúng (câu hỏi tương tự ở shop KHÁC)\n"
        "Học CÁCH LÀM: chọn kỳ, bộ lọc, công thức, kiểu đáp án. Tên bảng và tên cột của shop khác với shop này — "
        "không chép tên cột, dùng schema card của shop này."
    )
    return "\n\n".join([head] + [e.render(i) for i, (e, _) in enumerate(examples, 1)])
