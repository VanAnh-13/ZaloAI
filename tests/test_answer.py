import pytest

from datamate.agent.answer import AnswerError, canonical, normalize
from datamate.agent.loop import extract_code
from datamate.agent.solver import Vote, agreed, tally


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"type": "number", "value": 4820000.0}, {"type": "number", "answer": 4820000}),
        ({"type": "number", "value": 18.604}, {"type": "number", "answer": 18.6}),
        ({"type": "number", "value": "4.820.000"}, {"type": "number", "answer": 4820000}),
        ({"type": "percent", "value": -12.455}, {"type": "percent", "answer": -12.46}),
        ({"type": "string", "value": ["Sting dâu 330ml"]}, {"type": "string", "answer": "Sting dâu 330ml"}),
        ({"type": "date", "value": "2026-08-15T00:00:00"}, {"type": "date", "answer": "2026-08-15"}),
        ({"type": "date", "value": "15/08/2026"}, {"type": "date", "answer": "2026-08-15"}),
        ({"type": "date", "value": "9/2025"}, {"type": "date", "answer": "2025-09"}),
        ({"type": "boolean", "value": "true"}, {"type": "boolean", "answer": True}),
        ({"type": "list", "value": [13.0, 2]}, {"type": "list", "answer": [13, 2]}),
        (
            {"type": "table", "value": {"columns": ["a", "b"], "rows": [["x", 1.0]]}},
            {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x", 1]]}},
        ),
        (
            {"type": "unanswerable", "reason": "ngoài khoảng dữ liệu"},
            {"type": "unanswerable", "reason": "ngoài khoảng dữ liệu"},
        ),
        ({"type": "ABSTAIN"}, {"type": "abstain"}),
    ],
)
def test_normalize(payload, expected):
    assert normalize(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "money", "value": 1},
        {"type": "number", "value": float("nan")},
        {"type": "number", "value": "abc"},
        {"type": "number"},
        {"type": "string", "value": ["a", "b"]},
        {"type": "list", "value": []},
        {"type": "table", "value": {"columns": ["a"], "rows": [["x", 1]]}},
        {"type": "date", "value": "tháng 9"},
    ],
)
def test_normalize_rejects(payload):
    with pytest.raises(AnswerError):
        normalize(payload)


def test_extract_code():
    assert extract_code("Suy nghĩ…\n```python\nprint(1)\n```") == "print(1)"
    assert extract_code("```python\na=1\n```\n```py\nb=2\n```") == "b=2"
    assert extract_code("```python\nprint(2)") == "print(2)"  # cut by max_tokens
    assert extract_code("không có code") is None


def test_votes_are_counted_by_how_the_answer_scores():
    a = {"type": "string", "answer": "Bia Tiger"}
    b = {"type": "string", "answer": "BIA TIGER!"}  # scores the same -> same vote
    c = {"type": "number", "answer": 3}
    assert canonical(a) == canonical(b)
    V = lambda src, ans: Vote(src, ans)  # noqa: E731
    assert tally([V("template", a), V("code", c), V("code", b)]) == [(a, 2), (c, 1)]
    assert agreed([V("plan", a), V("code", b)]) == a
    assert agreed([V("plan", a), V("code", c)]) is None  # one vote each
    assert agreed([V("plan", a)]) is None and agreed([V("plan", a)], need=1) == a  # effort "fast"
    assert agreed([V("plan", a), V("code", c)], need=1) is None  # a tie never wins
    assert agreed([]) is None and tally([]) == []
