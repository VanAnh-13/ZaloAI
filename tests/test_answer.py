import pytest

from datamate.agent.answer import AnswerError, canonical, normalize
from datamate.agent.loop import extract_code
from datamate.agent.solver import vote


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


def test_vote():
    a = {"type": "string", "answer": "Bia Tiger"}
    b = {"type": "string", "answer": "BIA TIGER!"}  # scores the same -> same vote
    c = {"type": "number", "answer": 3}
    assert canonical(a) == canonical(b)
    assert vote([a], 0.5) == (a, 1.0)
    assert vote([a, c], 0.5)[0] == {"type": "abstain"}  # tie
    assert vote([a, b, c], 0.5)[0] == a
    assert vote([None, None], 0.5)[0] == {"type": "abstain"}
