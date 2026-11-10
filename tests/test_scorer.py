import json
from pathlib import Path

import pytest

from eval.scorer import evaluate, score_one

ANSWERS = Path("C:/Users/Admin/Downloads/train/train/answers.jsonl")


def g(t, a=None, alias=None):
    return {"type": t, "answer": a, "alias": alias or []}


def test_number_and_percent():
    assert score_one({"type": "number", "answer": 4820000}, g("number", 4820000)) == 1
    assert score_one({"type": "number", "answer": 4820000.0}, g("number", 4820000)) == 1
    assert score_one({"type": "number", "answer": "4820000"}, g("number", 4820000)) == -1  # string is wrong format
    assert score_one({"type": "percent", "answer": 12.404}, g("percent", 12.4)) == 1
    assert score_one({"type": "percent", "answer": 0.124}, g("percent", 12.4)) == -1
    assert score_one({"type": "number", "answer": 12.4}, g("percent", 12.4)) == -1  # type must match


def test_string_normalization_and_alias():
    gold = g("string", "Sting dâu 330ml", ["SP0042"])
    assert score_one({"type": "string", "answer": "STING   dâu 330ml!"}, gold) == 1
    assert score_one({"type": "string", "answer": "sp0042"}, gold) == 1
    assert score_one({"type": "string", "answer": "Sting dau 330ml"}, gold) == -1  # diacritics matter
    assert score_one({"type": "string", "answer": ["Sting dâu 330ml"]}, gold) == -1


def test_list_is_ordered():
    gold = g("list", ["A", "B"], [["SP1"], ["SP2"]])
    assert score_one({"type": "list", "answer": ["a", "sp2"]}, gold) == 1
    assert score_one({"type": "list", "answer": ["B", "A"]}, gold) == -1
    assert score_one({"type": "list", "answer": [13, 2]}, g("list", [13, 2])) == 1


def test_table_rows_as_set_columns_by_name_or_position():
    gold = g(
        "table",
        {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Ba", 100]]},
        [["KH1"], ["KH2"]],
    )
    rows_swapped = {"columns": ["khach_hang", "so_tien_no"], "rows": [["Anh Ba", 100], ["kh1", 3200000]]}
    assert score_one({"type": "table", "answer": rows_swapped}, gold) == 1
    by_name = {"columns": ["so_tien_no", "khach_hang"], "rows": [[3200000, "Chị Lan"], [100, "Anh Ba"]]}
    assert score_one({"type": "table", "answer": by_name}, gold) == 1
    by_position = {"columns": ["Khách", "Nợ"], "rows": [["Chị Lan", 3200000], ["Anh Ba", 100]]}
    assert score_one({"type": "table", "answer": by_position}, gold) == 1
    missing_row = {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000]]}
    assert score_one({"type": "table", "answer": missing_row}, gold) == -1


def test_unanswerable_and_abstain():
    assert score_one({"type": "unanswerable", "reason": "x"}, g("unanswerable")) == 1
    assert score_one({"type": "number", "answer": 0}, g("unanswerable")) == -1
    assert score_one({"type": "unanswerable"}, g("number", 5)) == -1
    assert score_one({"type": "abstain"}, g("number", 5)) == 0
    assert score_one(None, g("number", 5)) == 0


@pytest.mark.skipif(not ANSWERS.exists(), reason="train answers not available")
def test_gold_scores_perfectly():
    golds = {a["id"]: a for a in map(json.loads, ANSWERS.read_text(encoding="utf-8").splitlines())}
    preds = {i: {k: v for k, v in a.items() if k != "alias"} for i, a in golds.items()}
    r = evaluate(preds, golds, sorted(golds))
    assert r["pa"] == 1.0
