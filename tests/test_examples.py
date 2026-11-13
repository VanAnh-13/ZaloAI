"""Retrieval of worked examples for the code agent."""

import json

from datamate.agent.examples import Example, ExampleBank, render, signature


def E(i, biz, question, code="answer('number', 1)"):
    return Example(i, biz, question, "2026-01-01", "number", code)


def test_signature_masks_dates_numbers_and_teencode():
    assert signature("Doanh thu tháng 7/2025 là bao nhiêu?") == "doanh thu <thang> la bao nhieu"
    assert signature("dthu th3/2026 bnhiu z") == "doanh thu <thang> bao nhieu vay"
    assert signature("Top 3 mặt hàng bán chạy nhất quý 2/2025") == "top <so> mat hang ban chay nhat <quy>"
    assert signature("Ngày 05/03/2026 bán được mấy đơn") == "ngay <ngay> ban duoc may don"
    assert signature("tuần rồi có bao nhiêu đơn bị huỷ") == "tuan roi co bao nhieu don bi huy"  # relative periods stay


def test_retrieve_finds_the_same_kind_of_question():
    bank = ExampleBank(
        [
            E("a", "b1", "Doanh thu tháng 7/2025 là bao nhiêu?"),
            E("b", "b1", "Top 3 mặt hàng bán chạy nhất tháng 5/2025"),
            E("c", "b2", "Khách nào còn nợ nhiều nhất đến hết hôm nay"),
            E("d", "b2", "doanh thu thang 1/2026 la bao nhieu"),
        ]
    )
    assert [e.id for e, _ in bank.retrieve("dthu th3/2026 bnhiu z", 3)] == ["a"]  # "d" has the same wording
    assert [e.id for e, _ in bank.retrieve("top 5 sp bán chạy nhất th10/2025", 3)] == ["b"]
    assert [e.id for e, _ in bank.retrieve("ai đang nợ nhiều nhất tính đến hôm nay", 3)] == ["c"]
    assert bank.retrieve("tồn kho cuối ngày của sữa tươi", 3) == []  # nothing similar: show nothing
    assert [e.id for e, _ in bank.retrieve("dthu th3/2026 bnhiu z", 3, exclude_business="b1")] == ["d"]
    assert bank.retrieve("doanh thu tháng 2/2026", 0) == [] and ExampleBank([]).retrieve("x", 3) == []


def test_load_and_render(tmp_path):
    path = tmp_path / "examples.jsonl"
    assert len(ExampleBank.load(path)) == 0  # no file: the agent runs without examples
    row = {
        "id": "q9",
        "business_id": "biz_1",
        "question": "Doanh thu tháng 7/2025?",
        "as_of": "2025-08-01",
        "type": "number",
        "code": "answer('number', S.revenue(dm.month(7, 2025)))",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    bank = ExampleBank.load(path)
    text = render(bank.retrieve("doanh thu tháng 9/2025 bao nhiêu", 2))
    assert "shop KHÁC" in text and '(AS_OF 2025-08-01): "Doanh thu tháng 7/2025?"' in text and "S.revenue" in text
    assert render([]) == ""
