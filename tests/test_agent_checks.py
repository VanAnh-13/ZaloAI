import time

from datamate.agent.confidence import Calibrator
from datamate.agent.hints import analyze
from datamate.agent.verify import check
from datamate.nlu.entities import EntityIndex
from datamate.pipeline import TimeBudget

INDEX = EntityIndex([("Cà phê sữa (M)", "SP", "Tên"), ("Bạc xỉu (L)", "SP", "Tên")])


def test_analysis_flags_periods_outside_the_data():
    a = analyze(
        "Tháng 2/2025 doanh thu thay đổi bao nhiêu % so với tháng liền trước?",
        "2026-06-01",
        None,
        INDEX,
        ("2025-02-01", "2026-05-31"),
        "2026-05-31",
    )
    assert a.expected_type == "percent"
    assert [str(p) for p in a.outside] == ["kỳ trước của tháng 2/2025: 2025-01-01 → 2025-01-31"]
    assert any("HOÀN TOÀN ngoài" in line for line in a.lines())
    inside = analyze("Doanh thu tháng 3/2025 là bao nhiêu", "2026-06-01", None, INDEX, ("2025-02-01", "2026-05-31"))
    assert inside.all_inside and not inside.outside


def test_verifier_warnings():
    a = analyze("Tháng 3/2025 món nào bán chạy nhất?", "2026-06-01", None, INDEX, ("2025-02-01", "2026-05-31"))
    assert check({"type": "string", "answer": "Cà phê sữa (M)"}, a, INDEX) == []
    assert any("không có nguyên văn" in w for w in check({"type": "string", "answer": "Ca phe sua M"}, a, INDEX))
    assert any("cần type 'string'" in w for w in check({"type": "number", "answer": 3}, a, INDEX))
    rev = analyze("Doanh thu tháng 3/2025 là bao nhiêu", "2026-06-01", None, INDEX, ("2025-02-01", "2026-05-31"))
    assert any("bằng 0" in w for w in check({"type": "number", "answer": 0}, rev, INDEX))
    assert any("unanswerable" in w for w in check({"type": "unanswerable"}, rev, INDEX))
    assert check({"type": "abstain"}, rev, INDEX) == []


def test_calibrator():
    assert Calibrator().keep({"vote_share": 0.0})  # no calibration file: keep everything
    c = Calibrator(weights={"vote_share": 4.0, "warnings": -2.0}, bias=-2.0, threshold=0.5)
    assert c.keep({"vote_share": 1.0, "warnings": 0})
    assert not c.keep({"vote_share": 1.0, "warnings": 2})


def test_time_budget_drops_to_one_attempt_when_late():
    b = TimeBudget(deadline=time.time() + 3600, total=100)
    assert b.attempts(3) == 3
    b.start -= 600  # 5 questions took 10 minutes: 95 more cannot fit in the hour left
    for _ in range(5):
        b.record()
    assert b.attempts(3) == 1
    assert b.attempts(1) == 1
