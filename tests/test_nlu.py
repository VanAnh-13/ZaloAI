from datetime import date

import pytest

from datamate import dm
from datamate.nlu.answer_type import expected_type
from datamate.nlu.entities import EntityIndex
from datamate.nlu.normalize import _telex, rewrite
from datamate.nlu.periods import extract


def spans(periods):
    return [(p.start, p.end) for p in periods]


def test_quarter_is_not_read_as_month():
    ps = extract("dthu quý 2/2026 của cửa hàng là bnhiu", "2026-07-09")
    assert spans(ps)[0] == (date(2026, 4, 1), date(2026, 6, 30))
    assert (date(2026, 2, 1), date(2026, 2, 28)) not in spans(ps)
    assert spans(extract("Trong quý 1 năm 2026 có bao nhiêu đơn", "2026-09-03"))[0] == (
        date(2026, 1, 1),
        date(2026, 3, 31),
    )


def test_explicit_periods_come_first():
    ps = extract("So với tháng liền trước, doanh thu tháng 3 năm 2026 thay đổi bao nhiêu %?", "2026-05-16")
    assert ps[0] == dm.month(3, 2026)
    ps = extract("Doanh thu tháng 8/2026 so với tháng 8/2025 tăng hay giảm bao nhiêu %?", "2026-09-20")
    assert ps[:2] == [dm.month(8, 2026), dm.month(8, 2025)]


@pytest.mark.parametrize(
    "q, as_of, expected",
    [
        ("Tuần rồi bán được bao nhiêu", "2026-03-04", dm.last_week(date(2026, 3, 4))),
        ("8/26 tiền bán ca chiều là bnhiu?", "2026-09-19", dm.month(8, 2026)),
        ("tính đến hết ngày 12/07/2025, khách nào nợ nhiều nhất", "2026-01-01", dm.until(date(2025, 7, 12))),
        ("30 ngày gần nhất bán được bao nhiêu tiền", "2026-05-10", dm.last_n_days(30, date(2026, 5, 10))),
    ],
)
def test_extract_cases(q, as_of, expected):
    assert expected in extract(q, as_of)


def test_rewrite_and_telex():
    out = rewrite("be oi th10/2025 top 3 mon ban nhiu nhut la j z ha")
    assert "tháng 10/2025" in out and "nhất" in out and "gì" in out
    assert _telex("ddieenj") == "điện" and _telex("nguwowif") == "người"
    assert rewrite("Hồng trà sữa bán mấy ly") == "Hồng trà sữa bán mấy ly"  # nothing to fix: unchanged


@pytest.mark.parametrize(
    "q, t",
    [
        ("Tháng 12/2025 ngày nào cửa hàng bán được nhiều tiền nhất?", "date"),
        ("trong các tháng 11/2025, 12/2025 và 1/2026, cửa hàng có dthu cao nhất vào tháng nào", "date"),
        ("tỉ lệ huỷ đơn tháng 6/2025 là nhiêu %", "percent"),
        ("Doanh thu có tăng liên tục qua 3 tháng 12/2025, 1/2026, 2/2026 không?", "boolean"),
        ("3 mặt hàng bán chạy nhất tuần này là những mặt hàng nào?", "list"),
        ("tính đến hết ngày 12/07/2025, 5 khách nào đang nợ nhiều nhất, mỗi người còn nợ bao nhiu", "table"),
        ("liet ke 3 khach no nhieu nhat tinh den het ngay 28/06/2026, kem so tien con no", "table"),
        ("sp bán dc nhiều nhất tuần rồi là gì", "string"),
        ("Tháng 6/2025 cửa hàng bán được tổng cộng bao nhiêu tiền?", "number"),
    ],
)
def test_expected_type(q, t):
    assert expected_type(q)[0] == t


def test_entity_index():
    idx = EntityIndex(
        [("Otrivin 0,1% xịt mũi chai 10ml", "SP", "Tên"), ("Bia Tiger lon", "SP", "Tên"), ("Bia", "SP", "Nhóm")]
    )
    found = [v for v, _, _ in idx.mentioned("Tháng 7/2025 Otrivin 0,1% xit mui chai 10ml bán mấy chai?")]
    assert found == ["Otrivin 0,1% xịt mũi chai 10ml"]
    assert "bia tiger lon" in idx and "Bia Tiger" not in idx
