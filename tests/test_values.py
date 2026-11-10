import pandas as pd
import pytest

from datamate.ingest.loaders import is_footer_row
from datamate.ingest.values import month_year_from_name, parse_date, parse_number, split_number_unit


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("55.000đ", 55000),
        ("49k", 49000),
        ("38,000", 38000),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("(1.000)", -1000),
        ("-30.500đ", -30500),
        ("0.5", 0.5),
        ("12,5", 12.5),
        ("1.000.000", 1_000_000),
        ("1tr900", 1_900_000),
        ("2tr045", 2_045_000),
        ("1tr5", 1_500_000),
        ("1 triệu 45 nghìn", 1_045_000),
        ("1k5", 1500),
        ("1,5tr", 1_500_000),
        ("1824,235tr", 1_824_235_000),
        ("5 kg", 5),
        ("2 xe 4m3", 2),
        (12, 12.0),
        (3.5, 3.5),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", None, "abc", "Bia Tiger", True, "1.2.3", float("nan")])
def test_parse_number_rejects(raw):
    assert parse_number(raw) is None


@pytest.mark.parametrize(
    "raw, unit",
    [
        ("2 xe 4m3", "xe 4m3"),
        ("1 bao 25kg", "bao 25kg"),
        ("2 hộp 30", "hộp 30"),
        ("3 Thùng", "Thùng"),
        ("55.000đ", None),
        ("49k", None),
        (7, None),
    ],
)
def test_split_number_unit(raw, unit):
    assert split_number_unit(raw)[1] == unit


@pytest.mark.parametrize(
    "raw, kw, expected",
    [
        ("01/02/2025", {}, "2025-02-01"),
        ("1 thg 6 2025", {}, "2025-06-01"),
        ("2025-02-01 06:00:21", {}, "2025-02-01 06:00:21"),
        ("14:30 05/03/2026", {}, "2026-03-05 14:30:00"),
        ("ngày 5 tháng 3 năm 2025", {}, "2025-03-05"),
        ("05/03", {"year_hint": 2026}, "2026-03-05"),
        (45689, {"allow_serial": True}, "2025-02-01"),
        ("45689", {"allow_serial": True}, "2025-02-01"),
    ],
)
def test_parse_date(raw, kw, expected):
    assert parse_date(raw, **kw) == pd.Timestamp(expected)


def test_parse_date_serial_needs_permission():
    assert parse_date(45689) is None
    assert parse_date("31/02/2025") is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("T01_2025", (1, 2025)),
        ("04.2025", (4, 2025)),
        ("Thang 02 2025", (2, 2025)),
        ("BanHang_T10_2025", (10, 2025)),
        ("SoBanHang_2026", (None, 2026)),
        ("Sheet1", (None, None)),
    ],
)
def test_month_year_from_name(name, expected):
    assert month_year_from_name(name) == expected


def test_footer_rows():
    assert is_footer_row(["Tổng tháng", None, None, 1_500_000, None, None], 6)
    assert is_footer_row([None, "Ghi chú: tiền tính bằng đồng", None, None, None, None], 6)
    assert not is_footer_row(["Công ty Minh Long", "ORD1", "Xi măng", 2, 90000, "Đã xong"], 6)
    assert not is_footer_row(["Tổng hợp vitamin C", "ORD2", 1, 50000, "xong", "lẻ"], 6)
