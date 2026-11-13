"""A tiny synthetic split with hand-checked answers, shared by the tests of the new flow.

July 2025: D1 (40.000, Shopee, KH1), D2 (30.000, walk-in), D3 cancelled, D4 (65.000, Zalo, KH2)
  → revenue 135.000 · 3 valid orders · 1 of 4 cancelled (25 %) · best day 15/07
  → quantities: Sting 4, Bia Tiger 3, Cà phê 2
August 2025: D5 (60.000), D6 returned → revenue 60.000 (−55,56 % against July)
"""

import json

import pytest

README = """# Dữ liệu biz_m01

Tạp hoá nhỏ, xuất từ phần mềm bán hàng.

## File
- `don_hang.csv` (csv)
- `chi_tiet.csv` (csv)
- `khach_hang.csv` (csv)

## Cột trong `don_hang.csv`
- **Mã HĐ**: mã đơn
- **Ngày**: ngày bán
- **Trạng thái**: trạng thái đơn
- **Mã KH**: mã khách (trống nếu khách lẻ)
- **Kênh**: kênh bán

## Cột trong `chi_tiet.csv`
- **Mã HĐ**: mã đơn
- **Mã hàng**: mã hàng
- **Tên hàng**: tên hàng
- **SL**: số lượng
- **Thành tiền**: thành tiền sau giảm

## Cột trong `khach_hang.csv`
- **Mã KH**: mã khách
- **Tên khách**: tên khách

## Quy ước
- Dữ liệu có đến hết 31/08/2025.
"""

ORDERS = """Mã HĐ,Ngày,Trạng thái,Mã KH,Kênh
D1,01/07/2025,Hoàn thành,KH1,Shopee
D2,02/07/2025,Hoàn thành,,Tại quầy
D3,03/07/2025,Đã hủy,KH2,Shopee
D4,15/07/2025,Hoàn thành,KH2,Zalo
D5,01/08/2025,Hoàn thành,KH1,Shopee
D6,02/08/2025,Trả hàng,KH1,Tại quầy
"""

LINES = """Mã HĐ,Mã hàng,Tên hàng,SL,Thành tiền
D1,SP1,Bia Tiger lon,2,30000
D1,SP2,Sting dâu 330ml,1,10000
D2,SP2,Sting dâu 330ml,3,30000
D3,SP1,Bia Tiger lon,5,75000
D4,SP3,Cà phê sữa (M),2,50000
D4,SP1,Bia Tiger lon,1,15000
D5,SP1,Bia Tiger lon,4,60000
D6,SP3,Cà phê sữa (M),1,25000
"""

CUSTOMERS = """Mã KH,Tên khách
KH1,Chị Lan
KH2,Anh Minh
"""

QUESTIONS = [
    ("q1", "doanh thu tháng 7/2025 là bao nhiêu", {"type": "number", "answer": 135000}),
    (
        "q2",
        # worded so that stage 1 does not take them: the planner and the code agent have to
        "top 2 mặt hàng bán chạy nhất tháng 7/2025 xét theo số lượng",
        {"type": "list", "answer": ["Sting dâu 330ml", "Bia Tiger lon"]},
    ),
    ("q3", "Bia Tiger lon tháng 7/2025 bán được mấy lon, không tính đơn huỷ", {"type": "number", "answer": 3}),
    ("q4", "khách nào còn nợ nhiều nhất", {"type": "unanswerable"}),
    ("q5", "doanh thu tháng 3/2024 là bao nhiêu", {"type": "unanswerable"}),
]


@pytest.fixture
def mini_split(tmp_path):
    split = tmp_path / "split"
    biz = split / "data" / "biz_m01"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text(README, encoding="utf-8")
    (biz / "don_hang.csv").write_text(ORDERS, encoding="utf-8")
    (biz / "chi_tiet.csv").write_text(LINES, encoding="utf-8")
    (biz / "khach_hang.csv").write_text(CUSTOMERS, encoding="utf-8")
    (split / "rulebook.md").write_text(
        "# Rulebook\n## 1\nĐơn huỷ và đơn trả hàng không tính doanh thu.\n", encoding="utf-8"
    )
    (split / "calendar.csv").write_text(
        "date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n2025-07-01,2,7,6,2025,false,\n",
        encoding="utf-8",
    )
    qs = [{"id": i, "business_id": "biz_m01", "question": t, "as_of_date": "2025-08-31"} for i, t, _ in QUESTIONS]
    (split / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")
    (split / "answers.jsonl").write_text(
        "\n".join(json.dumps({"id": i, **g}, ensure_ascii=False) for i, _, g in QUESTIONS), encoding="utf-8"
    )
    return split


@pytest.fixture
def mini_ws(mini_split):
    from datamate.workspace import build_workspace

    return build_workspace(mini_split / "data" / "biz_m01")


@pytest.fixture
def mini_ref(mini_split, tmp_path):
    from datamate.workspace import prepare_many

    return prepare_many(mini_split, ["biz_m01"], tmp_path / "ws", 1)["biz_m01"]
