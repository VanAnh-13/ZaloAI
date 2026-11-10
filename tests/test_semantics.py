"""Sales model on small synthetic shops, built through the real pipeline
(CSV -> ingest -> §8 cleaning -> SalesView)."""

from datetime import date

import pytest

from datamate import dm
from datamate.semantics.roles import fold_name
from datamate.semantics.status import CANCELLED, COMPLETED, RETURNED, SHIPPING, canonical_status, is_walk_in_name
from datamate.workspace import build_workspace


def make_shop(tmp_path, readme: str, files: dict[str, str]):
    d = tmp_path / "shop_x"
    d.mkdir()
    (d / "README.md").write_text(readme, encoding="utf-8")
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    return build_workspace(d)


HEADER_LINES_README = """# Dữ liệu shop_x

## File
- `don_hang.csv` (csv)
- `chi_tiet.csv` (csv)

## Cột trong `don_hang.csv`
- **Mã HĐ**: mã đơn
- **Ngày**: ngày bán
- **Trạng thái**: trạng thái đơn
- **Mã KH**: mã khách (trống nếu khách lẻ)

## Cột trong `chi_tiet.csv`
- **Mã HĐ**: mã đơn
- **Tên hàng**: tên hàng
- **SL**: số lượng
- **Thành tiền**: thành tiền sau giảm

## Quy ước
- Dữ liệu có đến hết 31/03/2026.
"""

ORDERS = """Mã HĐ,Ngày,Trạng thái,Mã KH
D1,01/03/2026,Hoàn thành,KH1
D2,02/03/2026,Đã hủy,KH2
D3,03/03/2026,Hoàn thành,
D4,05/03/2026,Đang vận chuyển,KH1
D5,28/02/2026,Hoàn thành,KH2
"""

LINES = """Mã HĐ,Tên hàng,SL,Thành tiền
D1,Bia Tiger,2,30.000đ
D1,Sting dâu,1,10k
D1,Sting dâu,1,10k
D2,Bia Tiger,1,15000
D3,Bia Tiger,3,45000
D3,Bia Tiger,-1,-15000
D4,Sting dâu,5,50000
D5,Bia Tiger,1,15000
"""


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Đã hoàn thành", COMPLETED),
        ("xong", COMPLETED),
        ("completed", COMPLETED),
        ("đã lấy", COMPLETED),
        ("Đã hủy", CANCELLED),
        ("hủy đơn", CANCELLED),
        ("khách hủy", CANCELLED),
        ("Không giao được", CANCELLED),
        ("Đổi trả", RETURNED),
        ("hàng trả về", RETURNED),
        ("Hoàn trả", RETURNED),
        ("returned", RETURNED),
        ("Đang vận chuyển", SHIPPING),  # "chuyen" contains "huy": must not be read as cancelled
        ("Đang giao hàng", SHIPPING),
        ("Đang xử lý", SHIPPING),
        ("???", None),
        (None, None),
    ],
)
def test_canonical_status(raw, expected):
    assert canonical_status(raw) == expected


def test_walk_in_and_fold_name():
    assert all(is_walk_in_name(v) for v in ("Khách lẻ", "kl", "lẻ", "khách lẻ", None, ""))
    assert not is_walk_in_name("Lê Văn An")
    assert fold_name("Mã hóa đơn") == "ma hoa don"
    assert fold_name("NgayBan") == "ngay ban"
    assert fold_name("TenKH") == "ten kh"


def test_sales_view_header_and_lines(tmp_path):
    ws = make_shop(tmp_path, HEADER_LINES_README, {"don_hang.csv": ORDERS, "chi_tiet.csv": LINES})
    S = ws.sales
    assert S is not None
    assert S.sources["status"] == "don_hang.Trạng thái" and S.sources["amount"] == "chi_tiet.Thành tiền"
    o = S.orders.set_index("order_id")
    assert o.loc["D3", "status"] == RETURNED  # a partial return excludes the whole order
    assert o.loc["D4", "status"] == SHIPPING and bool(o.loc["D4", "valid"])
    assert o.loc["D1", "revenue"] == 40000  # the duplicated line of D1 counts once (§8)
    assert bool(o.loc["D3", "walk_in"]) and o.loc["D1", "customer_key"] == "KH1"
    march = dm.month(3, 2026)
    assert S.revenue(march) == 90000  # D1 + D4
    assert S.order_count(march) == 2
    assert S.cancelled_count(march) == 1
    assert S.cancel_rate(march) == 25.0  # 1 of the 4 orders placed in March, any status
    assert S.avg_order_value(march) == 45000
    assert S.revenue(dm.month(2, 2026)) == 15000
    assert S.coverage() == (date(2026, 2, 28), date(2026, 3, 5))
    assert "Mô hình bán hàng chuẩn hoá `S`" in ws.card


def test_sales_view_without_order_ids(tmp_path):
    readme = (
        "# Dữ liệu shop_x\n\n## File\n- `so.csv` (csv)\n\n## Cột trong `so.csv`\n- **Ngày**: ngày bán\n"
        "- **Mặt hàng**: tên hàng\n- **SL**: số lượng\n- **Tiền**: thành tiền sau giảm\n"
    )
    book = "Ngày,Mặt hàng,SL,Tiền\n01/03/2026,Gạch,2,20000\n01/03/2026,Gạch,2,20000\n02/03/2026,Cát,1,1tr5\n"
    S = make_shop(tmp_path, readme, {"so.csv": book}).sales
    assert S.order_count() == 3  # no order id: every row is one sale, identical rows both count
    assert S.revenue() == 1_540_000


def test_sales_view_invoice_only(tmp_path):
    readme = (
        "# Dữ liệu shop_x\n\n## File\n- `hd.csv` (csv)\n\n## Cột trong `hd.csv`\n- **Mã bill**: mã đơn\n"
        "- **Ngày giao dịch**: ngày bán\n"
        "- **Tien hang**: tiền hàng của đơn: cộng thành tiền các mặt hàng, trước giảm giá của đơn\n"
        "- **TongTien**: số khách phải trả cho cả đơn (đã tính giảm giá của đơn và phí giao)\n"
        "- **Status**: trạng thái đơn\n"
    )
    invoices = (
        "Mã bill,Ngày giao dịch,Tien hang,TongTien,Status\nB1,01/03/2026,100000,95000,Hoàn thành\n"
        "B1,,100000,95000,\nB2,02/03/2026,50000,60000,khách hủy\n"
    )
    S = make_shop(tmp_path, readme, {"hd.csv": invoices}).sales
    assert S.sources["amount"] == "hd.Tien hang"  # goods total, not the amount paid (discount, shipping)
    assert S.revenue(dm.month(3, 2026)) == 100000  # the repeated row of B1 counts once
    assert S.cancel_rate(dm.month(3, 2026)) == 50.0
