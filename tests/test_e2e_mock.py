"""End to end on a tiny synthetic split with the mock LLM: ingest → prompt → sandbox →
submission.jsonl + logs. No GPU and no real data needed."""

import json

from datamate.run import main as run_main
from eval.scorer import evaluate, read_jsonl

README = """# Dữ liệu biz_t01

Sổ bán hàng ghi tay.

## File
- `ban_hang.csv` (csv)

## Cột trong `ban_hang.csv`
- **Ngày**: ngày bán
- **Mã đơn**: mã đơn
- **Tên hàng**: tên hàng
- **SL**: số lượng
- **Thành tiền**: thành tiền sau giảm
- **Trạng thái**: trạng thái đơn

## Quy ước
- Dữ liệu có đến hết 03/01/2026.
"""

CSV = """Sổ bán hàng tháng 1
Ngày,Mã đơn,Tên hàng,SL,Thành tiền,Trạng thái
01/01/2026,D1,Bia Tiger lon,2,"30.000đ",xong
,D1,Sting dâu 330ml,1 chai,10k,
02/01/2026,D2,Bia Tiger lon,1,15000,hủy
Tổng tháng,,,,55000,
"""


def test_mock_pipeline(tmp_path):
    split = tmp_path / "split"
    biz = split / "data" / "biz_t01"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text(README, encoding="utf-8")
    (biz / "ban_hang.csv").write_text(CSV, encoding="utf-8")
    (split / "rulebook.md").write_text("# Rulebook\nĐơn huỷ không tính doanh thu.", encoding="utf-8")
    (split / "calendar.csv").write_text(
        "date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n2026-01-01,4,13,11,2025,false,\n",
        encoding="utf-8",
    )
    qs = [
        {"id": "q1", "business_id": "biz_t01", "question": "doanh thu hom nay bnhiu", "as_of_date": "2026-01-01"},
        {"id": "q2", "business_id": "biz_t01", "question": "Bia Tiger lon bán mấy lon", "as_of_date": "2026-01-03"},
    ]
    (split / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")

    out = tmp_path / "out"
    assert run_main(["--data", str(split), "--out", str(out), "--ws", str(tmp_path / "ws"), "--mock"]) == 0

    sub = read_jsonl(out / "submission.jsonl")
    assert sorted(r["id"] for r in sub) == ["q1", "q2"]
    assert all(r["type"] == "abstain" for r in sub)
    logs = read_jsonl(out / "logs" / "logs.jsonl")
    for rec in logs:  # fields required by the Docker guide §3.3
        assert {"id", "answer", "elapsed_ms", "thinking", "code", "output"} <= rec.keys()
        assert isinstance(rec["elapsed_ms"], int) and rec["elapsed_ms"] >= 0
        assert rec["code"] and "T.items()" in rec["code"][0]
        # the sandbox saw the cleaned table (title row skipped, footer row dropped) and the sales
        # model: D1 = 30.000 + 10.000, the cancelled D2 does not count
        assert "ban_hang (3, " in rec["output"][0] and "2026-01-03" in rec["output"][0]
        assert "S.revenue: 40000" in rec["output"][0]
    golds = {"q1": {"type": "number", "answer": 40000}, "q2": {"type": "number", "answer": 2}}
    assert evaluate({r["id"]: r for r in sub}, golds, ["q1", "q2"])["pa"] == 0.0
