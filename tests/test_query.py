"""Query API of the sales model, the deterministic engine, and the two stages that feed it."""

from datetime import date

import pytest

from datamate import dm
from datamate.agent import planner, templates
from datamate.agent.query import Query, QueryError, execute
from datamate.nlu.periods import extract

JULY, AUG = dm.month(7, 2025), dm.month(8, 2025)
COVERAGE = (date(2025, 7, 1), date(2025, 8, 31))


def test_where_restricts_every_metric(mini_ws):
    S = mini_ws.sales
    assert (S.revenue(JULY), S.order_count(JULY), S.cancelled_count(JULY), S.cancel_rate(JULY)) == (135000, 3, 1, 25.0)
    assert S.qty(JULY) == 9
    tiger = S.where(product="bia tiger lon")  # accents and case do not matter
    assert (tiger.revenue(JULY), tiger.qty(JULY), tiger.order_count(JULY)) == (45000, 3, 2)
    assert tiger.cancelled_count(JULY) == 1  # D3 was an order of Bia Tiger
    assert S.where(product="SP1").qty(JULY) == 3  # the code works too
    assert S.where(channel="Shopee").revenue(JULY) == 40000
    assert S.where(product="Sting dâu 330ml", channel="shopee").revenue(JULY) == 10000
    assert S.where(product=["Bia Tiger lon", "Cà phê sữa (M)"]).revenue(JULY) == 95000
    assert S.where(customer="Chị Lan").revenue() == 100000  # name joined from the customer list by code
    assert S.where(customer="KH2").order_count(JULY) == 1
    assert S.where(walk_in=True).revenue(JULY) == 30000
    assert S.revenue(JULY) == 135000  # where() never changes the original model


def test_where_refuses_unknown_names_instead_of_returning_zero(mini_ws):
    with pytest.raises(QueryError, match="gần giống.*Sting dâu 330ml"):
        mini_ws.sales.where(product="sting dau")
    with pytest.raises(QueryError, match="không lọc được theo 'mau'"):
        mini_ws.sales.where(mau="đỏ")
    with pytest.raises(QueryError, match="không có 'staff'"):
        mini_ws.sales.where(staff="An")


def test_rank_follows_the_rulebook(mini_ws):
    S = mini_ws.sales
    r = S.rank("product", "qty", JULY, k=2)
    assert r["product"].tolist() == ["Sting dâu 330ml", "Bia Tiger lon"] and r["code"].tolist() == ["SP2", "SP1"]
    assert not r.attrs["tie_at_boundary"]
    assert S.rank("product", "revenue", JULY)["product"].tolist() == [
        "Cà phê sữa (M)",
        "Bia Tiger lon",
        "Sting dâu 330ml",
    ]
    assert S.rank("product", "qty", JULY, ascending=True)["product"].tolist()[0] == "Cà phê sữa (M)"
    # walk-ins are nobody; D3 (cancelled) does not count for KH2
    c = S.rank("customer", "revenue", JULY)
    assert c["customer"].tolist() == ["Anh Minh", "Chị Lan"] and c["revenue"].tolist() == [65000, 40000]
    assert S.rank("channel", "orders", JULY)["channel"].tolist() == ["Shopee", "Tại quầy", "Zalo"]  # 1-1-1: by name
    assert S.rank("channel", "orders", JULY, k=1).attrs["tie_at_boundary"]
    assert S.rank("day", "revenue", JULY, k=1)["day"].tolist() == ["2025-07-15"]
    assert S.rank("month", "revenue")["month"].tolist() == ["2025-07", "2025-08"]
    # quantities follow the order's channel: D1 2+1 (Shopee), D2 3 (Tại quầy), D4 2+1 (Zalo)
    assert S.rank("channel", "qty", JULY)["qty"].tolist() == [3.0, 3.0, 3.0]
    assert S.rank("channel", "qty")["qty"].tolist() == [7.0, 3.0, 3.0]  # + D5 (Shopee); D6 was returned
    with pytest.raises(QueryError):
        S.rank("staff")


def test_engine_operations(mini_ws):
    S = mini_ws.sales

    def run(**kw):
        return execute(Query(**kw), S, COVERAGE).answer

    assert run(op="value", metric="revenue", period=JULY) == {"type": "number", "answer": 135000}
    assert run(op="value", metric="cancel_rate", period=JULY) == {"type": "percent", "answer": 25.0}
    assert run(op="value", metric="avg_order_value", period=JULY) == {"type": "number", "answer": 45000}
    assert run(op="value", metric="qty", period=JULY, filters={"product": "Bia Tiger lon"}) == {
        "type": "number",
        "answer": 3,
    }
    assert run(op="change", metric="revenue", period=AUG, base=JULY) == {"type": "percent", "answer": -55.56}
    assert run(op="top", metric="qty", period=JULY, by="product", k=2) == {
        "type": "list",
        "answer": ["Sting dâu 330ml", "Bia Tiger lon"],
    }
    assert run(op="top", metric="revenue", period=JULY, by="customer", k=1) == {"type": "string", "answer": "Anh Minh"}
    assert run(op="best_day", metric="revenue", period=JULY) == {"type": "date", "answer": "2025-07-15"}


def test_engine_declines_or_says_unanswerable(mini_ws):
    S = mini_ws.sales
    # §7: entirely outside the data → unanswerable; a part inside → computed on that part
    outside = execute(Query("value", "revenue", dm.month(3, 2024)), S, COVERAGE).answer
    assert outside["type"] == "unanswerable" and "ngoài khoảng dữ liệu" in outside["reason"]
    assert execute(Query("change", "revenue", JULY, dm.month(6, 2025)), S, COVERAGE).answer["type"] == "unanswerable"
    assert execute(Query("value", "revenue", dm.quarter(3, 2025)), S, COVERAGE).answer["answer"] == 195000
    assert execute(Query("value", "revenue", dm.until(date(2025, 7, 31))), S, COVERAGE).answer["answer"] == 135000
    for bad in (
        Query("top", "orders", JULY, by="channel", k=1),  # tie at the boundary
        Query("top", "qty", JULY, by="product", k=5),  # only three products
        Query("value", "revenue", JULY, filters={"product": "Bia Hà Nội"}),  # not in the data
        Query("value", "avg_order_value", dm.day(date(2025, 7, 20))),  # no order that day
        Query("top", "cancel_rate", JULY, by="product"),
        Query("median", "revenue", JULY),
    ):
        with pytest.raises(QueryError):
            execute(bad, S, COVERAGE)
    with pytest.raises(QueryError):
        execute(Query("value", "revenue", JULY), None, COVERAGE)


@pytest.mark.parametrize(
    "question, as_of, expected",
    [
        ("doanh thu tháng 7/2025 là bao nhiêu", "2025-08-31", {"type": "number", "answer": 135000}),
        ("dthu th7/2025 bnhiu z", "2025-08-31", {"type": "number", "answer": 135000}),
        ("tháng 7/2025 có bao nhiêu đơn bị huỷ", "2025-08-31", {"type": "number", "answer": 1}),
        ("tỉ lệ huỷ đơn tháng 7/2025 là bao nhiêu %", "2025-08-31", {"type": "percent", "answer": 25.0}),
        ("tháng 7/2025 ngày nào doanh thu cao nhất", "2025-08-31", {"type": "date", "answer": "2025-07-15"}),
        (
            "doanh thu tháng 8/2025 thay đổi bao nhiêu % so với tháng trước",
            "2025-08-31",
            {"type": "percent", "answer": -55.56},
        ),
        ("doanh thu tháng 3/2024 là bao nhiêu", "2025-08-31", {"type": "unanswerable"}),
    ],
)
def test_template_stage_answers_plain_questions(mini_ws, question, as_of, expected):
    hit = templates.resolve(question, extract(question, as_of, None))
    assert hit is not None
    answer = execute(hit[1], mini_ws.sales, COVERAGE).answer
    assert {k: answer[k] for k in expected} == expected


@pytest.mark.parametrize(
    "question",
    [
        "Bia Tiger lon tháng 7/2025 bán được mấy lon",  # a product: not plain
        "doanh thu kênh Shopee tháng 7/2025 là bao nhiêu",
        "top 3 mặt hàng bán chạy nhất tháng 7/2025",
        "doanh thu là bao nhiêu",  # no period
    ],
)
def test_template_stage_leaves_everything_else_to_the_model(question):
    assert templates.resolve(question, extract(question, "2025-08-31", None)) is None


def test_planner_spec_parsing():
    known = [dm.this_month(date(2025, 8, 20))]
    assert (
        planner.extract_json('Đây là spec:\n```json\n{"route": "code", "analysis": "nợ {x}"}\n```')["route"] == "code"
    )
    with pytest.raises(planner.SpecError):
        planner.extract_json("không có json")
    q = planner.to_query(
        {"op": "change", "metric": "revenue", "period": {"start": "2025-08-01", "end": "2025-08-20"}, "base": "prev"},
        known,
    )
    assert q.period.label == "tháng này"  # the recognized period, with its meaning, not just two dates
    assert (q.base.start, q.base.end) == (date(2025, 7, 1), date(2025, 7, 20))  # same number of days last month
    q = planner.to_query(
        {
            "op": "top",
            "metric": "qty",
            "by": "product",
            "k": 3,
            "order": "asc",
            "filters": {"channel": "Shopee", "staff": ""},
        },
        [],
    )
    assert (q.by, q.k, q.ascending, q.filters, q.period) == ("product", 3, True, {"channel": "Shopee"}, None)
    for bad in (
        {"op": "sum", "metric": "revenue"},
        {"op": "value", "metric": "profit"},
        {"op": "value", "metric": "revenue", "period": {"start": "2025-08-31", "end": "2025-08-01"}},
        {"op": "value", "metric": "revenue", "period": "tháng 8"},
        {"op": "value", "metric": "revenue", "filters": {"color": "đỏ"}},
        {"op": "top", "metric": "qty", "by": "colour"},
        {"op": "top", "metric": "qty", "by": "product", "k": "ba"},
        {"op": "change", "metric": "revenue"},
    ):
        with pytest.raises(planner.SpecError):
            planner.to_query(bad, [])


class _OneReply:
    def __init__(self, text):
        self.text = text

    def chat(self, messages, enable_thinking=None, max_tokens=None):
        from datamate.agent.llm import Reply

        assert "QUERY_SPEC" in messages[0]["content"] and enable_thinking is False
        return Reply(self.text, None, "stop", {"completion_tokens": 40}, latency_ms=7)


def _plan(mini_ws, reply, expected_type=None, question="x"):
    S = mini_ws.sales
    q = {"id": "q", "business_id": "b", "question": question, "as_of_date": "2025-08-31"}
    dims = planner.shop_dimensions(S, COVERAGE)
    return planner.run_plan(_OneReply(reply), q, S, "RB", dims, [], "Chủ nhật", [JULY], COVERAGE, expected_type)


def test_planner_stage_votes_only_on_valid_specs(mini_ws):
    spec = (
        '{"analysis": "sl bia tiger th7", "route": "sales", "op": "value", "metric": "qty", '
        '"filters": {"product": "Bia Tiger lon"}, "period": {"start": "2025-07-01", "end": "2025-07-31"}}'
    )
    p = _plan(mini_ws, spec, "number")
    assert p.answer == {"type": "number", "answer": 3} and p.route == "sales" and p.llm_ms == 7
    assert p.evidence["query"]["period"].startswith("tháng 7/2025")
    assert _plan(mini_ws, '{"route": "code"}').answer is None
    assert (
        _plan(mini_ws, '{"route": "unanswerable", "analysis": "shop không ghi công nợ"}').answer["type"]
        == "unanswerable"
    )
    # no vote: a name that is not in the data, a type the question does not ask for, a broken reply
    assert "gần giống" in _plan(mini_ws, spec.replace("Bia Tiger lon", "Bia Tiger")).error
    assert "cần 'list'" in _plan(mini_ws, spec, "list").error
    assert _plan(mini_ws, "tôi nghĩ là 3").answer is None
    assert _plan(mini_ws, '{"route": "sql"}').answer is None


def test_planner_prompt_shows_the_shop_dimensions(mini_ws):
    dims = planner.shop_dimensions(mini_ws.sales, COVERAGE)
    assert 'product (3): "Bia Tiger lon"' in dims and '"Shopee"' in dims and "2025-07-01 đến 2025-08-31" in dims
    assert 'route "code"' in planner.shop_dimensions(None, None)
    msgs = planner.build_messages({"question": "q?", "as_of_date": "2025-08-31"}, "LUẬT", dims, ["gợi ý 1"], "Chủ nhật")
    assert msgs[0]["role"] == "system" and "LUẬT" in msgs[1]["content"] and "gợi ý 1" in msgs[1]["content"]
