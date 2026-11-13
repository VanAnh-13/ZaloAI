"""The cascade end to end on the tiny split, with a scripted model: which stages run, how the
votes decide, what is logged. No GPU, no network."""

import json
import re

from datamate.agent.confidence import Calibrator
from datamate.agent.examples import Example, ExampleBank
from datamate.agent.llm import Reply
from datamate.agent.prompts import VARIANTS
from datamate.config import load_settings
from datamate.pipeline import Pipeline
from eval.scorer import evaluate, read_jsonl

JULY = '{"start": "2025-07-01", "end": "2025-07-31"}'
PLANS = {
    "q1": '{"analysis": "doanh thu tháng 7", "route": "sales", "op": "value", "metric": "revenue", "period": %s}'
    % JULY,
    "q2": '{"route": "sales", "op": "top", "metric": "qty", "by": "product", "k": 2, "period": %s}' % JULY,
    # a wrong reading: revenue instead of quantity
    "q3": '{"route": "sales", "op": "value", "metric": "revenue", "filters": {"product": "Bia Tiger lon"}, '
    '"period": %s}' % JULY,
    "q4": '{"analysis": "shop không ghi công nợ", "route": "unanswerable"}',
    "q5": '{"route": "sales", "op": "value", "metric": "revenue", '
    '"period": {"start": "2024-03-01", "end": "2024-03-31"}}',
}
TOP2 = "p = dm.month(7, 2025)\nr = S.rank('product', 'qty', p, k=2)\nr"
CODES = {  # question id -> attempt -> code blocks, one per step
    "q2": [[TOP2, "answer('list', r['product'])"]],  # second step relies on the session keeping `r`
    "q3": [
        ["answer('number', S.where(product='Bia Tiger lon').qty(dm.month(7, 2025)))"],
        ["L = S.valid_lines(dm.month(7, 2025))\nanswer('number', L[L['product'] == 'Bia Tiger lon']['qty'].sum())"],
    ],
    "q4": [["answer('unanswerable', reason='không có sổ công nợ')"]],
}


class Scripted:
    """Plays the planner and the code agent from the tables above; counts the calls."""

    def __init__(self, ids: dict, plans=PLANS, codes=CODES):
        self.ids, self.plans, self.codes = ids, plans, codes
        self.calls: list[tuple[str, str]] = []
        self.prompts: list[str] = []

    def wait_ready(self, timeout_s):
        return True

    def chat(self, messages, enable_thinking=None, max_tokens=None):
        first_user = messages[1]["content"]
        qid = self.ids[re.search(r'Câu hỏi: "(.*)"', first_user).group(1)]
        if "QUERY_SPEC" in messages[0]["content"]:
            self.calls.append((qid, "plan"))
            return Reply(self.plans.get(qid, '{"route": "code"}'), None, "stop", {"completion_tokens": 50}, 5)
        self.calls.append((qid, "code"))
        self.prompts.append(first_user)
        attempt = max(i for i, note in enumerate(VARIANTS) if note in first_user)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        blocks = self.codes.get(qid) or [[]]
        blocks = blocks[attempt] if attempt < len(blocks) else ["answer('abstain')"]
        code = blocks[turn] if turn < len(blocks) else "answer('abstain')"
        return Reply(f"```python\n{code}\n```", f"nghĩ {qid}/{attempt}/{turn}", "stop", {"completion_tokens": 200}, 20)


def run(split, tmp_path, name="out", client=None, calibrator=None, examples=None, **overrides):
    qs = read_jsonl(split / "questions.jsonl")
    client = client or Scripted({q["question"]: q["id"] for q in qs})
    settings = load_settings(ws_dir=str(tmp_path / "ws"), concurrency=2, **overrides)
    pipe = Pipeline(settings, split, tmp_path / name, client, calibrator=calibrator or Calibrator())
    if examples is not None:
        resources = pipe._resources
        pipe._resources = lambda: _with_examples(resources(), examples)
    pipe.run(qs)
    sub = {r["id"]: r for r in read_jsonl(tmp_path / name / "submission.jsonl")}
    logs = {r["id"]: r for r in read_jsonl(tmp_path / name / "logs" / "logs.jsonl")}
    return sub, logs, client


def _with_examples(res, examples):
    res.examples = examples
    return res


def test_balanced_effort_spends_model_time_only_where_votes_are_missing(mini_split, tmp_path):
    sub, logs, client = run(mini_split, tmp_path)
    gold = {g["id"]: g for g in read_jsonl(mini_split / "answers.jsonl")}
    assert evaluate(sub, gold, sorted(gold))["pa"] == 1.0

    def sources(i):
        return [v["source"] for v in logs[i]["votes"]]

    # q1 plain question: template + planner agree → no code attempt at all
    assert sources("q1") == ["template", "plan"] and logs["q1"]["code"] is None
    assert [c for c in client.calls if c[0] == "q1"] == [("q1", "plan")]
    # q2: the planner's spec and the code agent's program agree after one attempt of two steps
    assert sources("q2") == ["plan", "code"] and len(logs["q2"]["code"]) == 2
    assert sub["q2"]["answer"] == ["Sting dâu 330ml", "Bia Tiger lon"]
    # q3: the planner misread the question (45000 đ), two code attempts outvote it
    assert sources("q3") == ["plan", "code", "code"] and sub["q3"]["answer"] == 3
    assert logs["q3"]["votes"][0]["answer"] == {"type": "number", "answer": 45000} and logs["q3"]["n_votes"] == 2
    assert logs["q3"]["confidence_features"]["plan_agrees"] == -1.0
    # q4: both the planner and the code agent say the shop does not record debts
    assert sources("q4") == ["plan", "code"] and sub["q4"]["type"] == "unanswerable"
    # q5: the period lies before the data — said twice by the engine, never by a code attempt
    assert sources("q5") == ["template", "plan"] and sub["q5"]["type"] == "unanswerable"
    assert logs["q5"]["confidence_features"]["periods_outside"] == 1.0

    # Docker guide §3.3 fields; thinking lists every model call in order (None for the planner call)
    for rec in logs.values():
        assert {"id", "answer", "elapsed_ms", "thinking", "code", "output"} <= rec.keys()
    assert logs["q3"]["thinking"] == [None, "nghĩ q3/0/0", "nghĩ q3/1/0"]
    assert logs["q1"]["thinking"] is None and logs["q1"]["timing"]["llm_calls"] == 1
    summary = json.loads((tmp_path / "out" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["answered_by"] == {"plan+template": 2, "code+plan": 2, "code": 1}
    assert summary["llm_calls"] == len(client.calls) == 5 + 2 + 2 + 1  # 5 plans, q2 two steps, q3 two attempts, q4


def test_fast_effort_takes_the_first_vote(mini_split, tmp_path):
    sub, logs, client = run(mini_split, tmp_path, effort="fast")
    assert [c for c in client.calls if c[0] in ("q1", "q5")] == []  # answered by the template stage alone
    assert sub["q1"]["answer"] == 135000 and sub["q5"]["type"] == "unanswerable"
    assert all(kind == "plan" for _, kind in client.calls)  # every other question stopped at the planner
    assert sub["q3"]["answer"] == 45000  # …including the one the planner got wrong: fast is for development


def test_no_agreement_means_abstain(mini_split, tmp_path):
    codes = {
        "q2": [
            ["answer('list', ['Bia Tiger lon', 'Sting dâu 330ml'])"],
            ["answer('list', ['Cà phê sữa (M)', 'Bia Tiger lon'])"],
        ],
        "q3": [["answer('abstain')"], ["answer('abstain')"], ["answer('number', 3)"]],
    }
    plans = {"q2": PLANS["q2"]}
    qs = read_jsonl(mini_split / "questions.jsonl")
    client = Scripted({q["question"]: q["id"] for q in qs}, plans, codes)
    sub, logs, _ = run(mini_split, tmp_path, client=client)
    # q2: planner, attempt 1 and attempt 2 give three different lists; attempt 3 abstains
    assert sub["q2"]["type"] == "abstain" and logs["q2"]["n_votes"] == 1
    assert logs["q2"]["voted"]["answer"] == ["Sting dâu 330ml", "Bia Tiger lon"]  # the earliest vote leads
    assert logs["q2"]["confidence_features"]["distinct_answers"] == 3.0
    assert sum(1 for st in logs["q2"]["stages"] if st["stage"] == "code") == 3
    # q3: nobody voted after two attempts → a third one could not reach two votes, so it is not run
    assert sub["q3"]["type"] == "abstain" and logs["q3"]["votes"] == []
    assert sum(1 for st in logs["q3"]["stages"] if st["stage"] == "code") == 2
    # q1: the planner handed the question over and the code agent abstained → one template vote only
    assert sub["q1"]["type"] == "abstain" and logs["q1"]["voted"] == {"type": "number", "answer": 135000}


def test_fitted_calibration_replaces_the_two_votes_rule(mini_split, tmp_path):
    qs = read_jsonl(mini_split / "questions.jsonl")
    client = Scripted({q["question"]: q["id"] for q in qs}, {}, {})  # planner: always "code"; code agent: abstains
    keep_all = Calibrator(weights={"votes": 1.0}, bias=5.0, threshold=0.5)
    sub, logs, _ = run(mini_split, tmp_path, "keep", client=client, calibrator=keep_all)
    assert sub["q1"]["answer"] == 135000 and logs["q1"]["n_votes"] == 1 and logs["q1"]["p_correct"] > 0.99
    drop_all = Calibrator(weights={"votes": 1.0}, bias=-9.0, threshold=0.5)
    sub, _, _ = run(mini_split, tmp_path, "drop", calibrator=drop_all)
    assert all(r["type"] == "abstain" for r in sub.values())


def test_stages_can_be_switched_off(mini_split, tmp_path):
    sub, logs, client = run(mini_split, tmp_path, use_templates=False, use_planner=False)
    assert all(kind == "code" for _, kind in client.calls)
    assert [v["source"] for v in logs["q3"]["votes"]] == ["code", "code"] and sub["q3"]["answer"] == 3
    assert sub["q1"]["type"] == "abstain"  # the script has no program for q1


def test_worked_examples_are_shown_to_the_code_agent(mini_split, tmp_path):
    bank = ExampleBank(
        [
            Example(
                "e1",
                "biz_other",
                "top 3 mặt hàng bán chạy nhất tháng 5/2025",
                "2025-06-30",
                "list",
                "answer('list', ...)",
            ),
            Example("e2", "biz_m01", "top 5 mặt hàng bán chạy nhất tháng 6/2025", "2025-06-30", "list", "# cùng shop"),
        ]
    )
    _, logs, client = run(mini_split, tmp_path, "a", examples=bank, use_planner=False)
    prompt = next(p for p in client.prompts if "top 2 mặt hàng" in p.split("# CÂU HỎI")[1])
    assert "Ví dụ đã giải đúng" in prompt and "tháng 5/2025" in prompt
    assert prompt.index("Ví dụ đã giải đúng") > prompt.index("## Bảng dữ liệu gốc")  # after the cached prefix
    assert [e["id"] for e in logs["q2"]["examples"]] == ["e1"]  # same wording once: the best match only
    assert logs["q4"]["examples"] is None  # nothing similar enough about debts
    # development on train: never show an example from the shop being asked about
    _, logs, _ = run(
        mini_split, tmp_path, "b", examples=ExampleBank(bank.examples[1:]), use_planner=False, examples_holdout=True
    )
    assert logs["q2"]["examples"] is None
