"""Solve one question with a cascade of voters, cheapest first.

    stage 1  template   regular expressions → query engine            no LLM call
    stage 2  planner    LLM fills a JSON spec → query engine          one short call
    stage 3  code       LLM writes Python in a sandbox session        several long calls, up to N attempts

Every stage that produces an answer casts one vote. With +1 / −1 / 0 scoring an answer is only
worth submitting when it is probably right, and two independent ways of reading the question
that reach the same value are the strongest signal there is:

    effort "balanced"  stop as soon as two votes agree; no agreement after every stage → abstain
    effort "fast"      the first vote wins (development runs, or when the run is behind schedule)

So plain questions cost no LLM time at all, most sales questions cost one short call plus one
code attempt, and extra code attempts are spent only on the questions that are actually
uncertain. A fitted calibration (eval/calibrate.py) replaces the two-votes rule by P(correct).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from datamate.agent import planner, templates
from datamate.agent.answer import AnswerError, canonical, normalize
from datamate.agent.confidence import Calibrator, features
from datamate.agent.examples import ExampleBank, render
from datamate.agent.hints import analyze
from datamate.agent.loop import Attempt, run_attempt
from datamate.agent.prompts import build_messages
from datamate.agent.query import QueryError, execute
from datamate.agent.sandbox import ExecContext, Sandbox
from datamate.agent.verify import check
from datamate.config import Settings
from datamate.core.periods import vn_weekday
from datamate.nlu.entities import EntityIndex
from datamate.semantics.sales import SalesView
from datamate.workspace import WorkspaceRef

ABSTAIN = {"type": "abstain"}
TEMPLATE, PLAN, CODE = "template", "plan", "code"


@dataclass
class Resources:
    """Run-level dependencies shared by every question."""

    client: object  # ChatClient or MockClient
    sandbox: Sandbox
    settings: Settings
    rulebook: str
    calendar_path: str | None
    cal: pd.DataFrame | None
    calibrator: Calibrator
    examples: ExampleBank = field(default_factory=lambda: ExampleBank([]))
    rulebook_path: str | None = None


@dataclass
class Shop:
    ref: WorkspaceRef
    index: EntityIndex
    sales: SalesView | None = None  # in-process copy for the query engine (stages 1 and 2)
    dimensions: str = ""  # planner view of the shop
    names: templates.Names | None = None  # the shop's product / group / channel names, for stage 1


@dataclass
class Vote:
    source: str  # template | plan | code
    answer: dict  # normalized submission object without id


def tally(votes: list[Vote]) -> list[tuple[dict, int]]:
    """Distinct answers with their number of votes, most voted first (earlier vote first on equality)."""
    order: list[str] = []
    counts: Counter = Counter()
    first: dict[str, dict] = {}
    for v in votes:
        k = canonical(v.answer)
        if k not in first:
            first[k] = v.answer
            order.append(k)
        counts[k] += 1
    ranked = sorted(order, key=lambda k: (-counts[k], order.index(k)))
    return [(first[k], counts[k]) for k in ranked]


def agreed(votes: list[Vote], need: int = 2) -> dict | None:
    """The answer that `need` votes share, provided no other answer has as many."""
    t = tally(votes)
    if t and t[0][1] >= need and (len(t) == 1 or t[1][1] < t[0][1]):
        return t[0][0]
    return None


def _engine_vote(payload: dict) -> dict | None:
    """Engine result -> the same normalized form as an answer(...) call of the code agent."""
    try:
        if payload["type"] == "unanswerable":
            return normalize({"type": "unanswerable", "reason": payload.get("reason")})
        return normalize({"type": payload["type"], "value": payload["answer"]})
    except AnswerError:
        return None


def solve(q: dict, shop: Shop, res: Resources, deadline: float, effort: str | None = None) -> tuple[dict, dict]:
    """Returns (submission record, log record)."""
    t0 = time.monotonic()
    s = res.settings
    effort = effort or s.effort
    need = 1 if effort == "fast" else 2
    analysis = analyze(q["question"], q["as_of_date"], res.cal, shop.index, shop.ref.coverage, shop.ref.data_end)
    hint_lines = analysis.lines()
    votes: list[Vote] = []
    stages: list[dict] = []
    thinking: list[str | None] = []
    usage: Counter = Counter()
    timing = Counter()

    def done() -> bool:
        return agreed(votes, need) is not None

    # ---- stage 1: template (no LLM) ------------------------------------------------------
    if s.use_templates and shop.sales is not None:
        st = {"stage": TEMPLATE, "answer": None}
        try:
            # exactly the call eval/oracle.py validates against the train answers
            hit = templates.resolve(q["question"], q["as_of_date"], res.cal, shop.names)
            if hit:
                kind, query = hit
                st.update(intent=kind, query=query.describe())
                r = execute(query, shop.sales, analysis.coverage)
                st["answer"] = _engine_vote(r.answer)
                st["evidence"] = r.evidence
        except QueryError as e:
            st["error"] = str(e)
        except Exception as e:  # never let a cheap stage break the question
            st["error"] = f"{type(e).__name__}: {e}"
        if "intent" in st or "error" in st:
            stages.append(st)
        if st["answer"] is not None:
            votes.append(Vote(TEMPLATE, st["answer"]))

    # ---- stage 2: planner (one short LLM call) -------------------------------------------
    if not done() and s.use_planner and shop.sales is not None and time.time() < deadline - 60:
        plan = planner.run_plan(
            res.client,
            q,
            shop.sales,
            res.rulebook,
            shop.dimensions,
            hint_lines,
            vn_weekday(q["as_of_date"]),
            analysis.periods,
            analysis.coverage,
            analysis.expected_type,
            thinking=s.plan_thinking,
            max_tokens=s.max_tokens if s.plan_thinking else s.plan_max_tokens,
        )
        vote = _engine_vote(plan.answer) if plan.answer else None
        stages.append(
            {
                "stage": PLAN,
                "route": plan.route,
                "spec": plan.spec,
                "answer": vote,
                "evidence": plan.evidence,
                "error": plan.error,
                "raw": plan.raw if plan.spec is None else None,
                "llm_ms": plan.llm_ms,
            }
        )
        if plan.raw is not None:
            thinking.append(plan.thinking)
            timing["llm_calls"] += 1
            timing["llm_ms"] += plan.llm_ms
            timing["cached_calls"] += bool(plan.cached)
            usage.update(plan.usage or {})
        if vote is not None:
            votes.append(Vote(PLAN, vote))

    # ---- stage 3: code agent (attempts until two votes agree) ----------------------------
    attempts: list[Attempt] = []
    ctx = ExecContext(shop.ref.pickle, shop.ref.data_dir, res.calendar_path, q["as_of_date"], q["question"])
    examples = ""
    shown: list = []
    if len(res.examples) and s.few_shot > 0:
        shown = res.examples.retrieve(
            q["question"], s.few_shot, exclude_business=q["business_id"] if s.examples_holdout else None
        )
        examples = render(shown)

    def verify(ans: dict) -> list[str]:
        return check(ans, analysis, shop.index)

    # "balanced" needs room for two code votes: a question no cheap stage can answer would otherwise
    # be given up without a single attempt (e.g. an old .env that still says DATAMATE_ATTEMPTS=1)
    max_attempts = 1 if effort == "fast" else max(2, s.attempts)

    def reachable() -> bool:
        """False once even unanimous remaining attempts could not give any answer `need` votes."""
        top = tally(votes)[0][1] if votes else 0
        return top + (max_attempts - len(attempts)) >= need

    while not done() and len(attempts) < max_attempts and reachable() and time.time() < deadline - 30:
        v = len(attempts)
        msgs = build_messages(q, shop.ref.card, res.rulebook, hint_lines, variant=v, examples=examples)
        # the worker starts loading the shop now, while the model writes its first code block
        with res.sandbox.open(ctx, allow_read=[res.rulebook_path] if res.rulebook_path else None) as session:
            att = run_attempt(res.client, session, msgs, s, deadline, variant=v, verify=verify)
        attempts.append(att)
        if att.answer is not None and att.answer.get("type") != "abstain":
            votes.append(Vote(CODE, att.answer))
        stages.append(
            {
                "stage": CODE,
                "variant": v,
                "answer": att.answer,
                "llm_calls": len(att.steps),
                "error": att.error,
                "warnings": att.warnings,
                "steps": [
                    {"code": st["code"], "error": "[lỗi]" in (st["output"] or "")} for st in att.steps if st["code"]
                ],
                "llm_ms": att.llm_ms,
                "exec_ms": att.exec_ms,
            }
        )
        thinking += [st["thinking"] for st in att.steps]
        usage.update(att.usage)
        timing["llm_calls"] += len(att.steps)
        timing["llm_ms"] += att.llm_ms
        timing["exec_ms"] += att.exec_ms
        timing["cached_calls"] += att.cached_calls

    # ---- decision ------------------------------------------------------------------------
    ranked = tally(votes)
    voted, n_votes = ranked[0] if ranked else (dict(ABSTAIN), 0)
    unique = bool(ranked) and (len(ranked) == 1 or ranked[1][1] < n_votes)
    feats = features(votes, voted, attempts, analysis, verify(voted) if ranked else [], stages)
    if not ranked:
        keep = True  # nothing to submit: the record is an abstain anyway
    elif res.calibrator.fitted:
        keep = res.calibrator.keep(feats)
    else:
        keep = unique and n_votes >= need
    final = voted if keep else dict(ABSTAIN)

    steps = [st for a in attempts for st in a.steps]
    executed = [st for st in steps if st["code"]]
    log = {
        "id": q["id"],
        "answer": final,
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
        # Docker guide §3.3: raw reasoning of every model call, in call order; null if none
        "thinking": thinking if any(thinking) else None,
        "code": [st["code"] for st in executed] or None,
        "output": [st["output"] for st in executed] or None,
        "business_id": q["business_id"],
        "as_of_date": q["as_of_date"],
        "question": q["question"],
        "hints": hint_lines,
        "effort": effort,
        "stages": stages,
        "votes": [{"source": v.source, "answer": v.answer} for v in votes],
        "voted": voted,
        "n_votes": n_votes,
        # the stages behind the leading answer (two "unanswerable" votes agree whatever their reasons)
        "decided_by": sorted({v.source for v in votes if canonical(v.answer) == canonical(voted)}) if ranked else [],
        "examples": [{"id": e.id, "score": sc} for e, sc in shown] or None,
        "confidence_features": feats,
        "p_correct": res.calibrator.probability(feats),
        "timing": dict(timing),
        "usage": dict(usage),
    }
    return {"id": q["id"], **final}, log
