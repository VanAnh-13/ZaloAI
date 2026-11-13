"""One attempt of the code-act loop: model writes code → the attempt's sandbox session runs it →
observation goes back → … until the code calls answer(...). The session is a notebook: variables
persist between steps. A suspicious answer is sent back once for a check."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from datamate.agent.answer import AnswerError, normalize
from datamate.agent.sandbox import ExecResult, Session
from datamate.config import Settings

_CODE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)
_CODE_OPEN = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*)$", re.S)  # cut off by max_tokens

Verifier = Callable[[dict], list[str]]


@dataclass
class Attempt:
    variant: int
    answer: dict | None = None  # normalized submission object (no id); None = no valid answer
    steps: list[dict] = field(default_factory=list)  # one per LLM call
    error: str | None = None
    usage: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)  # verifier warnings on the first submitted answer
    challenged: bool = False
    exec_errors: int = 0
    llm_ms: int = 0
    exec_ms: int = 0
    cached_calls: int = 0


def extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text)
    if blocks:
        return blocks[-1].strip() or None
    m = _CODE_OPEN.search(text)
    return (m.group(1).strip() or None) if m else None


def _clip(s: str, n: int) -> str:
    s = s.rstrip()
    return s if len(s) <= n else s[: n // 2] + f"\n… (bỏ {len(s) - n} ký tự) …\n" + s[-n // 2 :]


def observation(res: ExecResult, step: int, max_chars: int, last_chance: bool) -> str:
    parts = [f"[Kết quả lượt {step}]"]
    if res.restarted:
        parts.append("(phiên đã được khởi động lại: các biến tạo ở những lượt trước không còn)")
    out = res.stdout.strip()
    parts.append("stdout:\n" + _clip(out, max_chars) if out else "stdout: (trống)")
    if res.error:
        parts.append("lỗi:\n" + _clip(res.error, 1500))
    if res.answer is None and not res.error:
        parts.append("(chưa gọi answer)")
    if last_chance:
        parts.append('Đây là lượt cuối: hãy gọi answer(...) ngay — answer("abstain") nếu chưa chắc chắn.')
    return "\n".join(parts)


def output_record(res: ExecResult) -> str:
    """What the logs keep for one execution (Docker guide §3.3 `output`)."""
    s = _clip(res.stdout, 4000)
    if res.error:
        s += f"\n[lỗi]\n{res.error}"
    if res.answer is not None:
        s += f"\n[answer] {res.answer}"
    return s.strip()


def _challenge(warnings: list[str], ans: dict) -> str:
    return (
        "Kiểm tra lại trước khi nộp:\n"
        + "\n".join(f"- {w}" for w in warnings)
        + f"\nĐáp án vừa nộp: {ans}. Nếu sau khi kiểm tra vẫn đúng, gọi lại answer(...) với đáp án đó; "
        'nếu sai thì sửa; nếu không chắc thì answer("abstain").'
    )


def _thinking(mode: str, step: int, troubled: bool) -> bool:
    """on: every step · off: never · auto: the first step, and any step after an error or a challenge
    (follow-up steps that just carry out the plan do not need a second round of reasoning)."""
    if mode == "auto":
        return step == 0 or troubled
    return mode != "off"


def run_attempt(
    client,
    session: Session,
    messages: list[dict],
    settings: Settings,
    deadline: float,
    variant: int = 0,
    verify: Verifier | None = None,
) -> Attempt:
    att = Attempt(variant)
    max_steps = settings.max_steps + (1 if verify else 0)  # room for one check
    troubled = False  # the previous step failed or was challenged
    for i in range(max_steps):
        if time.time() > deadline - 20:
            att.error = "hết thời gian"
            return att
        try:
            reply = client.chat(messages, enable_thinking=_thinking(settings.thinking_mode, i, troubled))
        except Exception as e:  # server down, timeout…: this attempt abstains
            att.error = f"llm: {e}"
            return att
        att.usage.update({k: v for k, v in reply.usage.items() if isinstance(v, int)})
        att.llm_ms += getattr(reply, "latency_ms", 0)
        att.cached_calls += bool(getattr(reply, "cached", False))
        code = extract_code(reply.content)
        step = {
            "thinking": reply.reasoning,
            "code": code,
            "output": None,
            "finish": reply.finish_reason,
            "llm_ms": getattr(reply, "latency_ms", 0),
        }
        att.steps.append(step)
        messages.append({"role": "assistant", "content": reply.content})
        if code is None:
            note = (
                "Bạn hết token trước khi viết xong code; suy nghĩ ngắn hơn."
                if reply.finish_reason == "length"
                else "Không thấy khối ```python```."
            ) + " Trả lời bằng đúng một khối ```python```."
            messages.append({"role": "user", "content": note})
            troubled = True
            continue
        res = session.run(code)
        step["output"] = output_record(res)
        step["exec_ms"] = res.elapsed_ms
        att.exec_ms += res.elapsed_ms
        att.exec_errors += bool(res.error)
        troubled = bool(res.error)
        if res.answer is not None:
            try:
                ans = normalize(res.answer)
            except AnswerError as e:
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Kết quả lượt {i + 1}]\nanswer() bị từ chối: {e}\n"
                        "Sửa lại rồi gọi answer(...) đúng định dạng.",
                    }
                )
                troubled = True
                continue
            warnings = verify(ans) if verify and not att.challenged else []
            if warnings and i < max_steps - 1:
                att.warnings, att.challenged = warnings, True
                messages.append({"role": "user", "content": _challenge(warnings, ans)})
                troubled = True
                continue
            att.answer = ans
            return att
        messages.append(
            {"role": "user", "content": observation(res, i + 1, settings.obs_max_chars, i == max_steps - 2)}
        )
    att.error = "hết số lượt mà chưa gọi answer()"
    return att
