"""OpenAI-compatible chat client over urllib (no extra dependency).

Talks to the local vLLM server inside the container, or to any hosted
OpenAI-compatible endpoint during development (set DATAMATE_LLM_*).
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from datamate.config import Settings


@dataclass
class Reply:
    content: str
    reasoning: str | None  # raw thinking as generated, logged verbatim
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)


def split_reasoning(content: str) -> tuple[str, str | None]:
    """Fallback when the server has no reasoning parser: take <think>…</think> out of the content."""
    if "</think>" not in content:
        return content, None
    head, _, tail = content.partition("</think>")
    return tail.strip(), head.replace("<think>", "").strip()


class ChatClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.url = settings.llm_base_url.rstrip("/")

    def chat(self, messages: list[dict], enable_thinking: bool | None = None) -> Reply:
        s = self.s
        body = {
            "model": s.llm_model,
            "messages": messages,
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
            "seed": s.seed,
        }
        if s.send_template_kwargs:
            body["chat_template_kwargs"] = {
                "enable_thinking": s.enable_thinking if enable_thinking is None else enable_thinking
            }
        data = self._post("/chat/completions", body)
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        # vLLM <= 0.10 uses reasoning_content, newer servers use reasoning
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning is None:
            content, reasoning = split_reasoning(content)
        return Reply(content, reasoning, choice.get("finish_reason"), data.get("usage") or {})

    def wait_ready(self, timeout_s: float) -> bool:
        """Poll GET /models until the server answers (vLLM may still be loading)."""
        end = time.time() + timeout_s
        while time.time() < end:
            try:
                req = urllib.request.Request(self.url + "/models", headers=self._headers())
                with urllib.request.urlopen(req, timeout=5):
                    return True
            except (urllib.error.URLError, OSError):
                time.sleep(3)
        return False

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.s.llm_api_key}"}

    def _post(self, path: str, body: dict, retries: int = 3) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode("utf-8"), headers=self._headers(), method="POST"
        )
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.s.llm_timeout_s) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code < 500 and e.code != 429:
                    raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
                last = RuntimeError(f"LLM HTTP {e.code}: {detail}")
            except (socket.timeout, TimeoutError) as e:
                raise RuntimeError(f"LLM timeout after {self.s.llm_timeout_s:.0f}s") from e
            except (urllib.error.URLError, ConnectionError) as e:
                last = e
            time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"LLM request failed: {last}")


class MockClient:
    """Offline stand-in for development without a GPU: looks at the tables, then abstains.
    Exercises ingest, prompt building, sandbox and output writing end to end."""

    def chat(self, messages: list[dict], enable_thinking: bool | None = None) -> Reply:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn == 0:
            code = (
                "for name, df in T.items():\n    print(name, df.shape)\nprint(AS_OF, DATA_END)\n"
                "print('S.revenue:', None if S is None else S.revenue())"
            )
        else:
            code = 'answer("abstain")'
        return Reply(f"```python\n{code}\n```", f"(mock) bước {turn + 1}", "stop", {})

    def wait_ready(self, timeout_s: float) -> bool:
        return True
