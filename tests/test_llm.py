"""ChatClient against a local stand-in server: request shape, thinking switch, reply parsing, disk cache."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from datamate.agent.llm import ChatClient, split_reasoning
from datamate.config import load_settings


@pytest.fixture
def server():
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            msg = {"content": f"trả lời {len(seen)}", "reasoning_content": "nghĩ"}
            if body["messages"][-1]["content"].startswith("inline"):
                msg = {"content": "<think>nghĩ trong content</think>\nxong"}
            out = {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen
    httpd.shutdown()


def test_request_and_reply(server, monkeypatch):
    url, seen = server
    monkeypatch.setenv("DATAMATE_LLM_BASE_URL", url)
    client = ChatClient(load_settings())
    assert client.wait_ready(5)
    r = client.chat([{"role": "user", "content": "hỏi"}])
    assert (r.content, r.reasoning, r.finish_reason, r.cached) == ("trả lời 1", "nghĩ", "stop", False)
    assert seen[0]["temperature"] == 0.0 and seen[0]["seed"] == 0 and seen[0]["max_tokens"] == 6144
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}
    client.chat([{"role": "user", "content": "hỏi"}], enable_thinking=False, max_tokens=512)  # the planner call
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False} and seen[1]["max_tokens"] == 512
    r = client.chat([{"role": "user", "content": "inline"}])  # server without a reasoning parser
    assert (r.content, r.reasoning) == ("xong", "nghĩ trong content")
    assert split_reasoning("không có think") == ("không có think", None)


def test_hosts_without_template_kwargs_get_the_soft_switch(server, monkeypatch):
    url, seen = server
    monkeypatch.setenv("DATAMATE_LLM_BASE_URL", url)
    monkeypatch.setenv("DATAMATE_TEMPLATE_KWARGS", "0")
    client = ChatClient(load_settings())
    client.chat([{"role": "user", "content": "hỏi"}])
    client.chat([{"role": "user", "content": "hỏi"}], enable_thinking=False)
    assert "chat_template_kwargs" not in seen[0] and seen[0]["messages"][-1]["content"] == "hỏi"
    assert seen[1]["messages"][-1]["content"] == "hỏi /no_think"


def test_disk_cache_replays_identical_requests(server, monkeypatch, tmp_path):
    url, seen = server
    monkeypatch.setenv("DATAMATE_LLM_BASE_URL", url)
    monkeypatch.setenv("DATAMATE_LLM_CACHE", str(tmp_path / "cache"))
    client = ChatClient(load_settings())
    a = client.chat([{"role": "user", "content": "một"}])
    b = client.chat([{"role": "user", "content": "một"}])
    assert (a.cached, b.cached, b.content, b.reasoning) == (False, True, a.content, a.reasoning) and len(seen) == 1
    client.chat([{"role": "user", "content": "hai"}])  # another prompt
    client.chat([{"role": "user", "content": "một"}], max_tokens=99)  # other decoding settings
    assert len(seen) == 3
    assert ChatClient(load_settings()).chat([{"role": "user", "content": "một"}]).cached  # survives a restart
    monkeypatch.delenv("DATAMATE_LLM_CACHE")
    assert not ChatClient(load_settings()).chat([{"role": "user", "content": "một"}]).cached  # off by default
