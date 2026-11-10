"""Runtime settings.

Every knob has a fixed default (the organizers run predict.sh without arguments and
never change settings); environment variables exist only for local development.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # LLM endpoint (OpenAI-compatible: local vLLM in the container, any host during dev)
    llm_base_url: str = field(default_factory=lambda: _env("DATAMATE_LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    llm_model: str = field(default_factory=lambda: _env("DATAMATE_LLM_MODEL", "qwen3-14b"))
    llm_api_key: str = field(default_factory=lambda: _env("DATAMATE_LLM_API_KEY", "EMPTY"))
    # vLLM-specific `chat_template_kwargs`; disable for hosts that reject unknown fields
    send_template_kwargs: bool = field(default_factory=lambda: _env_bool("DATAMATE_TEMPLATE_KWARGS", True))
    llm_timeout_s: float = 600.0

    # Decoding — greedy + fixed seed for reproducibility (Rule 8)
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = field(default_factory=lambda: int(_env("DATAMATE_MAX_TOKENS", "6144")))
    enable_thinking: bool = field(default_factory=lambda: _env_bool("DATAMATE_THINKING", True))

    # Agent
    max_steps: int = 6  # LLM turns per attempt
    attempts: int = field(default_factory=lambda: int(_env("DATAMATE_ATTEMPTS", "1")))
    min_agreement: float = 0.5  # share of attempts that must agree, else abstain
    exec_timeout_s: float = 30.0
    obs_max_chars: int = 3000

    # Orchestration
    concurrency: int = field(default_factory=lambda: int(_env("DATAMATE_CONCURRENCY", "16")))
    ingest_workers: int = field(
        default_factory=lambda: int(_env("DATAMATE_INGEST_WORKERS", str(min(8, os.cpu_count() or 2))))
    )
    deadline_s: float = field(default_factory=lambda: float(_env("DATAMATE_DEADLINE_S", str(5.5 * 3600))))
    ws_dir: str = field(default_factory=lambda: _env("DATAMATE_WS", str(Path(tempfile.gettempdir()) / "datamate_ws")))

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def load_settings(**overrides) -> Settings:
    return Settings().with_overrides(**overrides)
