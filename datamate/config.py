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


def _thinking_mode(v: str) -> str:
    v = v.strip().lower()
    if v == "auto":
        return "auto"
    return "on" if v in ("1", "true", "yes", "on") else "off"


def _effort(v: str) -> str:
    v = v.strip().lower()
    return v if v in ("balanced", "fast") else "balanced"


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
    # code agent: "on" (every step), "off", or "auto" (first step + steps after an error or a challenge)
    thinking_mode: str = field(default_factory=lambda: _thinking_mode(_env("DATAMATE_THINKING", "1")))
    # dev only: folder of cached completions (greedy decoding makes identical requests repeatable)
    llm_cache_dir: str = field(default_factory=lambda: _env("DATAMATE_LLM_CACHE", ""))

    # Cascade (see agent/solver.py). Stages vote; the expensive ones run only while votes are missing.
    #   balanced: an answer needs two agreeing votes (template / planner / code attempts), else abstain
    #   fast:     the first stage that answers wins (quick development runs)
    effort: str = field(default_factory=lambda: _effort(_env("DATAMATE_EFFORT", "balanced")))
    use_templates: bool = field(default_factory=lambda: _env_bool("DATAMATE_TEMPLATES", True))  # stage 1, no LLM
    use_planner: bool = field(default_factory=lambda: _env_bool("DATAMATE_PLANNER", True))  # stage 2, one short call
    plan_thinking: bool = field(default_factory=lambda: _env_bool("DATAMATE_PLAN_THINKING", False))
    plan_max_tokens: int = 1024  # without thinking a spec is ~150 tokens

    # Code agent (stage 3)
    max_steps: int = 6  # LLM turns per attempt
    attempts: int = field(default_factory=lambda: int(_env("DATAMATE_ATTEMPTS", "3")))  # at most, per question
    few_shot: int = field(default_factory=lambda: int(_env("DATAMATE_FEWSHOT", "3")))  # solved train examples shown
    # dev only: when measuring on train, never show examples taken from the shop being asked about
    examples_holdout: bool = field(default_factory=lambda: _env_bool("DATAMATE_EXAMPLES_HOLDOUT", False))
    obs_max_chars: int = 3000

    # Sandbox
    exec_timeout_s: float = 30.0
    sandbox_guard: bool = field(default_factory=lambda: _env_bool("DATAMATE_SANDBOX_GUARD", True))
    sandbox_mem_mb: int = field(default_factory=lambda: int(_env("DATAMATE_SANDBOX_MEM_MB", "8192")))

    # Orchestration
    concurrency: int = field(default_factory=lambda: int(_env("DATAMATE_CONCURRENCY", "16")))
    ingest_workers: int = field(
        default_factory=lambda: int(_env("DATAMATE_INGEST_WORKERS", str(min(8, os.cpu_count() or 2))))
    )
    deadline_s: float = field(default_factory=lambda: float(_env("DATAMATE_DEADLINE_S", str(5.5 * 3600))))
    ws_dir: str = field(default_factory=lambda: _env("DATAMATE_WS", str(Path(tempfile.gettempdir()) / "datamate_ws")))

    @property
    def enable_thinking(self) -> bool:
        """Default for a call that does not say otherwise."""
        return self.thinking_mode != "off"

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def load_settings(**overrides) -> Settings:
    return Settings().with_overrides(**overrides)
