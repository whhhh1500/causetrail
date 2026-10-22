"""LLM clients behind one small interface.

* :class:`MeterlessClient` - Meterless Runtime / Nightshift. OpenAI-compatible
  ``POST {base}/chat/completions`` with a bearer token. Default model is an
  NVIDIA Nemotron model; if the configured id is not served, the client picks
  the first ``*nemotron*`` id from ``GET {base}/models``.
* :class:`CC1500Client` - development fallback (any OpenAI-compatible endpoint),
  configured from ``~/.secrets/cc1500.env`` so work is never blocked.
* :class:`NullLLM` - no model at all; the agent then uses its deterministic
  planner and template narration (fully reproducible demo).
* :class:`ScriptedLLM` - test double that answers by task name.

No keys live in code. Everything is env vars or ``~/.secrets/*.env``.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_METERLESS_URL = "http://127.0.0.1:8088/v1"
# Nemotron id in NVIDIA's catalog (the "deep lane" in the hackathon guide). Meterless model ids are routing
# modes (meterless/auto, meterless/text.deep ...) and the concrete model is picked by the Nightshift lane;
# resolve_model() keeps this id if the runtime lists it, else any *nemotron* id, else meterless/auto.
DEFAULT_NEMOTRON = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
APP_ID = "causetrail"


class LLMUnavailable(RuntimeError):
    pass


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; never prints values."""
    p = Path(path).expanduser()
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return out


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in model reply: {text[:200]!r}")


class LLMClient(ABC):
    name = "abstract"
    model = ""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0, max_tokens: int = 1200) -> str: ...

    def ask_json(self, task: str, system: str, user: str) -> Any:
        """One JSON-returning call. The ``TASK:`` header lets test doubles route."""
        reply = self.chat(
            [
                {"role": "system", "content": f"TASK: {task}\n{system}\nReply with JSON only."},
                {"role": "user", "content": user},
            ]
        )
        return extract_json(reply)

    def label(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name


class OpenAICompatClient(LLMClient):
    extra_headers: dict[str, str] = {}

    def __init__(self, base_url: str, api_key: str, model: str, *, name: str, timeout: float = 180.0):
        base = base_url.strip().rstrip("/")
        if not base.startswith("http"):
            base = "http://" + base
        for suffix in ("/chat/completions", "/models", "/healthz"):
            base = base.removesuffix(suffix)
        if not base.endswith("/v1"):
            base += "/v1"
        self.base, self.api_key, self.model, self.name, self.timeout = base, api_key, model, name, timeout
        self.calls = 0
        self.total_latency_s = 0.0

    def _req(self, path: str, body: dict | None = None) -> Any:
        req = Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            method="POST" if body is not None else "GET",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     **self.extra_headers},
        )
        try:
            with urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            raise LLMUnavailable(f"{self.name} HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from e
        except (URLError, OSError, TimeoutError) as e:
            raise LLMUnavailable(f"{self.name} unreachable at {self.base}: {getattr(e, 'reason', e)}") from e

    def list_models(self) -> list[str]:
        out = self._req("/models")
        return [m.get("id") for m in out.get("data", []) if isinstance(m, dict)]

    def chat(self, messages, *, temperature=0.0, max_tokens=1200) -> str:
        t0 = time.perf_counter()
        out = self._req(
            "/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        self.calls += 1
        self.total_latency_s += time.perf_counter() - t0
        if isinstance(out, dict) and out.get("error"):
            raise LLMUnavailable(f"{self.name} error: {out['error']}")
        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMUnavailable(f"{self.name}: unexpected reply shape {str(out)[:200]}") from e
        if isinstance(content, list):  # some runtimes return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""


class MeterlessClient(OpenAICompatClient):
    """Meterless Runtime / Nightshift (OpenAI-compatible).

    Env: METERLESS_BASE_URL       (default http://127.0.0.1:8088/v1),
         METERLESS_RUNTIME_TOKEN  (bearer; shown in Nightshift Console, written to <stateDir>/proxy-token;
                                   METERLESS_API_KEY is accepted as an alias),
         METERLESS_MODEL          (default: a Nemotron model).
    Every request carries ``meterless-app: causetrail`` and ``meterless-scope: user`` (provider contract v4).
    """

    extra_headers = {"meterless-app": APP_ID, "meterless-scope": "user"}

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        key = api_key or os.environ.get("METERLESS_RUNTIME_TOKEN") or os.environ.get("METERLESS_API_KEY", "")
        if not key:
            raise LLMUnavailable("METERLESS_RUNTIME_TOKEN is not set")
        super().__init__(
            base_url or os.environ.get("METERLESS_BASE_URL", DEFAULT_METERLESS_URL),
            key,
            model or os.environ.get("METERLESS_MODEL", DEFAULT_NEMOTRON),
            name="meterless",
        )

    def resolve_model(self) -> str:
        """Keep the configured id if served, else fall back to any Nemotron id."""
        ids = self.list_models()
        if self.model in ids:
            return self.model
        nemo = [i for i in ids if "nemotron" in i.lower()]
        if nemo:
            self.model = nemo[0]
        elif ids:
            self.model = "meterless/auto" if "meterless/auto" in ids else ids[0]
        return self.model


class NvidiaClient(OpenAICompatClient):
    """Dev fallback straight to NVIDIA's hosted Nemotron (build.nvidia.com key).
    Not a substitute for Meterless in the submission - the hackathon requires Nightshift as primary provider."""

    def __init__(self):
        env = {**load_env_file("~/.secrets/nvidia.env"), **{k: v for k, v in os.environ.items() if k.startswith("NVIDIA_")}}
        key = env.get("NVIDIA_API_KEY", "")
        if not key:
            raise LLMUnavailable("NVIDIA_API_KEY not set (env or ~/.secrets/nvidia.env)")
        super().__init__(env.get("NVIDIA_BASE_URL", NVIDIA_URL), key, env.get("NVIDIA_MODEL", DEFAULT_NEMOTRON),
                         name="nvidia")


class CC1500Client(OpenAICompatClient):
    """Dev fallback. Reads CC1500_BASE_URL / CC1500_API_KEY / CC1500_MODEL from
    the environment or ``~/.secrets/cc1500.env``."""

    def __init__(self, env_file: str = "~/.secrets/cc1500.env"):
        env = {**load_env_file(env_file), **{k: v for k, v in os.environ.items() if k.startswith("CC1500_")}}
        key = env.get("CC1500_API_KEY", "")
        if not key:
            raise LLMUnavailable(f"CC1500_API_KEY not found (env or {env_file})")
        super().__init__(
            env.get("CC1500_BASE_URL", "https://api.cc1500.top/v1"),
            key,
            env.get("CC1500_MODEL", "gemini-3.8-flash"),
            name="cc1500",
            timeout=90.0,
        )


class NullLLM(LLMClient):
    name = "none"

    def chat(self, messages, *, temperature=0.0, max_tokens=1200) -> str:
        raise LLMUnavailable("no LLM configured (deterministic mode)")


class ScriptedLLM(LLMClient):
    """Test double: ``responses[task]`` is a JSON-able value or a callable(user)->value."""

    name = "scripted"

    def __init__(self, responses: dict[str, Any | Callable[[str], Any]]):
        self.responses = responses
        self.seen: list[str] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=1200) -> str:
        task = messages[0]["content"].split("\n", 1)[0].removeprefix("TASK: ").strip()
        self.seen.append(task)
        if task not in self.responses:
            raise LLMUnavailable(f"scripted LLM has no answer for {task}")
        r = self.responses[task]
        r = r(messages[-1]["content"]) if callable(r) else r
        return r if isinstance(r, str) else json.dumps(r)


def make_llm(kind: str = "auto") -> LLMClient:
    """auto: Meterless if configured, else NVIDIA direct, else cc1500 dev fallback, else none."""
    if kind == "none":
        return NullLLM()
    if kind == "meterless":
        return MeterlessClient()
    if kind == "cc1500":
        return CC1500Client()
    if kind == "nvidia":
        return NvidiaClient()
    if kind == "auto":
        for factory in (MeterlessClient, NvidiaClient, CC1500Client):
            try:
                return factory()
            except LLMUnavailable:
                continue
        return NullLLM()
    raise ValueError(f"unknown llm {kind!r}")
