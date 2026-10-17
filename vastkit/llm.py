"""W&B Inference (CoreWeave) and any other OpenAI-compatible text/VLM endpoint.

Uses the openai SDK on purpose: after ``weave.init`` every call is traced with tokens and
cost. Base URL https://api.inference.wandb.ai/v1, key = WANDB_API_KEY, and
project="<entity>/<project>" (sent as the OpenAI-Project header, used for usage attribution).
Other servers: LLM_BASE_URL + LLM_API_KEY; the W&B key and project are only sent to W&B.

Model ids change: run ``python -m vastkit models`` on the day and override with the env vars below.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from . import http as _http
from .config import Settings, settings as _settings

WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"

# Defaults from the CoreWeave serverless model list (Oct 2026); override per env var.
MODELS = {
    "agent": os.getenv("AGENT_MODEL", "openai/gpt-oss-120b"),
    "fast": os.getenv("FAST_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"),
    "vlm": os.getenv("VLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"),
    "vlm_cheap": os.getenv("VLM_CHEAP", "google/gemma-4-26B-A4B-it"),
    "judge": os.getenv("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731"),
}


def _is_loopback(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_openai(base_url: str | None = None, api_key: str | None = None, *, project: str | None = None,
                cfg: Settings | None = None, timeout: float = 120.0, max_retries: int = 5):
    """OpenAI client for W&B Inference by default; pass base_url for local/other servers.

    Key: ``api_key`` if given; else WANDB_API_KEY for W&B Inference only, and LLM_API_KEY for
    any other server ("local" for a loopback host without one). The W&B key never leaves for other hosts.
    """
    import openai
    cfg = cfg or _settings()
    base_url = base_url or os.getenv("LLM_BASE_URL") or WANDB_BASE_URL
    wandb = base_url.startswith(WANDB_BASE_URL)
    _http.ensure_no_proxy()
    if wandb:
        key = api_key or cfg.wandb_api_key
        if not key:
            raise RuntimeError("WANDB_API_KEY unset (needed for W&B Inference)")
    else:
        key = api_key or os.getenv("LLM_API_KEY") or ("local" if _is_loopback(base_url) else "")
        if not key:
            raise RuntimeError(f"no API key for {urlsplit(base_url).hostname or base_url}: pass api_key= or set "
                               "LLM_API_KEY (WANDB_API_KEY is only sent to W&B Inference)")
    kw: dict[str, Any] = {"base_url": base_url, "api_key": key, "timeout": timeout, "max_retries": max_retries}
    proj = project or cfg.wandb_project_path
    if proj and wandb:
        kw["project"] = proj
    return openai.OpenAI(**kw)


def image_part(img: str | Path | bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    """OpenAI image_url part from a path, raw bytes or an existing data URI."""
    if isinstance(img, str) and img.startswith("data:"):
        url = img
    else:
        data = img if isinstance(img, bytes) else Path(img).read_bytes()
        url = f"data:{mime};base64," + base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": url}}


class LLM:
    """Thin wrapper: chat / chat_json (strict json_schema) / vision, with usage accounting."""

    def __init__(self, client: Any = None, *, model: str | None = None, **kw: Any):
        self.client = client or make_openai(**kw)
        self.model = model or MODELS["fast"]
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def _create(self, **kw: Any):
        t0 = time.time()
        r = self.client.chat.completions.create(**kw)
        self.usage["calls"] += 1
        self.usage["seconds"] += time.time() - t0
        if getattr(r, "usage", None):
            self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += r.usage.completion_tokens or 0
        return r

    @staticmethod
    def _msgs(prompt: str | list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
        if isinstance(prompt, list):
            return ([{"role": "system", "content": system}] if system else []) + prompt
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    @staticmethod
    def _extra(think: bool | None) -> dict[str, Any] | None:
        return {"chat_template_kwargs": {"enable_thinking": think}} if think is not None else None

    def chat(self, prompt: str | list[dict[str, Any]], *, system: str | None = None, model: str | None = None,
             think: bool | None = None, **kw: Any) -> str:
        r = self._create(model=model or self.model, messages=self._msgs(prompt, system),
                         extra_body=self._extra(think), **kw)
        return r.choices[0].message.content or ""

    def chat_json(self, prompt: str | list[dict[str, Any]], schema: dict[str, Any], *, name: str = "Result",
                  system: str | None = None, model: str | None = None, think: bool | None = False,
                  strict: bool = True, **kw: Any) -> Any:
        """Structured output; falls back to parsing JSON from text if the server ignores the schema."""
        from .gpu.cosmos import extract_json
        r = self._create(model=model or self.model, messages=self._msgs(prompt, system),
                         response_format={"type": "json_schema",
                                          "json_schema": {"name": name, "strict": strict, "schema": schema}},
                         extra_body=self._extra(think), **kw)
        text = r.choices[0].message.content or ""
        try:
            return json.loads(text)
        except ValueError:
            parsed = extract_json(text)
            if parsed is None:
                raise
            return parsed

    def vision(self, question: str, images: Iterable[str | Path | bytes], *, labels: Iterable[str] | None = None,
               model: str | None = None, system: str | None = None, **kw: Any) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": question}]
        labels = list(labels or [])
        for i, img in enumerate(images):
            if i < len(labels):
                content.append({"type": "text", "text": labels[i]})
            content.append(image_part(img))
        return self.chat([{"role": "user", "content": content}], system=system, model=model or MODELS["vlm"], **kw)

    def models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)
