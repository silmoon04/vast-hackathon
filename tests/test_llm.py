import os
import uuid

import pytest

pytest.importorskip("openai")

from vastkit.config import Settings  # noqa: E402
from vastkit.llm import LLM, WANDB_BASE_URL, make_openai  # noqa: E402

OTHER = "http://10.0.0.5:8000/v1"


def _fake() -> str:
    return f"fake-{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(var, raising=False)
    for var in ("NO_PROXY", "no_proxy"):  # make_openai appends the W&B hosts; restore afterwards
        monkeypatch.setenv(var, os.environ.get(var, ""))


def test_wandb_key_and_project_go_to_wandb_only():
    wb = _fake()
    cfg = Settings(wandb_api_key=wb, wandb_entity="ent", wandb_project="proj")
    c = make_openai(cfg=cfg)
    assert str(c.base_url).rstrip("/") == WANDB_BASE_URL and c.api_key == wb and c.project == "ent/proj"
    with pytest.raises(RuntimeError, match="LLM_API_KEY") as e:
        make_openai(base_url=OTHER, cfg=cfg)
    assert wb not in str(e.value)


def test_other_hosts_use_llm_api_key(monkeypatch):
    other = _fake()
    monkeypatch.setenv("LLM_API_KEY", other)
    cfg = Settings(wandb_api_key=_fake(), wandb_entity="ent", wandb_project="proj")
    c = make_openai(base_url=OTHER, cfg=cfg)
    assert c.api_key == other and c.project is None
    assert make_openai(base_url="http://127.0.0.1:8001/v1", cfg=cfg).api_key == other
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):  # LLM_API_KEY is never sent to W&B
        make_openai(cfg=Settings())


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001/v1", "http://localhost:8001/v1", "http://[::1]:8001/v1",
                                 "http://cosmos.localhost/v1"])
def test_loopback_uses_local_placeholder(url):
    assert make_openai(base_url=url, cfg=Settings(wandb_api_key=_fake())).api_key == "local"


def test_loopback_lookalike_needs_a_key():
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        make_openai(base_url="http://localhost.example.com/v1", cfg=Settings(wandb_api_key=_fake()))


def test_llm_base_url_env_and_explicit_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    assert make_openai(cfg=Settings(wandb_api_key=_fake())).api_key == "local"
    k = _fake()
    assert make_openai(base_url=OTHER, api_key=k, cfg=Settings()).api_key == k


def test_docstring_claims_no_tool_loop():
    assert "tool" not in (LLM.__doc__ or "").lower()
