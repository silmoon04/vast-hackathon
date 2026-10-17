"""The planner proposes one query; it cannot choose the footage scope or execute tools."""
import copy
import importlib.util
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "find_desk_planner", Path(__file__).resolve().parents[1] / "deploy" / "find-desk" / "planner.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_TEAM", "WANDB_PROJECT", "WANDB_PROJECT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-only")  # secret-scan: allow (inert test credential)
    monkeypatch.setenv("WANDB_ENTITY", "demo-team")
    monkeypatch.setenv("WANDB_PROJECT", "demo-project")


@pytest.fixture
def payload():
    return {"model": planner.MODEL, "choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{"type": "function", "function": {
            "name": planner.TOOL, "arguments": json.dumps({"query": "adult in a blue top carrying a bag"})}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 18, "total_tokens": 118}}


def mock_response(monkeypatch, payload):
    requests = []

    def open_request(request):
        requests.append(request)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(planner, "open_planner_request", open_request)
    return requests


def search():
    return planner.plan_search("Adult in a blue top", ["carrying a bag"], "cam-a", "demo-location")


def test_valid_tool_call_uses_pinned_provider_and_literal_fields(payload, monkeypatch):
    requests = mock_response(monkeypatch, payload)
    monkeypatch.setenv("LLM_BASE_URL", "https://another-provider.invalid/v1")
    result = search()
    assert len(requests) == 1 and requests[0].full_url == planner.ENDPOINT
    headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert headers["authorization"] == "Bearer test-only"
    assert headers["openai-project"] == "demo-team/demo-project"
    body = json.loads(requests[0].data)
    assert body["model"] == result["model"] == planner.MODEL
    assert body["thinking"] == {"type": "disabled"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tool_choice"]["function"]["name"] == result["tool"] == planner.TOOL
    assert body["parallel_tool_calls"] is False and body["max_tokens"] == 256
    assert json.loads(body["messages"][1]["content"]) == {
        "description": "Adult in a blue top", "refinements": ["carrying a bag"],
        "camera_id": "cam-a", "location": "demo-location"}
    assert result["query"] == "adult in a blue top carrying a bag"
    assert result["latency_ms"] >= 0 and result["usage"]["total_tokens"] == 118


def test_tool_schema_has_no_database_scope_or_other_controls(payload, monkeypatch):
    requests = mock_response(monkeypatch, payload)
    search()
    tools = json.loads(requests[0].data)["tools"]
    assert len(tools) == 1
    schema = tools[0]["function"]["parameters"]
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"] and schema["additionalProperties"] is False


@pytest.mark.parametrize("change", ["wrong_tool", "multiple", "plain_answer", "wrong_model", "unfinished"])
def test_rejects_unsupported_or_incomplete_calls(change, payload, monkeypatch):
    choice = payload["choices"][0]
    message = choice["message"]
    if change == "wrong_tool":
        message["tool_calls"][0]["function"]["name"] = "run_sql"
    elif change == "multiple":
        message["tool_calls"].append(copy.deepcopy(message["tool_calls"][0]))
    elif change == "plain_answer":
        message.update(content="I found the person.", tool_calls=[])
        choice["finish_reason"] = "stop"
    elif change == "wrong_model":
        payload["model"] = "another-model"
    else:
        choice["finish_reason"] = "length"
    mock_response(monkeypatch, payload)
    with pytest.raises(planner.PlannerError, match="original description"):
        search()


@pytest.mark.parametrize("arguments", [
    {}, {"query": ""}, {"query": "   "}, {"query": None}, {"query": ["blue"]},
    {"query": "blue\nshirt"}, {"query": "blue\x7fshirt"}, {"query": "x" * 1601},
    {"query": "blue shirt", "metadata_filters": {"location": "other"}},
    {"query": "blue shirt", "camera_id": "other"},
    {"query": "blue shirt", "source": "s3://other/video.mp4"},
    {"query": "blue shirt", "sql": "SELECT * FROM footage"},
    {"query": "blue shirt", "model": "another-model"},
])
def test_query_validation_and_scope_cannot_be_changed(arguments, payload, monkeypatch):
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    mock_response(monkeypatch, payload)
    with pytest.raises(planner.PlannerError):
        search()


def test_duplicate_argument_keys_fail(payload, monkeypatch):
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"query":"blue","query":"red"}'
    mock_response(monkeypatch, payload)
    with pytest.raises(planner.PlannerError):
        search()


def test_missing_credentials_fail_before_a_request(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setattr(planner, "open_planner_request", lambda request: pytest.fail("no request without a key"))
    assert planner.available() is False
    with pytest.raises(planner.PlannerUnavailable, match="not configured"):
        search()


def test_project_path_override_and_optional_attribution(payload, monkeypatch):
    requests = mock_response(monkeypatch, payload)
    monkeypatch.setenv("WANDB_PROJECT_PATH", "other-team/other-project")
    search()
    assert dict((key.lower(), value) for key, value in requests[-1].header_items())["openai-project"] == "other-team/other-project"
    for name in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_PROJECT_PATH"):
        monkeypatch.delenv(name)
    assert planner.available()
    search()
    assert "openai-project" not in {key.lower() for key, value in requests[-1].header_items()}


def test_existing_team_alias_is_accepted(payload, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY")
    monkeypatch.setenv("WANDB_TEAM", "demo-team-alias")
    requests = mock_response(monkeypatch, payload)
    search()
    headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert headers["openai-project"] == "demo-team-alias/demo-project"


@pytest.mark.parametrize("name,value", [("WANDB_API_KEY", "bad\r\nkey"),
                                        ("WANDB_PROJECT_PATH", "bad\r\nproject"),
                                        ("WANDB_PROJECT_PATH", "https://another.invalid/project")])
def test_invalid_header_configuration_is_unavailable(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    assert not planner.available()
    with pytest.raises(planner.PlannerUnavailable):
        search()


def test_http_errors_are_sanitized_and_never_retried(monkeypatch):
    calls = []

    def fail(request):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, "private request and credential detail", {}, None)

    monkeypatch.setattr(planner, "open_planner_request", fail)
    with pytest.raises(planner.PlannerError) as error:
        search()
    assert len(calls) == 1
    assert "private" not in str(error.value) and "credential" not in str(error.value)
    assert "wandb" not in str(error.value) and "blue" not in str(error.value)


def test_transport_disables_proxy_and_redirects(payload, monkeypatch):
    observed = {}

    class Opener:
        def open(self, request, timeout):
            observed.update(url=request.full_url, timeout=timeout)
            return io.BytesIO(json.dumps(payload).encode())

    def build_opener(*handlers):
        observed["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    search()
    assert observed["url"] == planner.ENDPOINT and observed["timeout"] == 20
    proxy, redirect = observed["handlers"]
    assert isinstance(proxy, urllib.request.ProxyHandler) and proxy.proxies == {}
    assert isinstance(redirect, planner.NoPlannerRedirect)
    assert redirect.redirect_request(None, None, 302, "Moved", {}, "https://another.invalid") is None


def test_oversized_response_and_expired_deadline_fail(payload, monkeypatch):
    monkeypatch.setattr(planner, "open_planner_request", lambda request: io.BytesIO(b" " * 65537))
    with pytest.raises(planner.PlannerError):
        search()
    mock_response(monkeypatch, payload)
    ticks = iter([1.0, 22.0])
    monkeypatch.setattr(planner.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(planner.PlannerError):
        search()


def test_usage_is_numeric_and_does_not_expose_provider_metadata(payload, monkeypatch):
    payload["usage"] = {"prompt_tokens": True, "completion_tokens": -1,
                        "total_tokens": 118, "private": "provider detail"}
    mock_response(monkeypatch, payload)
    assert search()["usage"] == {"total_tokens": 118}
