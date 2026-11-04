"""Local OpenAI-compatible provider: Anthropic shape, tool-use translation, fallback."""

import json

import httpx
import pytest

from chatqec.providers.base import ToolUseBlock
from chatqec.providers.fallback import Backend, FallbackLLMClient
from chatqec.providers.local import LocalLLM


def _llm(reply: dict, seen: list[dict] | None = None, status: int = 200) -> LocalLLM:
    """LocalLLM over a mock transport that records request bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=reply)

    http = httpx.Client(base_url="http://local/v1", transport=httpx.MockTransport(handler))
    return LocalLLM(base_url="http://local/v1", http=http)


def _reply(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }


def test_text_reply_is_anthropic_shaped():
    seen: list[dict] = []
    resp = _llm(_reply("on_topic"), seen).messages.create(
        model="muse-glimmer",
        max_tokens=10,
        system="You are a classifier.",
        messages=[{"role": "user", "content": "What is a surface code?"}],
    )
    assert resp.content[0].text == "on_topic"
    assert resp.stop_reason == "end_turn"
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (11, 5)
    assert seen[0]["messages"][0] == {"role": "system", "content": "You are a classifier."}
    assert "tools" not in seen[0]


def test_tool_call_reply_becomes_tool_use_block():
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "stim_simulate", "arguments": '{"distance": 3, "p": 0.001}'},
    }
    seen: list[dict] = []
    resp = _llm(_reply(None, [call]), seen).messages.create(
        model="muse-glimmer",
        max_tokens=100,
        messages=[{"role": "user", "content": "simulate d=3"}],
        tools=[{"name": "stim_simulate", "description": "sim", "input_schema": {"type": "object"}}],
    )
    assert resp.stop_reason == "tool_use"
    assert resp.content == [
        ToolUseBlock(id="call_1", name="stim_simulate", input={"distance": 3, "p": 0.001})
    ]
    assert seen[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "stim_simulate",
                "description": "sim",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_tool_history_is_translated_to_openai_roles():
    seen: list[dict] = []
    _llm(_reply("done"), seen).messages.create(
        model="muse-glimmer",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "simulate d=3"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "stim_simulate",
                        "input": {"distance": 3},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": '{"ler": 0.01}'}
                ],
            },
        ],
    )
    sent = seen[0]["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": "Running.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "stim_simulate", "arguments": '{"distance": 3}'},
            }
        ],
    }
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ler": 0.01}'}


def test_malformed_arguments_are_kept_not_dropped():
    call = {"id": "c", "function": {"name": "t", "arguments": "{not json"}}
    resp = _llm(_reply(None, [call])).messages.create(model="m", max_tokens=5, messages=[])
    assert resp.content[0].input == {"_raw_arguments": "{not json"}


def test_server_error_falls_back_to_next_local_model():
    down = _llm({"error": "boom"}, status=503)
    up = _llm(_reply("from gemma"))
    client = FallbackLLMClient(
        backends=[
            Backend(name="muse-glimmer", client=down, default_model="muse-glimmer"),
            Backend(name="gemma-4", client=up, default_model="gemma-4"),
        ],
        verbose=False,
    )
    resp = client.messages.create(model="muse-glimmer", max_tokens=5, messages=[])
    assert resp.content[0].text == "from gemma"
    assert client.tripped == {"muse-glimmer"}


def test_client_error_is_not_swallowed_by_fallback():
    bad = _llm({"error": "bad schema"}, status=400)
    client = FallbackLLMClient(
        backends=[
            Backend(name="muse-glimmer", client=bad),
            Backend(name="gemma-4", client=_llm(_reply("x"))),
        ],
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="rejected request"):
        client.messages.create(model="muse-glimmer", max_tokens=5, messages=[])


def test_connection_refused_is_recoverable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    llm = LocalLLM(
        base_url="http://local/v1",
        http=httpx.Client(base_url="http://local/v1", transport=httpx.MockTransport(refuse)),
    )
    assert llm.is_up() is False
    with pytest.raises(RuntimeError, match="local LLM unavailable"):
        llm.messages.create(model="m", max_tokens=5, messages=[])


def test_reasoning_controls_are_sent():
    seen: list[dict] = []
    llm = _llm(_reply("ok"), seen)
    llm.reasoning_strength = "low"
    llm.reasoning_headroom = 100
    llm.messages.create(model="m", max_tokens=10, messages=[])
    assert seen[0]["max_tokens"] == 110
    assert seen[0]["chat_template_kwargs"] == {"reasoning_strength": "low"}


def test_local_model_roles_default_is_primary_then_fallback():
    from chatqec.cli import _local_model_roles

    cfg = {"primary_model": "muse-glimmer", "fallback_model": "gemma-4"}
    assert _local_model_roles(cfg, pin=None) == ["primary", "fallback"]


def test_local_model_roles_pin_keeps_only_that_model():
    """CHATQEC_LOCAL_MODEL pins one served model, with no silent fallback, so an
    eval or UI run is attributable to exactly one model."""
    from chatqec.cli import _local_model_roles

    cfg = {"primary_model": "muse-glimmer", "fallback_model": "gemma-4"}
    assert _local_model_roles(cfg, pin="gemma-4") == ["fallback"]
    assert _local_model_roles(cfg, pin="muse-glimmer") == ["primary"]


def test_local_model_roles_unknown_pin_is_an_error():
    import pytest

    from chatqec.cli import _local_model_roles

    cfg = {"primary_model": "muse-glimmer", "fallback_model": "gemma-4"}
    with pytest.raises(ValueError, match="gpt-x"):
        _local_model_roles(cfg, pin="gpt-x")
